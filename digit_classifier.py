# digit_classifier.py
import os
os.environ["KERAS_BACKEND"] = "torch"

import cv2
import numpy as np
import keras
from keras import layers, models
import torch
from pathlib import Path
import random
import math
# No longer need train_test_split or class_weight from sklearn
import gc

from sudoku_renderer import SudokuRenderer
from digit_extractor import rectify_grid, split_into_cells

# --- Constants ---
GRID_SIZE = 9
MODEL_FILENAME = "sudoku_digit_classifier_cnn_v4.keras" # v4 for oversampling
MODEL_INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 11
EMPTY_LABEL = 10
STEPS_PER_EPOCH = 10
EPOCHS = 10 # Might need fewer epochs if oversampling converges faster
BATCH_SIZE = 128
VALIDATION_STEPS = 100
TARGET_CELL_CONTENT_SIZE = 20
# --- New Constant for Oversampling ---
# Target ratio of non-empty (digits) to empty samples in a batch.
# e.g., 1.0 means aim for 50% digits, 50% empty.
# Higher value means more digits relative to empty.
TARGET_DIGIT_RATIO = 1.5 # Aim for roughly 1.5 digits per 1 empty cell

# --- Modified Data Generator with Oversampling/Undersampling ---
def sudoku_data_generator(renderer, batch_size, preprocess_func, input_size, target_digit_ratio=TARGET_DIGIT_RATIO):
    """
    Yields batches of (processed_cells, labels) generated on the fly,
    attempting to balance classes by oversampling digits / undersampling empty cells.
    """
    grid_size_sq = GRID_SIZE * GRID_SIZE
    target_num_digits = int(batch_size * (target_digit_ratio / (target_digit_ratio + 1)))
    target_num_empty = batch_size - target_num_digits

    while True:
        batch_cells_processed = []
        batch_labels = []
        num_digits_in_batch = 0
        num_empty_in_batch = 0

        # Keep generating Sudokus until the batch targets are met
        while num_digits_in_batch < target_num_digits or num_empty_in_batch < target_num_empty:
            # Generate a new Sudoku image
            # Generate more filled grids to get digits faster? Maybe not necessary.
            allow_empty = random.random() < 0.8
            rendered_img, gt_grid, warped_corners = renderer.render_sudoku(allow_empty=allow_empty)

            if rendered_img is None or warped_corners is None: continue
            try:
                rectified_grid = rectify_grid(rendered_img, warped_corners)
            except Exception: continue
            extracted_cells, _ = split_into_cells(rectified_grid)
            if len(extracted_cells) != grid_size_sq: continue

            gt_labels_flat = gt_grid.flatten()

            # Process cells from this Sudoku and add selectively to batch
            indices = list(range(grid_size_sq))
            random.shuffle(indices) # Process cells in random order

            for i in indices:
                cell_img = extracted_cells[i]
                label = gt_labels_flat[i]
                is_empty = (label == 0)
                label = EMPTY_LABEL if is_empty else label # Assign final label

                # Decide whether to add this cell based on targets
                can_add_digit = not is_empty and num_digits_in_batch < target_num_digits
                can_add_empty = is_empty and num_empty_in_batch < target_num_empty

                if can_add_digit or can_add_empty:
                    processed_cell = preprocess_func(cell_img)
                    if processed_cell.shape != input_size:
                        print(f"[Gen ERROR] Preprocessed cell shape mismatch: {processed_cell.shape} vs {input_size}. Skipping cell.")
                        continue

                    batch_cells_processed.append(processed_cell)
                    batch_labels.append(label)

                    if is_empty: num_empty_in_batch += 1
                    else: num_digits_in_batch += 1

                # Early exit if batch is full (both targets met)
                if num_digits_in_batch >= target_num_digits and num_empty_in_batch >= target_num_empty:
                    break

        # Trim batch exactly to batch_size if we overshot slightly
        batch_cells_processed = batch_cells_processed[:batch_size]
        batch_labels = batch_labels[:batch_size]

        # Shuffle the final batch
        final_indices = np.random.permutation(len(batch_labels))
        X_batch = np.array(batch_cells_processed, dtype='float32')[final_indices]
        y_batch = np.array(batch_labels, dtype='int64')[final_indices]

        X_batch = np.expand_dims(X_batch, -1)

        # Debug: Check batch balance
        # unique_final, counts_final = np.unique(y_batch, return_counts=True)
        # print(f"[Debug Gen] Yielding Batch - Labels: {dict(zip(unique_final, counts_final))}")

        yield X_batch, y_batch
        del X_batch, y_batch, batch_cells_processed, batch_labels
        gc.collect()


# --- Digit Classifier Class ---
class DigitClassifier:
    def __init__(self, model_path=None, training_required=False):
        self.model_path = Path(model_path or Path(__file__).parent / MODEL_FILENAME)
        self.model = None
        self._model_input_size = MODEL_INPUT_SHAPE[:2]

        if not training_required and self.model_path.exists():
            print(f"Loading existing model from: {self.model_path}")
            try:
                self.model = keras.saving.load_model(self.model_path)
                if hasattr(self.model, 'input_shape') and self.model.input_shape[1:3] != self._model_input_size:
                     print(f"[Warning] Loaded model input shape {self.model.input_shape[1:3]} "
                           f"differs from expected {self._model_input_size}.")
                print("Model loaded successfully.")
            except Exception as e:
                print(f"[Error] Failed to load model: {e}. Will attempt training.")
                self.model = None
        else:
            if training_required: print("Training explicitly required.")
            else: print(f"Model not found at {self.model_path}. Training is required.")

    # _preprocess_cell_for_model remains the same (corrected version from previous step)
    def _preprocess_cell_for_model(self, cell_image, is_training=False):
        target_h, target_w = self._model_input_size
        if cell_image is None or cell_image.size < 10:
            return np.zeros((target_h, target_w), dtype=np.float32)
        if cell_image.ndim == 3: gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        else: gray = cell_image.copy()
        cell_h_orig, cell_w_orig = gray.shape[:2]
        min_dim = min(cell_h_orig, cell_w_orig)
        block_size = min(19, min_dim - 1 if min_dim % 2 == 0 else min_dim - 2)
        block_size = max(3, block_size)
        try:
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, 5)
        except cv2.error: return np.zeros((target_h, target_w), dtype=np.float32)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digit_contour = None
        max_area = 0
        if contours:
            for cnt in contours:
                area = cv2.contourArea(cnt)
                x, y, w, h = cv2.boundingRect(cnt)
                if area > 5 and area > max_area and (x > 0 and y > 0 and x+w < thresh.shape[1] and y+h < thresh.shape[0]):
                     max_area = area
                     digit_contour = cnt
        final_canvas = np.zeros((target_h, target_w), dtype=np.uint8)
        if digit_contour is not None:
            x, y, w, h = cv2.boundingRect(digit_contour)
            digit_roi = thresh[y:y+h, x:x+w]
            target_content_size = TARGET_CELL_CONTENT_SIZE
            scale = min(target_content_size / w, target_content_size / h) if h > 0 and w > 0 else 0
            new_w, new_h = int(w * scale), int(h * scale)
            if new_w > 0 and new_h > 0:
                try:
                    resized_roi = cv2.resize(digit_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    pad_top = (target_h - new_h) // 2
                    pad_left = (target_w - new_w) // 2
                    end_y = min(pad_top + new_h, target_h)
                    end_x = min(pad_left + new_w, target_w)
                    roi_h_slice = end_y - pad_top
                    roi_w_slice = end_x - pad_left
                    final_canvas[pad_top:end_y, pad_left:end_x] = resized_roi[:roi_h_slice, :roi_w_slice]
                except cv2.error: pass
        processed = final_canvas.astype("float32") / 255.0
        return processed

    # _build_cnn_model remains the same
    def _build_cnn_model(self):
        augment = keras.Sequential([
            layers.RandomRotation(0.1, fill_mode="constant", fill_value=0.0),
            layers.RandomTranslation(0.1, 0.1, fill_mode="constant", fill_value=0.0),
            layers.RandomZoom(0.1, 0.1, fill_mode="constant", fill_value=0.0),
        ], name="augmentation")
        model = models.Sequential([
            layers.Input(shape=MODEL_INPUT_SHAPE), augment,
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'), layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'), layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)), layers.Dropout(0.25),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'), layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'), layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)), layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation='relu'), layers.BatchNormalization(), layers.Dropout(0.5),
            layers.Dense(NUM_CLASSES, activation='softmax')
        ])
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print("CNN Model Summary:")
        model.summary()
        return model

    # Modified train method - removed class_weight calculation and argument
    def train(self, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, batch_size=BATCH_SIZE, validation_steps=VALIDATION_STEPS):
        """
        Trains the classifier using a generator for synthetic data with oversampling.
        """
        print(f"\n--- Starting Training (v4 - Generator with Oversampling) ---")
        print(f"Epochs: {epochs}, Steps/Epoch: {steps_per_epoch}, Batch Size: {batch_size}")

        # No need to calculate class weights beforehand when using oversampling generator

        train_renderer = SudokuRenderer()
        val_renderer = SudokuRenderer()

        # Create generators - they now handle the balancing internally
        train_generator = sudoku_data_generator(
            train_renderer, batch_size, self._preprocess_cell_for_model, self._model_input_size
        )
        val_generator = sudoku_data_generator(
            val_renderer, batch_size, self._preprocess_cell_for_model, self._model_input_size
        )

        self.model = self._build_cnn_model()

        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(self.model_path, monitor='val_loss', save_best_only=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
        ]

        print("\nStarting model fitting with oversampling generator...")
        try:
            history = self.model.fit(
                train_generator,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=val_generator,
                validation_steps=validation_steps,
                callbacks=callbacks,
                # class_weight argument is removed here
                verbose=1
            )
        except Exception as e:
             print(f"\n[Error] An exception occurred during model.fit with generator: {e}")
             import traceback
             traceback.print_exc()
             print("Aborting training.")
             del train_generator, val_generator; gc.collect()
             return

        print("\nTraining finished.")
        if self.model_path.exists():
             print(f"Reloading best model weights from {self.model_path}")
             try: self.model = keras.saving.load_model(self.model_path)
             except Exception as e: print(f"[Error] Failed to reload best model: {e}.")
        else: print("[Warning] Best model checkpoint file not found.")

        if self.model:
            print("Evaluating final model on validation generator...")
            try:
                eval_val_generator = sudoku_data_generator(val_renderer, batch_size, self._preprocess_cell_for_model, self._model_input_size)
                loss, accuracy = self.model.evaluate(eval_val_generator, steps=validation_steps, verbose=0)
                print(f"\nFinal Validation Loss: {loss:.4f}")
                print(f"Final Validation Accuracy: {accuracy:.4f}")
                del eval_val_generator
            except Exception as e: print(f"[Error] Failed to evaluate model: {e}")
        else: print("[Error] Model object is None after training.")

        if self.model:
            try:
                self.model.save(self.model_path)
                print(f"Final best model saved successfully to {self.model_path}")
            except Exception as e: print(f"[Error] Failed to save the final model: {e}")

        del train_generator, val_generator; gc.collect()


    # Modified recognise method - added .cpu() for MPS tensor conversion
    @torch.no_grad()
    def recognise(self, cell_image, confidence_threshold=0.7):
        """
        Recognises the digit in a single cell image. Handles MPS tensor conversion.
        """
        if self.model is None:
            print("[Error] Model not loaded or trained.")
            return 0, 0.0

        processed_cell = self._preprocess_cell_for_model(cell_image, is_training=False)
        if processed_cell.shape != self._model_input_size:
             print(f"[ERROR in recognise] Preprocessed cell shape mismatch: {processed_cell.shape}. Expected {self._model_input_size}.")
             return 0, 0.0

        model_input = np.expand_dims(processed_cell, axis=(0, -1))

        if keras.backend.backend() == 'torch':
            try:
                # Assume model might be on MPS, move input tensor there if needed
                # device = next(self.model.parameters()).device # Get model device (e.g., 'mps:0')
                # model_input_tensor = torch.from_numpy(model_input).float().to(device)
                # Simpler: Assume model handles device placement or runs on default
                 model_input_tensor = torch.from_numpy(model_input).float()
            except Exception as e:
                 print(f"[Error] Failed converting NumPy to Torch tensor: {e}")
                 return 0, 0.0
        else:
             model_input_tensor = model_input

        try:
            probabilities = self.model(model_input_tensor, training=False)[0]
        except Exception as e:
             print(f"[Error] Exception during model prediction: {e}")
             return 0, 0.0

        # --- FIX for MPS Tensor Conversion ---
        if isinstance(probabilities, torch.Tensor):
            # Check if tensor is on MPS or other non-CPU device
            if probabilities.device.type != 'cpu':
                 probabilities = probabilities.cpu() # Move to CPU
            # Detach if necessary (usually not for inference with no_grad)
            # probabilities = probabilities.detach()
            probabilities = probabilities.numpy() # Convert to NumPy
        # --- End FIX ---

        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])

        if predicted_class == EMPTY_LABEL: return 0, confidence
        if confidence < confidence_threshold: return 0, confidence
        return predicted_class, confidence

# --- Example Usage (for training) ---
if __name__ == "__main__":
    print("Testing DigitClassifier training (v4 - Oversampling Generator)...")
    force_train = False
    model_file = Path(MODEL_FILENAME) # Use v4 filename
    if force_train and model_file.exists():
        print("Forcing retraining, removing existing model file...")
        try: model_file.unlink()
        except OSError as e: print(f"Error removing model file: {e}")

    classifier = DigitClassifier()

    if classifier.model is None:
        classifier.train() # Use defaults
    else:
        print("Model already exists and loaded. Skipping training.")

    print("\nClassifier test complete.")