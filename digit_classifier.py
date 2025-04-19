# digit_classifier.py
import os
os.environ["KERAS_BACKEND"] = "torch" # Or "tensorflow"

import cv2
import numpy as np
import keras
from keras import layers, models
import torch
from pathlib import Path
import random
import math
from sklearn.model_selection import train_test_split # Still useful for initial weight calc split
from sklearn.utils import class_weight
import gc # Garbage collector

# Import necessary components from other modules
from sudoku_renderer import SudokuRenderer
from digit_extractor import rectify_grid, split_into_cells

# --- Constants ---
MODEL_FILENAME = "sudoku_digit_classifier_cnn_v3.keras" # New model name for generator training
MODEL_INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 11
EMPTY_LABEL = 10
# Training parameters adjusted for generator use
# Define epoch length in terms of batches/steps
STEPS_PER_EPOCH = 500  # Number of batches per "epoch"
EPOCHS = 30            # Total number of "epochs" (cycles of STEPS_PER_EPOCH batches)
BATCH_SIZE = 128       # Number of cell images per batch
VALIDATION_STEPS = 100 # Number of batches for validation per epoch
GRID_SIZE = 9
TARGET_CELL_CONTENT_SIZE = 20

# Helper function for data generation (can be inside class or outside)
def sudoku_data_generator(renderer, batch_size, preprocess_func, input_size):
    """
    Yields batches of (processed_cells, labels) generated on the fly.
    """
    while True: # Loop indefinitely
        batch_cells_processed = []
        batch_labels = []

        while len(batch_cells_processed) < batch_size:
            # Generate a new Sudoku image
            allow_empty = random.random() < 0.8
            rendered_img, gt_grid, warped_corners = renderer.render_sudoku(allow_empty=allow_empty)

            if rendered_img is None or warped_corners is None:
                # print("[Gen Warn] Renderer failed. Skipping.")
                continue

            # Rectify using known corners
            try:
                rectified_grid = rectify_grid(rendered_img, warped_corners)
            except Exception:
                # print(f"[Gen Warn] Rectify failed: {e}. Skipping.")
                continue

            # Split into cells
            extracted_cells, _ = split_into_cells(rectified_grid)

            if len(extracted_cells) != GRID_SIZE * GRID_SIZE:
                # print(f"[Gen Warn] Split failed. Skipping.")
                continue

            gt_labels_flat = gt_grid.flatten()

            for i, cell_img in enumerate(extracted_cells):
                if len(batch_cells_processed) >= batch_size:
                    break # Stop if batch is full

                label = gt_labels_flat[i]
                if label == 0: label = EMPTY_LABEL

                processed_cell = preprocess_func(cell_img) # Call the preprocessing method

                if processed_cell.shape != input_size:
                     # This check should ideally be unnecessary if preprocess_func is correct
                     print(f"[Gen ERROR] Preprocessed cell shape mismatch: {processed_cell.shape} vs {input_size}. Skipping.")
                     continue

                batch_cells_processed.append(processed_cell)
                batch_labels.append(label)

        # Convert batch to NumPy arrays
        X_batch = np.array(batch_cells_processed, dtype='float32')
        y_batch = np.array(batch_labels, dtype='int64')

        # Add channel dimension
        X_batch = np.expand_dims(X_batch, -1)

        # print(f"[Debug Gen] Yielding batch X:{X_batch.shape}, y:{y_batch.shape}") # Debug
        yield X_batch, y_batch
        # Explicitly delete large arrays to potentially help memory
        del X_batch, y_batch, batch_cells_processed, batch_labels
        gc.collect()


class DigitClassifier:
    def __init__(self, model_path=None, training_required=False):
        self.model_path = Path(model_path or Path(__file__).parent / MODEL_FILENAME)
        self.model = None
        self._model_input_size = MODEL_INPUT_SHAPE[:2]

        if not training_required and self.model_path.exists():
            print(f"Loading existing model from: {self.model_path}")
            try:
                self.model = keras.saving.load_model(self.model_path)
                if self.model.input_shape[1:3] != self._model_input_size:
                     print(f"[Warning] Loaded model input shape {self.model.input_shape[1:3]} "
                           f"differs from expected {self._model_input_size}.")
                print("Model loaded successfully.")
            except Exception as e:
                print(f"[Error] Failed to load model: {e}. Will attempt training.")
                self.model = None
        else:
            if training_required: print("Training explicitly required.")
            else: print(f"Model not found at {self.model_path}. Training is required.")

    # --- _preprocess_cell_for_model remains the same as the previous corrected version ---
    def _preprocess_cell_for_model(self, cell_image, is_training=False):
        """
        Prepares a single extracted cell image for model input (28x28).
        (Implementation from previous response - verified to return 28x28)
        """
        target_h, target_w = self._model_input_size # Should be (28, 28)

        if cell_image is None or cell_image.size < 10: # Ignore very small cells
            return np.zeros((target_h, target_w), dtype=np.float32)

        if cell_image.ndim == 3: gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        else: gray = cell_image.copy()

        cell_h_orig, cell_w_orig = gray.shape[:2]
        min_dim = min(cell_h_orig, cell_w_orig)
        block_size = min(19, min_dim - 1 if min_dim % 2 == 0 else min_dim - 2)
        block_size = max(3, block_size)

        try:
            thresh = cv2.adaptiveThreshold(gray, 255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV,
                                           block_size, 5)
        except cv2.error:
             return np.zeros((target_h, target_w), dtype=np.float32)

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
                except cv2.error:
                    pass

        processed = final_canvas.astype("float32") / 255.0
        return processed

    # --- _build_cnn_model remains the same as the previous version ---
    def _build_cnn_model(self):
        """Defines and compiles the CNN architecture with augmentation."""
        augment = keras.Sequential([
            layers.RandomRotation(0.1, fill_mode="constant", fill_value=0.0),
            layers.RandomTranslation(0.1, 0.1, fill_mode="constant", fill_value=0.0),
            layers.RandomZoom(0.1, 0.1, fill_mode="constant", fill_value=0.0),
        ], name="augmentation")

        model = models.Sequential([
            layers.Input(shape=MODEL_INPUT_SHAPE),
            augment,
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(NUM_CLASSES, activation='softmax')
        ])
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        print("CNN Model Summary:")
        model.summary()
        return model

    def train(self, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, batch_size=BATCH_SIZE, validation_steps=VALIDATION_STEPS):
        """
        Trains the classifier using a generator for synthetic data.
        """
        print(f"\n--- Starting Training (v3 - Generator) ---")
        print(f"Epochs: {epochs}, Steps/Epoch: {steps_per_epoch}, Batch Size: {batch_size}")

        # --- Calculate Class Weights (Estimate or Sample) ---
        # Option 1: Estimate based on renderer defaults (e.g., ~40% empty)
        # Option 2: Generate a large sample batch first to calculate more accurately
        print("Calculating class weights from initial sample...")
        temp_renderer = SudokuRenderer() # Use a temporary renderer instance
        # Generate ~50 batches to estimate weights
        initial_sample_size = batch_size * 50
        sample_labels = []
        temp_gen = sudoku_data_generator(temp_renderer, batch_size, self._preprocess_cell_for_model, self._model_input_size)
        while len(sample_labels) < initial_sample_size:
             _, y_sample_batch = next(temp_gen)
             sample_labels.extend(y_sample_batch.tolist())
        del temp_gen
        del temp_renderer
        gc.collect()

        sample_labels = np.array(sample_labels[:initial_sample_size]) # Trim if overshot
        unique_labels_sample, counts_sample = np.unique(sample_labels, return_counts=True)

        if len(unique_labels_sample) < NUM_CLASSES:
             print("[Warning] Initial sample for weight calculation might be missing some classes!")
             # Add missing classes with a count of 1 to avoid division by zero
             all_classes = np.arange(NUM_CLASSES) # 0 to 10
             present_classes = dict(zip(unique_labels_sample, counts_sample))
             full_counts = [present_classes.get(cls, 1) for cls in all_classes] # Use 1 if missing
             unique_labels_sample = all_classes
             counts_sample = np.array(full_counts)


        weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=unique_labels_sample, # Should be 0-10
            y=sample_labels # Use the sampled labels
        )
        class_weights_dict = dict(zip(unique_labels_sample, weights))
        print(f"Calculated Class Weights: {class_weights_dict}")
        if len(class_weights_dict) < NUM_CLASSES:
             print("[Error] Failed to calculate weights for all classes. Check sampling.")
             # Fallback: Use equal weights? Or abort? Abort is safer.
             print("Aborting training due to class weight calculation error.")
             return
        # --- End Class Weight Calculation ---


        # Create separate renderers for train/val if desired (e.g., different seeds)
        # For simplicity, we use the same renderer settings here.
        train_renderer = SudokuRenderer()
        val_renderer = SudokuRenderer() # Could use different params/seed

        # Create generators
        train_generator = sudoku_data_generator(
            train_renderer, batch_size, self._preprocess_cell_for_model, self._model_input_size
        )
        val_generator = sudoku_data_generator(
            val_renderer, batch_size, self._preprocess_cell_for_model, self._model_input_size
        )

        # Build model
        self.model = self._build_cnn_model()

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(self.model_path, monitor='val_loss', save_best_only=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
        ]

        # Train using the generator
        print("\nStarting model fitting with generator...")
        try:
            history = self.model.fit(
                train_generator,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=val_generator,
                validation_steps=validation_steps, # How many batches to draw from val_generator
                callbacks=callbacks,
                class_weight=class_weights_dict,
                verbose=1
            )
        except Exception as e:
             print(f"\n[Error] An exception occurred during model.fit with generator: {e}")
             import traceback
             traceback.print_exc()
             print("Aborting training.")
             return

        print("\nTraining finished.")
        # Load best weights saved by checkpoint
        if self.model_path.exists():
             print(f"Reloading best model weights from {self.model_path}")
             try:
                 # Load the model saved by the checkpoint
                 self.model = keras.saving.load_model(self.model_path)
             except Exception as e:
                  print(f"[Error] Failed to reload best model: {e}. Using final weights (might be suboptimal).")
                  # If loading fails, self.model still holds the model from the end of fit()
        else:
             print("[Warning] Best model checkpoint file not found. Using final weights.")

        # Evaluate the final model (best loaded or last) using the validation generator
        if self.model:
            print("Evaluating final model on validation generator...")
            try:
                # Important: Create a fresh validation generator instance for evaluation
                eval_val_generator = sudoku_data_generator(
                    val_renderer, batch_size, self._preprocess_cell_for_model, self._model_input_size
                )
                loss, accuracy = self.model.evaluate(
                    eval_val_generator,
                    steps=validation_steps, # Use the same number of steps as during training validation
                    verbose=0
                )
                print(f"\nFinal Validation Loss: {loss:.4f}")
                print(f"Final Validation Accuracy: {accuracy:.4f}")
                del eval_val_generator # Clean up generator
            except Exception as e:
                 print(f"[Error] Failed to evaluate model: {e}")
        else:
            print("[Error] Model object is None after training. Cannot evaluate.")
            return

        # Explicitly save the final best model (might be redundant but safe)
        if self.model:
            try:
                self.model.save(self.model_path)
                print(f"Final best model saved successfully to {self.model_path}")
            except Exception as e:
                print(f"[Error] Failed to save the final model: {e}")


    # --- recognise method remains the same as the previous corrected version ---
    @torch.no_grad()
    def recognise(self, cell_image, confidence_threshold=0.7):
        """
        Recognises the digit in a single cell image.
        (Implementation from previous response)
        """
        if self.model is None:
            print("[Error] Model not loaded or trained. Cannot recognise.")
            return 0, 0.0

        processed_cell = self._preprocess_cell_for_model(cell_image, is_training=False)

        if processed_cell.shape != self._model_input_size:
             print(f"[ERROR in recognise] Preprocessed cell has wrong shape: {processed_cell.shape}. Expected {self._model_input_size}. Returning 0.")
             return 0, 0.0

        model_input = np.expand_dims(processed_cell, axis=(0, -1))

        if keras.backend.backend() == 'torch':
            try:
                model_input_tensor = torch.from_numpy(model_input).float()
                # Add GPU transfer if applicable
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

        if isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.numpy() # Add .cpu() if needed

        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])

        if predicted_class == EMPTY_LABEL:
            return 0, confidence

        if confidence < confidence_threshold:
            return 0, confidence

        return predicted_class, confidence

# --- Example Usage (for training) ---
if __name__ == "__main__":
    print("Testing DigitClassifier training (v3 - Generator)...")
    force_train = False # Set to True to force retraining
    model_file = Path(MODEL_FILENAME) # Use new filename
    if force_train and model_file.exists():
        print("Forcing retraining, removing existing model file...")
        try:
            model_file.unlink()
        except OSError as e:
            print(f"Error removing model file: {e}")

    classifier = DigitClassifier() # Will load or announce training needed

    if classifier.model is None: # Check if training is needed
        # Train using the generator approach with default steps/epochs
        classifier.train()
    else:
        print("Model already exists and loaded. Skipping training.")
        # Optional: Test recognition
        # test_cell_path = "extracted_cells/cell_0_0.png"
        # if Path(test_cell_path).exists():
        #      test_cell = cv2.imread(test_cell_path)
        #      if test_cell is not None:
        #          digit, conf = classifier.recognise(test_cell)
        #          print(f"Test recognition on {test_cell_path}: Digit={digit}, Conf={conf:.3f}")

    print("\nClassifier test complete.")