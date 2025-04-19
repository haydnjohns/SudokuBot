# digit_classifier.py
import os
os.environ["KERAS_BACKEND"] = "torch"

import cv2
import numpy as np
import keras
from keras import layers, models, callbacks # Import callbacks
import torch
from pathlib import Path
import random
import math
import gc

from sudoku_renderer import SudokuRenderer, generate_and_save_test_example # Import the new function
from digit_extractor import extract_cells_from_image, rectify_grid, split_into_cells, GRID_SIZE # Import GRID_SIZE too
# Import print_sudoku_grid for use in the callback
from sudoku_recogniser import print_sudoku_grid, FINAL_CONFIDENCE_THRESHOLD

# --- Constants ---
MODEL_FILENAME = "sudoku_digit_classifier_cnn_v6.keras" # v6 for callback test
MODEL_INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 11
EMPTY_LABEL = 10
EPOCHS = 40          # Keep increased training time (EarlyStopping will manage)
STEPS_PER_EPOCH = 150 # Keep increased
BATCH_SIZE = 128
VALIDATION_STEPS = 50 # Keep increased
TARGET_CELL_CONTENT_SIZE = 24 # Keep adjusted size
TARGET_DIGIT_RATIO = 1.5 # Keep ratio

# --- Data Generator ---
def sudoku_data_generator(renderer, batch_size, preprocess_func, input_size, target_digit_ratio=TARGET_DIGIT_RATIO):
    """
    Yields batches of (processed_cells, labels) generated on the fly,
    attempting to balance classes by oversampling digits / undersampling empty cells.
    """
    grid_size_sq = GRID_SIZE * GRID_SIZE
    target_num_digits = int(batch_size * (target_digit_ratio / (target_digit_ratio + 1)))
    target_num_empty = batch_size - target_num_digits
    input_shape_no_channel = input_size[:2] # e.g., (28, 28)

    while True:
        batch_cells_processed = []
        batch_labels = []
        num_digits_in_batch = 0
        num_empty_in_batch = 0
        processed_sudokus = 0

        while num_digits_in_batch < target_num_digits or num_empty_in_batch < target_num_empty:
            allow_empty = random.random() < 0.8
            rendered_img, gt_grid, warped_corners = renderer.render_sudoku(allow_empty=allow_empty)
            processed_sudokus += 1

            if processed_sudokus > batch_size * 3 and not batch_cells_processed: # Increased safety break limit
                 print("[Gen WARN] Processed many Sudokus without filling batch, breaking loop.")
                 if not batch_cells_processed: continue
                 break

            if rendered_img is None or warped_corners is None: continue
            try:
                rectified_grid = rectify_grid(rendered_img, warped_corners)
                if rectified_grid is None: continue
            except Exception: continue
            extracted_cells, _ = split_into_cells(rectified_grid)
            if len(extracted_cells) != grid_size_sq: continue

            gt_labels_flat = gt_grid.flatten()
            indices = list(range(grid_size_sq))
            random.shuffle(indices)

            for i in indices:
                cell_img = extracted_cells[i]
                label = gt_labels_flat[i]
                is_empty = (label == 0)
                label = EMPTY_LABEL if is_empty else label

                can_add_digit = not is_empty and num_digits_in_batch < target_num_digits
                can_add_empty = is_empty and num_empty_in_batch < target_num_empty

                if can_add_digit or can_add_empty:
                    processed_cell = preprocess_func(cell_img)

                    if processed_cell is None or processed_cell.shape != input_shape_no_channel:
                        continue

                    batch_cells_processed.append(processed_cell)
                    batch_labels.append(label)

                    if is_empty: num_empty_in_batch += 1
                    else: num_digits_in_batch += 1

                if num_digits_in_batch >= target_num_digits and num_empty_in_batch >= target_num_empty:
                    break

        batch_cells_processed = batch_cells_processed[:batch_size]
        batch_labels = batch_labels[:batch_size]

        if not batch_labels:
            print("[Gen WARN] Yielding empty batch!")
            continue

        final_indices = np.random.permutation(len(batch_labels))
        try:
            X_batch = np.array(batch_cells_processed, dtype='float32')[final_indices]
            y_batch = np.array(batch_labels, dtype='int64')[final_indices]
        except ValueError as e:
            print(f"[Gen ERROR] Failed to create batch arrays: {e}")
            continue

        X_batch = np.expand_dims(X_batch, -1)

        if X_batch.shape[1:] != input_size:
             print(f"[Gen ERROR] Final batch shape mismatch: {X_batch.shape} vs {(len(batch_labels),) + input_size}. Skipping batch.")
             continue

        yield X_batch, y_batch
        del X_batch, y_batch, batch_cells_processed, batch_labels
        gc.collect()


# --- Keras Callback for Epoch-End Testing ---
class EpochTestCallback(callbacks.Callback):
    def __init__(self, test_image_path, ground_truth_grid, classifier_instance, frequency=1):
        super().__init__()
        self.test_image_path = test_image_path
        self.ground_truth_grid = ground_truth_grid
        self.classifier = classifier_instance # To access _preprocess_cell_for_model and model input size
        self.frequency = frequency
        self.preprocessed_cells = None
        self.input_shape_no_channel = classifier_instance._model_input_size # e.g., (28, 28)

        print(f"\n[Callback Init] Preparing test image '{self.test_image_path}' for epoch-end evaluation...")
        try:
            cells, _, _ = extract_cells_from_image(self.test_image_path, debug=False)
            if cells is None or len(cells) != GRID_SIZE * GRID_SIZE:
                print("[Callback Init ERROR] Failed to extract cells from test image. Callback disabled.")
                return

            processed = []
            for i, cell_img in enumerate(cells):
                processed_cell = self.classifier._preprocess_cell_for_model(cell_img)
                if processed_cell is None or processed_cell.shape != self.input_shape_no_channel:
                     print(f"[Callback Init WARN] Preprocessing failed or wrong shape for cell {i}. Using zeros.")
                     processed_cell = np.zeros(self.input_shape_no_channel, dtype=np.float32)
                processed.append(processed_cell)

            self.preprocessed_cells = np.array(processed, dtype=np.float32)
            self.preprocessed_cells = np.expand_dims(self.preprocessed_cells, -1) # Add channel dim
            print("[Callback Init] Test image preprocessed successfully.")

        except Exception as e:
            print(f"[Callback Init ERROR] Failed during test image prep: {e}. Callback disabled.")
            self.preprocessed_cells = None

    def on_epoch_end(self, epoch, logs=None):
        if self.preprocessed_cells is None or (epoch + 1) % self.frequency != 0:
            return

        print(f"\n--- Epoch {epoch + 1} Test Example Evaluation ---")
        logs = logs or {}

        try:
            # Ensure the model used by the callback is the one being trained
            if not hasattr(self, 'model') or self.model is None:
                 print("[Callback ERROR] Model not set in callback.")
                 return

            raw_predictions = self.model.predict(self.preprocessed_cells, verbose=0)
            predicted_indices = np.argmax(raw_predictions, axis=1)
            confidences = np.max(raw_predictions, axis=1)

            final_predictions = []
            current_threshold = FINAL_CONFIDENCE_THRESHOLD # Use the global threshold
            for idx, conf in zip(predicted_indices, confidences):
                digit = 0
                if idx != EMPTY_LABEL and conf >= current_threshold:
                    digit = idx # idx is 1-9
                # Otherwise, it remains 0 (empty or low confidence)
                final_predictions.append(digit)

            predicted_grid = np.array(final_predictions).reshape((GRID_SIZE, GRID_SIZE))

            print("Ground Truth:")
            # Print GT without thresholding issues
            print_sudoku_grid(self.ground_truth_grid, threshold=1.1) # Threshold > 1 ensures no '?'

            print(f"\nPrediction (Epoch {epoch + 1}, Threshold={current_threshold:.2f}):")
            confidence_grid = confidences.reshape((GRID_SIZE, GRID_SIZE))
            print_sudoku_grid(predicted_grid, confidence_grid, threshold=current_threshold)

            correct_cells = np.sum(predicted_grid == self.ground_truth_grid)
            total_cells = GRID_SIZE * GRID_SIZE
            accuracy = correct_cells / total_cells
            print(f"Accuracy on this example: {correct_cells}/{total_cells} = {accuracy:.4f}")
            print("--- End Epoch Test ---")

        except Exception as e:
            print(f"[Callback ERROR] Failed during epoch-end evaluation: {e}")
            import traceback
            traceback.print_exc()


# --- Digit Classifier Class ---
class DigitClassifier:
    def __init__(self, model_path=None, training_required=False):
        self.model_path = Path(model_path or Path(__file__).parent / MODEL_FILENAME)
        self.model = None
        self._model_input_size = MODEL_INPUT_SHAPE[:2] # (28, 28)

        if not training_required and self.model_path.exists():
            print(f"Loading existing model from: {self.model_path}")
            try:
                self.model = keras.saving.load_model(self.model_path)
                loaded_input_shape = tuple(self.model.input_shape[1:3])
                if loaded_input_shape != self._model_input_size:
                     print(f"[Warning] Loaded model input shape {loaded_input_shape} differs from expected {self._model_input_size}.")
                print("Model loaded successfully.")
            except Exception as e:
                print(f"[Error] Failed to load model: {e}. Will attempt training.")
                self.model = None
        else:
            if training_required: print("Training explicitly required.")
            elif not self.model_path.exists(): print(f"Model not found at {self.model_path}. Training is required.")
            else: print("Model found but training_required=True. Will retrain.")

    def _preprocess_cell_for_model(self, cell_image):
        """ Preprocesses a single cell image for the CNN model (v5: simpler approach). """
        target_h, target_w = self._model_input_size
        canvas_size = target_h

        if cell_image is None or cell_image.size < 10:
            return np.zeros((target_h, target_w), dtype=np.float32)
        if cell_image.ndim == 3: gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        else: gray = cell_image.copy()

        h_orig, w_orig = gray.shape
        block_size = max(3, min(h_orig, w_orig) // 4)
        if block_size % 2 == 0: block_size += 1
        try:
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, blockSize=block_size, C=7)
        except cv2.error:
            return np.zeros((target_h, target_w), dtype=np.float32)

        coords = cv2.findNonZero(thresh)
        if coords is None: return np.zeros((target_h, target_w), dtype=np.float32)
        x, y, w, h = cv2.boundingRect(coords)
        roi = thresh[y:y+h, x:x+w]

        target_content_size = TARGET_CELL_CONTENT_SIZE
        current_h, current_w = roi.shape
        if current_h == 0 or current_w == 0: return np.zeros((target_h, target_w), dtype=np.float32)
        scale = min(target_content_size / current_w, target_content_size / current_h) if current_w > 0 and current_h > 0 else 0
        new_w, new_h = max(1, int(current_w * scale)), max(1, int(current_h * scale))
        try:
            resized_roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except cv2.error:
            return np.zeros((target_h, target_w), dtype=np.float32)

        final_canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        pad_top = max(0, (canvas_size - new_h) // 2)
        pad_left = max(0, (canvas_size - new_w) // 2)
        end_y = min(canvas_size, pad_top + new_h)
        end_x = min(canvas_size, pad_left + new_w)
        roi_h_slice = end_y - pad_top
        roi_w_slice = end_x - pad_left

        if roi_h_slice > 0 and roi_w_slice > 0:
             # Ensure resized_roi slice matches target slice dimensions
             final_canvas[pad_top:end_y, pad_left:end_x] = resized_roi[:roi_h_slice, :roi_w_slice]

        processed = final_canvas.astype("float32") / 255.0
        if processed.shape != (target_h, target_w):
             processed = cv2.resize(processed, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return processed

    def _build_cnn_model(self):
        """ Builds the CNN model. """
        inputs = keras.Input(shape=MODEL_INPUT_SHAPE)
        augment = keras.Sequential([
            layers.RandomRotation(0.08, fill_mode="constant", fill_value=0.0),
            layers.RandomTranslation(0.08, 0.08, fill_mode="constant", fill_value=0.0),
            layers.RandomZoom(0.08, 0.08, fill_mode="constant", fill_value=0.0),
        ], name="augmentation")
        x = augment(inputs)
        x = layers.Conv2D(32, (3, 3), padding='same')(x); x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
        x = layers.Conv2D(32, (3, 3), padding='same')(x); x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x); x = layers.Dropout(0.25)(x)
        x = layers.Conv2D(64, (3, 3), padding='same')(x); x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
        x = layers.Conv2D(64, (3, 3), padding='same')(x); x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x); x = layers.Dropout(0.25)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128)(x); x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x); x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
        model = models.Model(inputs=inputs, outputs=outputs)
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print("CNN Model Summary:"); model.summary()
        return model

    def train(self, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, batch_size=BATCH_SIZE, validation_steps=VALIDATION_STEPS):
        """ Trains the classifier, now includes the EpochTestCallback. """
        print(f"\n--- Starting Training (v6 - With Epoch Test Callback) ---")
        print(f"Epochs: {epochs}, Steps/Epoch: {steps_per_epoch}, Batch Size: {batch_size}, Val Steps: {validation_steps}")
        print(f"Target Digit Ratio: {TARGET_DIGIT_RATIO}, Model: {self.model_path.name}")

        try:
            test_img_path, test_gt_grid = generate_and_save_test_example()
            use_callback = True
        except Exception as e:
            print(f"[ERROR] Could not generate/load test image for callback: {e}. Callback disabled.")
            use_callback = False

        train_renderer = SudokuRenderer()
        val_renderer = SudokuRenderer()

        train_generator = sudoku_data_generator(
            train_renderer, batch_size, self._preprocess_cell_for_model, MODEL_INPUT_SHAPE, TARGET_DIGIT_RATIO
        )
        val_generator = sudoku_data_generator(
            val_renderer, batch_size, self._preprocess_cell_for_model, MODEL_INPUT_SHAPE, TARGET_DIGIT_RATIO
        )

        if self.model is None: self.model = self._build_cnn_model()
        elif not isinstance(self.model, keras.Model): self.model = self._build_cnn_model()
        else: print("Using pre-loaded or existing model for training.")

        if self.model is None:
             print("[ERROR] Failed to build or load the model before training.")
             return

        callback_list = [
            callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True),
            callbacks.ModelCheckpoint(str(self.model_path), monitor='val_loss', save_best_only=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
        ]
        if use_callback:
            epoch_test_cb = EpochTestCallback(test_img_path, test_gt_grid, self, frequency=1)
            if epoch_test_cb.preprocessed_cells is not None:
                 callback_list.append(epoch_test_cb)
            else:
                 print("[WARN] EpochTestCallback initialization failed, not adding to callbacks.")

        print("\nStarting model fitting with generator...")
        try:
            # Pass the model to the callback explicitly if needed (Keras usually handles this)
            # if use_callback and epoch_test_cb.preprocessed_cells is not None:
            #     epoch_test_cb.set_model(self.model) # Ensure callback has the model reference

            history = self.model.fit(
                train_generator,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=val_generator,
                validation_steps=validation_steps,
                callbacks=callback_list,
                verbose=1
            )
        except Exception as e:
             print(f"\n[Error] An exception occurred during model.fit with generator: {e}")
             import traceback; traceback.print_exc(); print("Aborting training.")
             del train_generator, val_generator; gc.collect()
             return

        print("\nTraining finished.")

        # Reload best model if EarlyStopping restored weights (it does)
        # No explicit reload needed here if restore_best_weights=True worked.
        # self.model should already be the best version.

        if self.model is None: # Check if model became None somehow
             print("[Error] Model is None after training attempt. Trying to load best checkpoint.")
             if self.model_path.exists():
                 try: self.model = keras.saving.load_model(self.model_path); print("Successfully loaded best model from checkpoint.")
                 except Exception as e: print(f"[Error] Failed to reload best model after training failure: {e}."); return
             else: print("[Error] Best model checkpoint not found. Cannot evaluate or save."); return

        if self.model:
            print("Evaluating final (best) model on validation generator...")
            try:
                eval_val_generator = sudoku_data_generator(val_renderer, batch_size, self._preprocess_cell_for_model, MODEL_INPUT_SHAPE, TARGET_DIGIT_RATIO)
                loss, accuracy = self.model.evaluate(eval_val_generator, steps=validation_steps, verbose=1)
                print(f"\nFinal Validation Loss: {loss:.4f}"); print(f"Final Validation Accuracy: {accuracy:.4f}")
                del eval_val_generator; gc.collect()
            except Exception as e: print(f"[Error] Failed to evaluate model: {e}"); import traceback; traceback.print_exc()
        else: print("[Error] Model object is None after training and reload attempts.")

        if self.model:
            print(f"Attempting to save final best model to {self.model_path}...")
            try: self.model.save(self.model_path); print(f"Final best model saved successfully.")
            except Exception as e: print(f"[Error] Failed to save the final model: {e}")

        del train_generator, val_generator; gc.collect()

    @torch.no_grad()
    def recognise(self, cell_image, confidence_threshold=0.7):
        """ Recognises the digit in a single cell image. """
        if self.model is None: return 0, 0.0
        processed_cell = self._preprocess_cell_for_model(cell_image)
        if processed_cell is None or processed_cell.shape != self._model_input_size: return 0, 0.0
        model_input = np.expand_dims(processed_cell, axis=(0, -1))
        if keras.backend.backend() == 'torch':
            try: model_input_tensor = torch.from_numpy(model_input).float()
            except Exception as e: print(f"[Error] Failed converting NumPy to Torch tensor: {e}"); return 0, 0.0
        else: model_input_tensor = model_input
        try: probabilities = self.model(model_input_tensor, training=False)[0]
        except Exception as e: print(f"[Error] Exception during model prediction: {e}"); import traceback; traceback.print_exc(); return 0, 0.0
        if isinstance(probabilities, torch.Tensor):
            if probabilities.device.type != 'cpu': probabilities = probabilities.cpu()
            probabilities = probabilities.numpy()
        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])
        if predicted_class == EMPTY_LABEL: return 0, confidence
        elif confidence < confidence_threshold: return 0, confidence
        else: return predicted_class, confidence


# --- Example Usage (__main__) ---
if __name__ == "__main__":
    print(f"Testing DigitClassifier training (v6 - Epoch Callback)...")
    force_train = False
    model_file = Path(MODEL_FILENAME)

    if force_train and model_file.exists():
        print("Forcing retraining, removing existing model file...")
        try: model_file.unlink()
        except OSError as e: print(f"Error removing model file: {e}")

    classifier = DigitClassifier(training_required=force_train)

    if classifier.model is None:
        print("Classifier model needs training.")
        classifier.train()
    else:
        print("Model already exists and loaded. Skipping training (unless force_train=True).")

    if classifier.model:
         print("\nPerforming a quick recognition test on a dummy image...")
         dummy_cell = np.zeros((50, 50), dtype=np.uint8); cv2.line(dummy_cell, (25, 10), (25, 40), 255, 3)
         pred, conf = classifier.recognise(dummy_cell, confidence_threshold=0.5)
         print(f"Dummy cell ('1') prediction: {pred}, Confidence: {conf:.4f}")
         dummy_empty = np.zeros((50, 50), dtype=np.uint8)
         pred_e, conf_e = classifier.recognise(dummy_empty, confidence_threshold=0.5)
         print(f"Dummy empty cell prediction: {pred_e}, Confidence: {conf_e:.4f}")

    print("\nClassifier test complete.")