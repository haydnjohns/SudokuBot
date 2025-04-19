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
import gc

from sudoku_renderer import SudokuRenderer
from digit_extractor import rectify_grid, split_into_cells

# --- Constants ---
GRID_SIZE = 9
MODEL_FILENAME = "sudoku_digit_classifier_cnn_v5.keras" # v5 for new preprocessing & more training
MODEL_INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 11
EMPTY_LABEL = 10
# --- Increased Training Parameters ---
STEPS_PER_EPOCH = 150 # Significantly increased
EPOCHS = 40          # Significantly increased (EarlyStopping will manage)
BATCH_SIZE = 128
VALIDATION_STEPS = 50 # Increased
# --- Adjusted Content Size ---
TARGET_CELL_CONTENT_SIZE = 24 # Increased slightly from 20
# --- Oversampling Ratio ---
TARGET_DIGIT_RATIO = 1.5 # Keep for now

# --- Data Generator (Keep as is for now) ---
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
        processed_sudokus = 0
        while num_digits_in_batch < target_num_digits or num_empty_in_batch < target_num_empty:
            # Generate a new Sudoku image
            allow_empty = random.random() < 0.8
            rendered_img, gt_grid, warped_corners = renderer.render_sudoku(allow_empty=allow_empty)
            processed_sudokus += 1

            # Safety break if generator gets stuck (e.g., renderer fails repeatedly)
            if processed_sudokus > batch_size * 2 and not batch_cells_processed:
                 print("[Gen WARN] Processed many Sudokus without filling batch, breaking loop.")
                 # Yield whatever we have, even if small, or handle error
                 if not batch_cells_processed: continue # Try again if completely empty
                 break # Yield partial batch

            if rendered_img is None or warped_corners is None: continue
            try:
                rectified_grid = rectify_grid(rendered_img, warped_corners)
                if rectified_grid is None: continue # Handle potential failure in rectify
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
                    # --- Preprocessing Call ---
                    processed_cell = preprocess_func(cell_img) # Use the passed function
                    # ---

                    if processed_cell is None or processed_cell.shape != input_size[:2]: # Check shape before expand_dims
                        # print(f"[Gen WARN] Preprocessed cell shape mismatch or None: {processed_cell.shape if processed_cell is not None else 'None'} vs {input_size[:2]}. Skipping cell.")
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

        # Handle cases where the batch is smaller than expected (e.g., generator issues)
        if not batch_labels:
            print("[Gen WARN] Yielding empty batch!")
            continue # Skip yielding empty batch

        # Shuffle the final batch
        final_indices = np.random.permutation(len(batch_labels))
        # Ensure consistent dtype and shape before expanding dims
        try:
            X_batch = np.array(batch_cells_processed, dtype='float32')[final_indices]
            y_batch = np.array(batch_labels, dtype='int64')[final_indices]
        except ValueError as e:
            print(f"[Gen ERROR] Failed to create batch arrays: {e}")
            # Debug: Print shapes of items in batch_cells_processed
            # for idx, item in enumerate(batch_cells_processed):
            #     print(f"Item {idx} shape: {item.shape if hasattr(item, 'shape') else type(item)}")
            continue # Skip this faulty batch


        # Add channel dimension AFTER array creation
        X_batch = np.expand_dims(X_batch, -1)

        # Final shape check
        if X_batch.shape[1:] != input_size:
             print(f"[Gen ERROR] Final batch shape mismatch: {X_batch.shape} vs {(len(batch_labels),) + input_size}. Skipping batch.")
             continue

        # Debug: Check batch balance
        # unique_final, counts_final = np.unique(y_batch, return_counts=True)
        # print(f"[Debug Gen] Yielding Batch ({X_batch.shape}, {y_batch.shape}) - Labels: {dict(zip(unique_final, counts_final))}")

        yield X_batch, y_batch
        # Minimal cleanup needed here as loop continues
        del X_batch, y_batch, batch_cells_processed, batch_labels
        gc.collect()


# --- Digit Classifier Class ---
class DigitClassifier:
    def __init__(self, model_path=None, training_required=False):
        self.model_path = Path(model_path or Path(__file__).parent / MODEL_FILENAME) # Use v5 filename
        self.model = None
        self._model_input_size = MODEL_INPUT_SHAPE[:2] # Should be (28, 28)

        if not training_required and self.model_path.exists():
            print(f"Loading existing model from: {self.model_path}")
            try:
                # Ensure custom objects are handled if necessary (though not needed for this model)
                self.model = keras.saving.load_model(self.model_path)
                # Verify input shape after loading
                loaded_input_shape = tuple(self.model.input_shape[1:3]) # e.g., (28, 28)
                if loaded_input_shape != self._model_input_size:
                     print(f"[Warning] Loaded model input shape {loaded_input_shape} "
                           f"differs from expected {self._model_input_size}.")
                     # Decide how to handle: error out, retrain, or proceed with caution?
                     # For now, just warn.
                print("Model loaded successfully.")
            except Exception as e:
                print(f"[Error] Failed to load model: {e}. Will attempt training.")
                self.model = None
        else:
            if training_required: print("Training explicitly required.")
            elif not self.model_path.exists(): print(f"Model not found at {self.model_path}. Training is required.")
            else: print("Model found but training_required=True. Will retrain.") # Clarify logic

    # --- Revised Preprocessing Function ---
    def _preprocess_cell_for_model(self, cell_image):
        """
        Preprocesses a single cell image for the CNN model (v5: simpler approach).
        Resizes content and centers it on a standard canvas.
        """
        target_h, target_w = self._model_input_size # e.g., (28, 28)
        canvas_size = target_h # Assume square

        # Handle empty or invalid input
        if cell_image is None or cell_image.size < 10: # Basic check for validity
            return np.zeros((target_h, target_w), dtype=np.float32)

        # 1. Convert to Grayscale
        if cell_image.ndim == 3:
            gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell_image.copy()

        # 2. Adaptive Thresholding (Inverted Binary: digit is white, background black)
        # Use a block size relative to image size, ensure it's odd and >= 3
        h_orig, w_orig = gray.shape
        block_size = max(3, min(h_orig, w_orig) // 4)
        if block_size % 2 == 0: block_size += 1 # Ensure odd
        try:
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, blockSize=block_size, C=7) # C=7 is common
        except cv2.error:
            # Fallback or return empty if thresholding fails
            # print("[PreProc WARN] Adaptive threshold failed. Returning empty.")
            return np.zeros((target_h, target_w), dtype=np.float32)

        # 3. Find Bounding Box of non-zero pixels (the digit 'ink')
        coords = cv2.findNonZero(thresh)
        if coords is None: # Empty cell after thresholding
            return np.zeros((target_h, target_w), dtype=np.float32)

        x, y, w, h = cv2.boundingRect(coords)

        # 4. Extract the ROI (Region of Interest - the digit)
        roi = thresh[y:y+h, x:x+w]

        # 5. Resize ROI to fit within TARGET_CELL_CONTENT_SIZE, maintaining aspect ratio
        target_content_size = TARGET_CELL_CONTENT_SIZE # e.g., 24
        current_h, current_w = roi.shape

        if current_h == 0 or current_w == 0:
             return np.zeros((target_h, target_w), dtype=np.float32)

        scale = min(target_content_size / current_w, target_content_size / current_h)
        new_w, new_h = int(current_w * scale), int(current_h * scale)

        # Ensure dimensions are at least 1x1
        new_w = max(1, new_w)
        new_h = max(1, new_h)

        try:
            resized_roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except cv2.error:
            # print(f"[PreProc WARN] Resize failed for size ({new_w}, {new_h}). Returning empty.")
            return np.zeros((target_h, target_w), dtype=np.float32)


        # 6. Create final canvas and paste the resized ROI in the center
        final_canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8) # Black background

        pad_top = (canvas_size - new_h) // 2
        pad_left = (canvas_size - new_w) // 2

        # Calculate end coordinates, ensuring they don't exceed canvas bounds
        end_y = pad_top + new_h
        end_x = pad_left + new_w

        # Ensure slicing doesn't go out of bounds (shouldn't happen with centered paste)
        pad_top = max(0, pad_top)
        pad_left = max(0, pad_left)
        end_y = min(canvas_size, end_y)
        end_x = min(canvas_size, end_x)

        # Adjust ROI slicing if padding caused issues (rare, but safety)
        roi_h_slice = end_y - pad_top
        roi_w_slice = end_x - pad_left

        if roi_h_slice > 0 and roi_w_slice > 0:
             final_canvas[pad_top:end_y, pad_left:end_x] = resized_roi[:roi_h_slice, :roi_w_slice]
        # else:
             # print(f"[PreProc WARN] Calculated slice size is zero or negative ({roi_h_slice}x{roi_w_slice}).")


        # 7. Normalize to float32 between 0.0 and 1.0
        processed = final_canvas.astype("float32") / 255.0

        # Ensure final shape matches exactly
        if processed.shape != (target_h, target_w):
             # print(f"[PreProc ERROR] Final shape mismatch: {processed.shape} vs {(target_h, target_w)}. Resizing.")
             processed = cv2.resize(processed, (target_w, target_h), interpolation=cv2.INTER_AREA)

        return processed
    # --- End of Revised Preprocessing ---

    # _build_cnn_model remains the same
    def _build_cnn_model(self):
        # --- Consider adding Input layer explicitly for clarity ---
        inputs = keras.Input(shape=MODEL_INPUT_SHAPE)

        # --- Augmentation Layer ---
        augment = keras.Sequential([
            layers.RandomRotation(0.08, fill_mode="constant", fill_value=0.0), # Reduced rotation slightly
            layers.RandomTranslation(0.08, 0.08, fill_mode="constant", fill_value=0.0), # Reduced translation
            layers.RandomZoom(0.08, 0.08, fill_mode="constant", fill_value=0.0), # Reduced zoom
        ], name="augmentation")
        x = augment(inputs)

        # --- Convolutional Base ---
        x = layers.Conv2D(32, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(32, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)

        # --- Classifier Head ---
        x = layers.Flatten()(x)
        x = layers.Dense(128)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

        model = models.Model(inputs=inputs, outputs=outputs)

        # --- Optimizer ---
        # Start with Adam, ReduceLROnPlateau will adjust it
        optimizer = keras.optimizers.Adam(learning_rate=0.001)

        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        print("CNN Model Summary:")
        model.summary()
        return model

    # Modified train method - uses new defaults, removed class_weight
    def train(self, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, batch_size=BATCH_SIZE, validation_steps=VALIDATION_STEPS):
        """
        Trains the classifier using a generator for synthetic data with oversampling.
        Uses updated default parameters for longer training.
        """
        print(f"\n--- Starting Training (v5 - Longer Training & New Preprocessing) ---")
        print(f"Epochs: {epochs}, Steps/Epoch: {steps_per_epoch}, Batch Size: {batch_size}, Val Steps: {validation_steps}")
        print(f"Target Digit Ratio: {TARGET_DIGIT_RATIO}")
        print(f"Model Input Shape: {MODEL_INPUT_SHAPE}, Target Content Size: {TARGET_CELL_CONTENT_SIZE}")

        train_renderer = SudokuRenderer()
        val_renderer = SudokuRenderer() # Use a separate renderer instance for validation

        # Create generators - pass the instance method for preprocessing
        train_generator = sudoku_data_generator(
            train_renderer, batch_size, self._preprocess_cell_for_model, MODEL_INPUT_SHAPE, TARGET_DIGIT_RATIO
        )
        val_generator = sudoku_data_generator(
            val_renderer, batch_size, self._preprocess_cell_for_model, MODEL_INPUT_SHAPE, TARGET_DIGIT_RATIO
        )

        if self.model is None: # Build model only if not loaded
             self.model = self._build_cnn_model()
        elif not isinstance(self.model, keras.Model): # Check if it's a valid model object
             print("[Warning] self.model exists but is not a valid Keras model. Rebuilding.")
             self.model = self._build_cnn_model()
        else:
             print("Using pre-loaded or existing model for training.")
             # Re-compile in case optimizer state needs reset? Optional.
             # self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
             #                    loss='sparse_categorical_crossentropy',
             #                    metrics=['accuracy'])


        # Ensure model is built before setting up callbacks that might need it
        if self.model is None:
             print("[ERROR] Failed to build or load the model before training.")
             return

        callbacks = [
            # Increased patience for early stopping
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=1, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(str(self.model_path), monitor='val_loss', save_best_only=True, verbose=1),
            # Keep ReduceLROnPlateau
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
        ]

        print("\nStarting model fitting with generator...")
        try:
            history = self.model.fit(
                train_generator,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=val_generator,
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=1
            )
        except Exception as e:
             print(f"\n[Error] An exception occurred during model.fit with generator: {e}")
             import traceback
             traceback.print_exc()
             print("Aborting training.")
             # Clean up generators explicitly on error
             del train_generator, val_generator; gc.collect()
             return # Exit training method

        print("\nTraining finished.")

        # --- Post-Training Evaluation and Saving ---
        # Check if a model exists (might be None if training failed early)
        if self.model is None:
             print("[Error] Model is None after training attempt. Trying to load best checkpoint.")
             if self.model_path.exists():
                 try:
                     self.model = keras.saving.load_model(self.model_path)
                     print("Successfully loaded best model from checkpoint.")
                 except Exception as e:
                     print(f"[Error] Failed to reload best model after training failure: {e}.")
                     return # Cannot proceed
             else:
                 print("[Error] Best model checkpoint not found. Cannot evaluate or save.")
                 return # Cannot proceed

        # If EarlyStopping restored weights, self.model should be the best one.
        # If not, explicitly load the best checkpoint.
        if not any(isinstance(cb, keras.callbacks.EarlyStopping) and cb.restore_best_weights for cb in callbacks):
             if self.model_path.exists():
                 print(f"Reloading best model weights from {self.model_path}")
                 try:
                     self.model = keras.saving.load_model(self.model_path)
                 except Exception as e:
                     print(f"[Error] Failed to reload best model: {e}.")
             else:
                 print("[Warning] Best model checkpoint file not found.")


        # Evaluate the final (best) model
        if self.model:
            print("Evaluating final model on validation generator...")
            try:
                # Create a fresh generator for final evaluation
                eval_val_generator = sudoku_data_generator(val_renderer, batch_size, self._preprocess_cell_for_model, MODEL_INPUT_SHAPE, TARGET_DIGIT_RATIO)
                loss, accuracy = self.model.evaluate(eval_val_generator, steps=validation_steps, verbose=1)
                print(f"\nFinal Validation Loss: {loss:.4f}")
                print(f"Final Validation Accuracy: {accuracy:.4f}")
                del eval_val_generator; gc.collect() # Clean up generator
            except Exception as e:
                 print(f"[Error] Failed to evaluate model: {e}")
                 import traceback
                 traceback.print_exc()
        else:
            print("[Error] Model object is None after training and reload attempts.")


        # Save the final best model (even if loaded from checkpoint)
        if self.model:
            print(f"Attempting to save final best model to {self.model_path}...")
            try:
                self.model.save(self.model_path)
                print(f"Final best model saved successfully.")
            except Exception as e:
                print(f"[Error] Failed to save the final model: {e}")

        # Clean up generators
        del train_generator, val_generator; gc.collect()


    # recognise method remains the same (including MPS fix)
    @torch.no_grad()
    def recognise(self, cell_image, confidence_threshold=0.7):
        """
        Recognises the digit in a single cell image. Handles MPS tensor conversion.
        Uses the revised _preprocess_cell_for_model.
        """
        if self.model is None:
            print("[Error] Model not loaded or trained.")
            return 0, 0.0 # Return 0 (empty) with 0 confidence

        # --- Use the revised preprocessing ---
        processed_cell = self._preprocess_cell_for_model(cell_image)
        # ---

        # Check shape AFTER preprocessing
        if processed_cell is None or processed_cell.shape != self._model_input_size:
             # print(f"[WARN in recognise] Preprocessed cell shape mismatch or None: {processed_cell.shape if processed_cell is not None else 'None'}. Expected {self._model_input_size}.")
             return 0, 0.0 # Treat as empty if preprocessing failed

        # Add batch and channel dimensions: (1, 28, 28, 1)
        model_input = np.expand_dims(processed_cell, axis=(0, -1))

        # Convert to Torch tensor if using Torch backend
        if keras.backend.backend() == 'torch':
            try:
                 # Let Keras/Torch handle device placement during prediction
                 model_input_tensor = torch.from_numpy(model_input).float()
                 # If MPS issues persist, explicitly move:
                 # if torch.backends.mps.is_available():
                 #     mps_device = torch.device("mps")
                 #     model_input_tensor = model_input_tensor.to(mps_device)
                 #     # Ensure model is also on MPS (usually happens automatically if available)
                 #     # self.model.to(mps_device) # Might be needed once at init/load
            except Exception as e:
                 print(f"[Error] Failed converting NumPy to Torch tensor: {e}")
                 return 0, 0.0
        else:
             # TensorFlow/other backends handle NumPy directly
             model_input_tensor = model_input

        # Perform prediction
        try:
            # Use the tensor (Torch) or numpy array (TF) as input
            probabilities = self.model(model_input_tensor, training=False)[0] # Get probabilities for the first (only) item in batch
        except Exception as e:
             print(f"[Error] Exception during model prediction: {e}")
             import traceback
             traceback.print_exc()
             return 0, 0.0

        # Convert Torch tensor output to NumPy array (including handling MPS device)
        if isinstance(probabilities, torch.Tensor):
            if probabilities.device.type != 'cpu':
                 probabilities = probabilities.cpu() # Move to CPU
            probabilities = probabilities.numpy() # Convert to NumPy

        # Get predicted class and confidence
        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])

        # Map prediction to digit (0 for empty/unknown)
        if predicted_class == EMPTY_LABEL: # Predicted empty
             # Return 0, but use the model's confidence in the 'empty' prediction
             return 0, confidence
        elif confidence < confidence_threshold: # Predicted a digit, but low confidence
             return 0, confidence # Return 0 (unknown), along with the low confidence score
        else: # Predicted a digit with sufficient confidence
             return predicted_class, confidence


# --- Example Usage (for training) ---
if __name__ == "__main__":
    print(f"Testing DigitClassifier training (v5 - Longer, New Preprocessing)...")
    force_train = False # Set to True to force retraining even if model exists
    # Ensure using the new model filename
    model_file = Path(MODEL_FILENAME) # Uses v5 filename defined above

    if force_train and model_file.exists():
        print("Forcing retraining, removing existing model file...")
        try:
            model_file.unlink()
        except OSError as e:
            print(f"Error removing model file: {e}")

    # Pass training_required=True if force_train is set
    classifier = DigitClassifier(training_required=force_train)

    # Train only if the model wasn't loaded successfully or if forced
    if classifier.model is None:
        print("Classifier model needs training.")
        classifier.train() # Use new defaults: EPOCHS=40, STEPS_PER_EPOCH=150, etc.
    else:
        print("Model already exists and loaded. Skipping training (unless force_train=True).")

    # --- Add a simple test after training/loading ---
    if classifier.model:
         print("\nPerforming a quick recognition test on a dummy image...")
         dummy_cell = np.zeros((50, 50), dtype=np.uint8) # Create a dummy black cell
         # Draw a white '1' roughly in the center for testing
         cv2.line(dummy_cell, (25, 10), (25, 40), 255, 3)
         pred, conf = classifier.recognise(dummy_cell, confidence_threshold=0.5)
         print(f"Dummy cell ('1') prediction: {pred}, Confidence: {conf:.4f}")

         dummy_empty = np.zeros((50, 50), dtype=np.uint8) # Empty cell
         pred_e, conf_e = classifier.recognise(dummy_empty, confidence_threshold=0.5)
         print(f"Dummy empty cell prediction: {pred_e}, Confidence: {conf_e:.4f}")

    print("\nClassifier test complete.")