# digit_classifier.py
import os
os.environ["KERAS_BACKEND"] = "torch"

import cv2
import numpy as np
import keras
from keras import layers, models, callbacks
import torch
from pathlib import Path
import random
import math
import gc

# Local imports
from sudoku_renderer import SudokuRenderer, generate_and_save_test_example
from digit_extractor import extract_cells_from_image, rectify_grid, split_into_cells, GRID_SIZE
from sudoku_recogniser import print_sudoku_grid, FINAL_CONFIDENCE_THRESHOLD

# --- Constants ---
MODEL_FILENAME = "sudoku_digit_classifier_cnn.keras"
MODEL_INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 11  # 0-9 digits + 1 empty class
EMPTY_LABEL = 10  # Label for the empty cell class
EPOCHS = 40
STEPS_PER_EPOCH = 150
BATCH_SIZE = 128
VALIDATION_STEPS = 50
TARGET_CELL_CONTENT_SIZE = 24 # Target pixel size for the digit within the cell
TARGET_DIGIT_RATIO = 1.5 # Target ratio of digit examples to empty examples in batches

# --- Data Generator ---
def sudoku_data_generator(renderer, batch_size, preprocess_func, input_size, target_digit_ratio=TARGET_DIGIT_RATIO):
    """
    Yields batches of (processed_cells, labels) generated on the fly,
    attempting to balance classes by oversampling digits / undersampling empty cells.

    Args:
        renderer (SudokuRenderer): Instance to generate Sudoku images.
        batch_size (int): Number of samples per batch.
        preprocess_func (callable): Function to preprocess extracted cell images.
        input_size (tuple): Expected input shape for the model (H, W, C).
        target_digit_ratio (float): Desired ratio of digit cells to empty cells in a batch.

    Yields:
        tuple: A batch of (X_batch, y_batch) where X_batch is the processed cell images
               and y_batch are the corresponding labels.
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
        max_sudokus_to_process = batch_size * 4 # Safety break limit

        while len(batch_cells_processed) < batch_size:
            # Generate a new Sudoku image and its ground truth
            allow_empty = random.random() < 0.8 # Sometimes generate grids with fewer digits
            rendered_img, gt_grid, warped_corners = renderer.render_sudoku(allow_empty=allow_empty)
            processed_sudokus += 1

            # Safety break if batch isn't filling up
            if processed_sudokus > max_sudokus_to_process and not batch_cells_processed:
                 print("[Generator WARN] Processed many Sudokus without filling batch, trying again.")
                 break # Break inner loop to generate a new Sudoku

            if rendered_img is None or warped_corners is None:
                continue

            # Extract cells from the rendered image
            try:
                rectified_grid = rectify_grid(rendered_img, warped_corners)
                if rectified_grid is None: continue
                extracted_cells, _ = split_into_cells(rectified_grid)
                if len(extracted_cells) != grid_size_sq: continue
            except Exception as e:
                # print(f"[Generator DEBUG] Cell extraction failed: {e}") # Optional debug
                continue

            gt_labels_flat = gt_grid.flatten()
            indices = list(range(grid_size_sq))
            random.shuffle(indices) # Process cells in random order

            # Add cells to the batch, respecting the target ratio
            for i in indices:
                cell_img = extracted_cells[i]
                label = gt_labels_flat[i]
                is_empty = (label == 0)
                model_label = EMPTY_LABEL if is_empty else label # Map 0 to EMPTY_LABEL

                can_add_digit = not is_empty and num_digits_in_batch < target_num_digits
                can_add_empty = is_empty and num_empty_in_batch < target_num_empty

                if can_add_digit or can_add_empty:
                    processed_cell = preprocess_func(cell_img)

                    # Validate preprocessing output
                    if processed_cell is None or processed_cell.shape != input_shape_no_channel:
                        # print(f"[Generator WARN] Preprocessing failed or wrong shape for a cell. Skipping.")
                        continue

                    batch_cells_processed.append(processed_cell)
                    batch_labels.append(model_label)

                    if is_empty:
                        num_empty_in_batch += 1
                    else:
                        num_digits_in_batch += 1

                # Check if the batch targets are met
                if num_digits_in_batch >= target_num_digits and num_empty_in_batch >= target_num_empty:
                    # Fill the rest of the batch if needed, prioritizing under-represented class
                    needed = batch_size - len(batch_cells_processed)
                    if needed > 0:
                        priority_empty = num_empty_in_batch < target_num_empty
                        for j in indices: # Re-iterate if necessary
                            if len(batch_cells_processed) >= batch_size: break
                            cell_img_fill = extracted_cells[j]
                            label_fill = gt_labels_flat[j]
                            is_empty_fill = (label_fill == 0)
                            model_label_fill = EMPTY_LABEL if is_empty_fill else label_fill

                            # Add if it matches the priority class or if the other is full
                            if (priority_empty and is_empty_fill) or \
                               (not priority_empty and not is_empty_fill) or \
                               (is_empty_fill and num_digits_in_batch >= target_num_digits) or \
                               (not is_empty_fill and num_empty_in_batch >= target_num_empty):

                                processed_cell_fill = preprocess_func(cell_img_fill)
                                if processed_cell_fill is not None and processed_cell_fill.shape == input_shape_no_channel:
                                    batch_cells_processed.append(processed_cell_fill)
                                    batch_labels.append(model_label_fill)
                    break # Exit cell loop once targets are met or batch is full

            if len(batch_cells_processed) >= batch_size:
                break # Exit Sudoku generation loop

        # Finalize and yield the batch
        batch_cells_processed = batch_cells_processed[:batch_size]
        batch_labels = batch_labels[:batch_size]

        if not batch_labels:
            print("[Generator WARN] Yielding empty batch!")
            continue # Skip this iteration

        # Shuffle the final batch
        final_indices = np.random.permutation(len(batch_labels))
        try:
            # Convert to NumPy arrays
            X_batch = np.array(batch_cells_processed, dtype='float32')[final_indices]
            y_batch = np.array(batch_labels, dtype='int64')[final_indices]
        except ValueError as e:
            print(f"[Generator ERROR] Failed to create batch arrays: {e}. Skipping batch.")
            # print(f"[Generator DEBUG] Shapes: {[c.shape for c in batch_cells_processed]}") # Optional debug
            continue

        # Add channel dimension for CNN
        X_batch = np.expand_dims(X_batch, -1)

        # Final shape check
        if X_batch.shape[1:] != input_size:
             print(f"[Generator ERROR] Final batch shape mismatch: {X_batch.shape} vs {(len(batch_labels),) + input_size}. Skipping batch.")
             continue

        yield X_batch, y_batch

        # Clean up memory
        del X_batch, y_batch, batch_cells_processed, batch_labels
        gc.collect()


# --- Keras Callback for Epoch-End Testing ---
class EpochTestCallback(callbacks.Callback):
    """
    Keras Callback to evaluate the model on a fixed test Sudoku image at the end of each epoch.
    """
    def __init__(self, test_image_path, ground_truth_grid, classifier_instance, frequency=1):
        """
        Args:
            test_image_path (str | Path): Path to the test Sudoku image.
            ground_truth_grid (np.ndarray): The 9x9 ground truth grid for the test image.
            classifier_instance (DigitClassifier): The classifier instance (used for preprocessing).
            frequency (int): Evaluate every `frequency` epochs.
        """
        super().__init__()
        self.test_image_path = test_image_path
        self.ground_truth_grid = ground_truth_grid
        self.classifier = classifier_instance
        self.frequency = frequency
        self.preprocessed_cells = None
        self.input_shape_no_channel = classifier_instance._model_input_size # e.g., (28, 28)

        print(f"\n[Callback] Initializing with test image: '{self.test_image_path}'")
        try:
            # Extract and preprocess cells from the test image
            cells, _, _ = extract_cells_from_image(self.test_image_path, debug=False)
            if cells is None or len(cells) != GRID_SIZE * GRID_SIZE:
                print("[Callback ERROR] Failed to extract cells from test image. Callback disabled.")
                return

            processed = []
            for i, cell_img in enumerate(cells):
                processed_cell = self.classifier._preprocess_cell_for_model(cell_img)
                # Handle preprocessing failures by using a blank image
                if processed_cell is None or processed_cell.shape != self.input_shape_no_channel:
                     print(f"[Callback WARN] Preprocessing failed or wrong shape for test cell {i}. Using zeros.")
                     processed_cell = np.zeros(self.input_shape_no_channel, dtype=np.float32)
                processed.append(processed_cell)

            # Prepare the input batch for the model
            self.preprocessed_cells = np.array(processed, dtype=np.float32)
            self.preprocessed_cells = np.expand_dims(self.preprocessed_cells, -1) # Add channel dim
            print("[Callback] Test image preprocessed successfully.")

        except Exception as e:
            print(f"[Callback ERROR] Failed during test image preparation: {e}. Callback disabled.")
            self.preprocessed_cells = None

    def on_epoch_end(self, epoch, logs=None):
        """Runs the evaluation at the end of an epoch."""
        if self.preprocessed_cells is None or (epoch + 1) % self.frequency != 0:
            return # Skip if preprocessing failed or not the right epoch

        print(f"\n--- Epoch {epoch + 1} Test Example Evaluation ---")
        logs = logs or {}

        try:
            # Ensure the callback uses the current state of the model being trained
            if not hasattr(self, 'model') or self.model is None:
                 print("[Callback ERROR] Model not found in callback instance.")
                 return

            # Get predictions from the model
            raw_predictions = self.model.predict(self.preprocessed_cells, verbose=0)
            predicted_indices = np.argmax(raw_predictions, axis=1)
            confidences = np.max(raw_predictions, axis=1)

            # Apply the final confidence threshold to determine the displayed digit
            final_predictions = []
            current_threshold = FINAL_CONFIDENCE_THRESHOLD
            for idx, conf in zip(predicted_indices, confidences):
                digit = 0 # Default to empty/unknown
                if idx != EMPTY_LABEL and conf >= current_threshold:
                    digit = idx # Use the predicted digit (1-9)
                final_predictions.append(digit)

            predicted_grid = np.array(final_predictions).reshape((GRID_SIZE, GRID_SIZE))

            # Print Ground Truth (use high threshold to avoid '?' marks)
            print("Ground Truth:")
            print_sudoku_grid(self.ground_truth_grid, threshold=1.1)

            # Print Prediction
            print(f"\nPrediction (Epoch {epoch + 1}, Threshold={current_threshold:.2f}):")
            confidence_grid = confidences.reshape((GRID_SIZE, GRID_SIZE))
            print_sudoku_grid(predicted_grid, confidence_grid, threshold=current_threshold)

            # Calculate and print accuracy on this specific example
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
    """
    Handles loading, training, and using the CNN model for digit classification.
    """
    def __init__(self, model_path=None, training_required=False):
        """
        Initializes the classifier. Loads an existing model or prepares for training.

        Args:
            model_path (str | Path | None): Path to the model file. Defaults to MODEL_FILENAME.
            training_required (bool): If True, forces training even if a model file exists.
        """
        self.model_path = Path(model_path or Path(__file__).parent / MODEL_FILENAME)
        self.model = None
        self._model_input_size = MODEL_INPUT_SHAPE[:2] # Store as (H, W), e.g., (28, 28)

        # Attempt to load model if not forced to train and file exists
        if not training_required and self.model_path.exists():
            print(f"Loading existing model from: {self.model_path}")
            try:
                self.model = keras.saving.load_model(self.model_path)
                # Verify input shape compatibility
                loaded_input_shape = tuple(self.model.input_shape[1:3])
                if loaded_input_shape != self._model_input_size:
                     print(f"[Warning] Loaded model input shape {loaded_input_shape} differs from expected {self._model_input_size}.")
                     # Potentially raise an error or attempt to adapt? For now, just warn.
                print("Model loaded successfully.")
            except Exception as e:
                print(f"[Error] Failed to load model: {e}. Model will need training.")
                self.model = None # Ensure model is None if loading failed
        else:
            if training_required:
                print("Training explicitly required.")
            elif not self.model_path.exists():
                print(f"Model not found at {self.model_path}. Training is required.")
            # Implicit else: model exists but training_required=True
            # No message needed here, training will proceed.

    def _preprocess_cell_for_model(self, cell_image):
        """
        Preprocesses a single cell image for the CNN model.
        Includes thresholding, finding the digit contour, resizing, and centering.

        Args:
            cell_image (np.ndarray): The image of a single Sudoku cell (BGR or Grayscale).

        Returns:
            np.ndarray | None: The preprocessed grayscale image (normalized to 0-1)
                               ready for the model, or None if preprocessing fails.
        """
        target_h, target_w = self._model_input_size
        canvas_size = target_h # Assume square input for simplicity

        # Handle empty or invalid input
        if cell_image is None or cell_image.size < 10: # Basic check for validity
            return np.zeros((target_h, target_w), dtype=np.float32)

        # Convert to grayscale if necessary
        if cell_image.ndim == 3:
            gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell_image.copy()

        # Adaptive thresholding to binarize the image
        h_orig, w_orig = gray.shape
        # Determine a reasonable block size for adaptive thresholding
        block_size = max(3, min(h_orig, w_orig) // 4)
        if block_size % 2 == 0: block_size += 1 # Block size must be odd
        try:
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, blockSize=block_size, C=7)
        except cv2.error:
            # Handle potential errors like invalid block size for very small images
            return np.zeros((target_h, target_w), dtype=np.float32)

        # Find the contour of the digit within the cell
        coords = cv2.findNonZero(thresh)
        if coords is None: # Empty cell after thresholding
            return np.zeros((target_h, target_w), dtype=np.float32)

        # Get bounding box of the non-zero pixels (the digit)
        x, y, w, h = cv2.boundingRect(coords)
        roi = thresh[y:y+h, x:x+w] # Region of Interest containing the digit

        # Resize the digit ROI to fit within the target content size, maintaining aspect ratio
        target_content_size = TARGET_CELL_CONTENT_SIZE
        current_h, current_w = roi.shape
        if current_h == 0 or current_w == 0:
            return np.zeros((target_h, target_w), dtype=np.float32)

        scale = min(target_content_size / current_w, target_content_size / current_h)
        new_w, new_h = max(1, int(current_w * scale)), max(1, int(current_h * scale))

        try:
            resized_roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except cv2.error: # Handle potential resize errors
            return np.zeros((target_h, target_w), dtype=np.float32)

        # Create a canvas of the target model input size and paste the resized digit into the center
        final_canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        pad_top = max(0, (canvas_size - new_h) // 2)
        pad_left = max(0, (canvas_size - new_w) // 2)

        # Calculate end coordinates, ensuring they don't exceed canvas bounds
        end_y = min(canvas_size, pad_top + new_h)
        end_x = min(canvas_size, pad_left + new_w)

        # Calculate the slice dimensions from the resized ROI to paste
        roi_h_slice = end_y - pad_top
        roi_w_slice = end_x - pad_left

        # Paste the ROI slice, ensuring dimensions match
        if roi_h_slice > 0 and roi_w_slice > 0:
             final_canvas[pad_top:end_y, pad_left:end_x] = resized_roi[:roi_h_slice, :roi_w_slice]

        # Normalize the image to [0, 1] range
        processed = final_canvas.astype("float32") / 255.0

        # Final check and resize if canvas size didn't match target H, W (shouldn't happen if square)
        if processed.shape != (target_h, target_w):
             processed = cv2.resize(processed, (target_w, target_h), interpolation=cv2.INTER_AREA)

        return processed

    def _build_cnn_model(self):
        """ Builds the Convolutional Neural Network model architecture. """
        inputs = keras.Input(shape=MODEL_INPUT_SHAPE)

        # Data Augmentation Layer
        augment = keras.Sequential([
            layers.RandomRotation(0.08, fill_mode="constant", fill_value=0.0),
            layers.RandomTranslation(0.08, 0.08, fill_mode="constant", fill_value=0.0),
            layers.RandomZoom(0.08, 0.08, fill_mode="constant", fill_value=0.0),
        ], name="augmentation")
        x = augment(inputs)

        # Convolutional Block 1
        x = layers.Conv2D(32, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(32, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)

        # Convolutional Block 2
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)

        # Fully Connected Layers
        x = layers.Flatten()(x)
        x = layers.Dense(128)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.5)(x)

        # Output Layer
        outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

        model = models.Model(inputs=inputs, outputs=outputs)

        # Compile the model
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        print("CNN Model Summary:")
        model.summary()
        return model

    def train(self, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, batch_size=BATCH_SIZE, validation_steps=VALIDATION_STEPS):
        """
        Trains the digit classifier model using generated Sudoku data.

        Args:
            epochs (int): Maximum number of training epochs.
            steps_per_epoch (int): Number of batches per training epoch.
            batch_size (int): Number of samples per batch.
            validation_steps (int): Number of batches for validation evaluation per epoch.
        """
        print(f"\n--- Starting Classifier Training ---")
        print(f"Epochs: {epochs}, Steps/Epoch: {steps_per_epoch}, Batch Size: {batch_size}, Val Steps: {validation_steps}")
        print(f"Target Digit Ratio: {TARGET_DIGIT_RATIO}, Model: {self.model_path.name}")

        # Prepare the test example for the EpochTestCallback
        try:
            test_img_path, test_gt_grid = generate_and_save_test_example()
            use_epoch_test_callback = True
        except Exception as e:
            print(f"[ERROR] Could not generate/load test image for callback: {e}. Callback disabled.")
            use_epoch_test_callback = False

        # Initialize data generators
        train_renderer = SudokuRenderer()
        val_renderer = SudokuRenderer() # Use a separate renderer instance for validation

        train_generator = sudoku_data_generator(
            train_renderer, batch_size, self._preprocess_cell_for_model, MODEL_INPUT_SHAPE, TARGET_DIGIT_RATIO
        )
        val_generator = sudoku_data_generator(
            val_renderer, batch_size, self._preprocess_cell_for_model, MODEL_INPUT_SHAPE, TARGET_DIGIT_RATIO
        )

        # Build the model if it wasn't loaded or needs rebuilding
        if self.model is None or not isinstance(self.model, keras.Model):
            self.model = self._build_cnn_model()
        else:
            print("Continuing training with pre-loaded model.")

        if self.model is None:
             print("[ERROR] Failed to build or load the model before training.")
             return

        # Define Keras callbacks
        callback_list = [
            callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True), # Increased patience slightly
            callbacks.ModelCheckpoint(str(self.model_path), monitor='val_loss', save_best_only=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
        ]
        # Add the custom epoch test callback if it initialized correctly
        if use_epoch_test_callback:
            epoch_test_cb = EpochTestCallback(test_img_path, test_gt_grid, self, frequency=1)
            if epoch_test_cb.preprocessed_cells is not None:
                 callback_list.append(epoch_test_cb)
            else:
                 print("[WARN] EpochTestCallback initialization failed, not adding to callbacks.")

        print("\nStarting model training...")
        try:
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
             print(f"\n[Error] An exception occurred during model training: {e}")
             import traceback; traceback.print_exc()
             print("Aborting training.")
             # Clean up generators
             del train_generator, val_generator; gc.collect()
             return

        print("\nTraining finished.")

        # The model object should hold the best weights due to EarlyStopping's restore_best_weights=True
        if self.model is None:
             print("[Error] Model is None after training attempt. Trying to load best checkpoint.")
             if self.model_path.exists():
                 try:
                     self.model = keras.saving.load_model(self.model_path)
                     print("Successfully loaded best model from checkpoint.")
                 except Exception as e:
                     print(f"[Error] Failed to reload best model after training failure: {e}.")
                     return
             else:
                 print("[Error] Best model checkpoint not found. Cannot evaluate or save.")
                 return

        # Final evaluation using the best model
        if self.model:
            print("Evaluating final (best) model on validation generator...")
            try:
                # Create a fresh generator for final evaluation
                eval_val_generator = sudoku_data_generator(val_renderer, batch_size, self._preprocess_cell_for_model, MODEL_INPUT_SHAPE, TARGET_DIGIT_RATIO)
                loss, accuracy = self.model.evaluate(eval_val_generator, steps=validation_steps, verbose=1)
                print(f"\nFinal Validation Loss: {loss:.4f}")
                print(f"Final Validation Accuracy: {accuracy:.4f}")
                del eval_val_generator; gc.collect()
            except Exception as e:
                print(f"[Error] Failed to evaluate final model: {e}")
                import traceback; traceback.print_exc()
        else:
            print("[Error] Model object is None after training and reload attempts.")

        # Save the final best model (should be redundant if ModelCheckpoint worked, but safe)
        if self.model:
            print(f"Saving final best model to {self.model_path}...")
            try:
                self.model.save(self.model_path)
                print(f"Final best model saved successfully.")
            except Exception as e:
                print(f"[Error] Failed to save the final model: {e}")

        # Clean up generators
        del train_generator, val_generator; gc.collect()

    @torch.no_grad() # Disable gradient calculations for inference
    def recognise(self, cell_image, confidence_threshold=0.7):
        """
        Recognises the digit in a single cell image using the trained model.

        Args:
            cell_image (np.ndarray): The image of a single Sudoku cell.
            confidence_threshold (float): Minimum confidence score to accept a digit prediction.

        Returns:
            tuple: (predicted_digit, confidence_score)
                   - predicted_digit (int): The recognised digit (1-9), or 0 for empty/uncertain.
                   - confidence_score (float): The model's confidence (0.0-1.0).
        """
        if self.model is None:
            print("[Error] Recognise called but model is not loaded.")
            return 0, 0.0

        # Preprocess the input cell
        processed_cell = self._preprocess_cell_for_model(cell_image)
        if processed_cell is None or processed_cell.shape != self._model_input_size:
            # print("[Debug] Preprocessing failed or returned None/wrong shape in recognise.")
            return 0, 0.0 # Treat preprocessing failure as empty

        # Prepare input for the model (add batch and channel dimensions)
        model_input = np.expand_dims(processed_cell, axis=(0, -1)) # Shape: (1, H, W, 1)

        # Handle backend-specific tensor conversion if needed (PyTorch)
        if keras.backend.backend() == 'torch':
            try:
                model_input_tensor = torch.from_numpy(model_input).float()
                # If using GPU, move tensor: model_input_tensor = model_input_tensor.to(device)
            except Exception as e:
                print(f"[Error] Failed converting NumPy to Torch tensor: {e}")
                return 0, 0.0
        else: # TensorFlow or other backends
            model_input_tensor = model_input

        # Perform prediction
        try:
            # Use `training=False` for inference mode (disables dropout, etc.)
            probabilities = self.model(model_input_tensor, training=False)[0] # Get probabilities for the first (only) item in batch
        except Exception as e:
            print(f"[Error] Exception during model prediction: {e}")
            import traceback; traceback.print_exc()
            return 0, 0.0

        # Convert back to NumPy if prediction returns a Torch tensor
        if isinstance(probabilities, torch.Tensor):
            if probabilities.device.type != 'cpu':
                probabilities = probabilities.cpu() # Move to CPU if needed
            probabilities = probabilities.numpy()

        # Determine predicted class and confidence
        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])

        # Interpret the prediction
        if predicted_class == EMPTY_LABEL:
            return 0, confidence # Return 0 for the empty class
        elif confidence < confidence_threshold:
            return 0, confidence # Return 0 if confidence is below threshold
        else:
            # predicted_class is 1-9 (since EMPTY_LABEL is 10)
            return predicted_class, confidence


# --- Example Usage (__main__) ---
if __name__ == "__main__":
    print(f"Testing DigitClassifier...")
    force_train = False # Set to True to force retraining even if model exists
    model_file = Path(MODEL_FILENAME)

    # Optionally remove existing model if forcing training
    if force_train and model_file.exists():
        print("Forcing retraining, removing existing model file...")
        try:
            model_file.unlink()
        except OSError as e:
            print(f"Error removing model file '{model_file}': {e}")

    # Initialize the classifier
    classifier = DigitClassifier(training_required=force_train)

    # Train if the model wasn't loaded or training is forced
    if classifier.model is None:
        print("Classifier model needs training.")
        classifier.train() # Start the training process
    else:
        print("Model already exists and loaded. Skipping training.")

    # Perform a simple test if the model is available
    if classifier.model:
         print("\nPerforming a quick recognition test on dummy images...")
         # Create a dummy cell image resembling '1'
         dummy_cell_1 = np.zeros((50, 50), dtype=np.uint8)
         cv2.line(dummy_cell_1, (25, 10), (25, 40), 255, 3) # Draw a white line
         pred_1, conf_1 = classifier.recognise(dummy_cell_1, confidence_threshold=0.5)
         print(f"Dummy cell ('1') prediction: {pred_1}, Confidence: {conf_1:.4f}")

         # Create a dummy empty cell image
         dummy_empty = np.zeros((50, 50), dtype=np.uint8)
         pred_e, conf_e = classifier.recognise(dummy_empty, confidence_threshold=0.5)
         print(f"Dummy empty cell prediction: {pred_e}, Confidence: {conf_e:.4f}")

    print("\nClassifier test complete.")

