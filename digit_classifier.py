# digit_classifier.py
import os
# Set Keras backend *before* importing Keras
os.environ["KERAS_BACKEND"] = "torch" # Or "tensorflow"

import cv2
import numpy as np
import keras
from keras import layers, models
import torch # Explicit import for torch backend specifics
from pathlib import Path
import random
import math
from sklearn.model_selection import train_test_split # Still useful for initial weight calc split
from sklearn.utils import class_weight
import gc # Garbage collector

# Import necessary components from other modules
from sudoku_renderer import SudokuRenderer
from digit_extractor import rectify_grid, split_into_cells # _order_points not needed here

# --- Constants ---
MODEL_FILENAME = "sudoku_digit_classifier_cnn_v3.keras" # Model name for generator training
MODEL_INPUT_SHAPE = (28, 28, 1) # Grayscale images, 28x28 pixels
NUM_CLASSES = 11 # Digits 0-9 + Empty class
EMPTY_LABEL = 10 # Label for the empty class
# Training parameters adjusted for generator use
# Define epoch length in terms of batches/steps
STEPS_PER_EPOCH = 500  # Number of batches per "epoch" (Increase for more training per epoch)
EPOCHS = 30            # Total number of "epochs" (cycles of STEPS_PER_EPOCH batches)
BATCH_SIZE = 128       # Number of cell images per batch
VALIDATION_STEPS = 100 # Number of batches for validation per epoch
TARGET_CELL_CONTENT_SIZE = 20 # Target size of digit within the 28x28 frame

# --- Helper function for data generation ---
def sudoku_data_generator(renderer, batch_size, preprocess_func, input_size):
    """
    Yields batches of (processed_cells, labels) generated on the fly.

    Args:
        renderer (SudokuRenderer): An instance of the Sudoku image generator.
        batch_size (int): The number of samples per batch.
        preprocess_func (callable): The function to preprocess each extracted cell.
        input_size (tuple): The expected (height, width) of the preprocessed cell.
    """
    grid_size_sq = GRID_SIZE * GRID_SIZE # 81
    while True: # Loop indefinitely to generate batches
        batch_cells_processed = []
        batch_labels = []

        # Keep generating Sudokus until the batch is full
        while len(batch_cells_processed) < batch_size:
            # Generate a new Sudoku image
            allow_empty = random.random() < 0.8 # Control frequency of empty cells
            rendered_img, gt_grid, warped_corners = renderer.render_sudoku(allow_empty=allow_empty)

            if rendered_img is None or warped_corners is None:
                # print("[Gen Warn] Renderer failed. Skipping.")
                continue # Skip this Sudoku if rendering failed

            # Rectify using known ground truth corners
            try:
                # Use default rectified size (450) for consistency during generation
                rectified_grid = rectify_grid(rendered_img, warped_corners)
            except Exception as e:
                # print(f"[Gen Warn] Rectify failed: {e}. Skipping.")
                continue # Skip if rectification fails

            # Split into cells
            extracted_cells, _ = split_into_cells(rectified_grid)

            if len(extracted_cells) != grid_size_sq:
                # print(f"[Gen Warn] Split failed (got {len(extracted_cells)} cells). Skipping.")
                continue # Skip if splitting fails

            gt_labels_flat = gt_grid.flatten()

            # Process cells from this Sudoku and add to batch
            for i, cell_img in enumerate(extracted_cells):
                if len(batch_cells_processed) >= batch_size:
                    break # Stop adding if batch is full

                label = gt_labels_flat[i]
                if label == 0: label = EMPTY_LABEL # Assign empty label

                processed_cell = preprocess_func(cell_img) # Call the preprocessing method

                # Verification: Ensure preprocessing returned the correct shape
                if processed_cell.shape != input_size:
                     print(f"[Gen ERROR] Preprocessed cell shape mismatch: {processed_cell.shape} vs {input_size}. Skipping cell.")
                     continue # Skip this specific cell

                batch_cells_processed.append(processed_cell)
                batch_labels.append(label)

        # Convert the completed batch to NumPy arrays
        X_batch = np.array(batch_cells_processed, dtype='float32')
        y_batch = np.array(batch_labels, dtype='int64')

        # Add channel dimension for CNN input
        X_batch = np.expand_dims(X_batch, -1)

        # Yield the batch
        yield X_batch, y_batch

        # Clean up to potentially help memory management
        del X_batch, y_batch, batch_cells_processed, batch_labels
        gc.collect()


# --- Digit Classifier Class ---
class DigitClassifier:
    """
    Trains and uses a CNN model to classify Sudoku cell images (0-9 or empty).
    Uses a generator for on-the-fly training data generation.
    """
    def __init__(self, model_path=None, training_required=False):
        """
        Initializes the classifier. Loads an existing model or prepares for training.
        """
        self.model_path = Path(model_path or Path(__file__).parent / MODEL_FILENAME)
        self.model = None
        self._model_input_size = MODEL_INPUT_SHAPE[:2] # (height, width) -> (28, 28)

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

    def _preprocess_cell_for_model(self, cell_image, is_training=False):
        """
        Prepares a single extracted cell image for model input (28x28).
        - Converts to grayscale
        - Applies adaptive threshold
        - Finds largest contour (digit) within the cell
        - Resizes digit preserving aspect ratio and centers it on a 28x28 canvas
        - Normalizes
        """
        target_h, target_w = self._model_input_size # Should be (28, 28)

        if cell_image is None or cell_image.size < 10: # Ignore very small cells
            return np.zeros((target_h, target_w), dtype=np.float32)

        # 1. Grayscale
        if cell_image.ndim == 3: gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        else: gray = cell_image.copy()

        # --- Adaptive Thresholding Robustness ---
        cell_h_orig, cell_w_orig = gray.shape[:2]
        min_dim = min(cell_h_orig, cell_w_orig)
        # Calculate a reasonable block size, ensuring it's odd and >= 3
        block_size = min(19, min_dim - 1 if min_dim % 2 == 0 else min_dim - 2)
        block_size = max(3, block_size)
        # ---

        # 2. Adaptive Threshold
        try:
            thresh = cv2.adaptiveThreshold(gray, 255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, # Digit white, bg black
                                           block_size, 5) # Use calculated block_size, C=5
        except cv2.error as e:
             # Handle cases where block_size might still be invalid for tiny cells
             # print(f"[Warning] cv2 error during adaptiveThreshold: {e}. Using blank cell.")
             return np.zeros((target_h, target_w), dtype=np.float32)


        # 3. Find Largest Contour (Digit)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digit_contour = None
        max_area = 0
        if contours:
            for cnt in contours:
                area = cv2.contourArea(cnt)
                x, y, w, h = cv2.boundingRect(cnt)
                # Filter tiny noise and contours touching border
                if area > 5 and area > max_area and (x > 0 and y > 0 and x+w < thresh.shape[1] and y+h < thresh.shape[0]):
                     max_area = area
                     digit_contour = cnt

        # --- Create the FINAL 28x28 canvas ---
        final_canvas = np.zeros((target_h, target_w), dtype=np.uint8) # Explicitly 28x28 black canvas
        # ---

        if digit_contour is not None:
            x, y, w, h = cv2.boundingRect(digit_contour)
            digit_roi = thresh[y:y+h, x:x+w] # Extract the digit ROI

            # --- Resize ROI to fit within TARGET_CELL_CONTENT_SIZE (e.g., 20x20) ---
            # --- while preserving aspect ratio ---
            target_content_size = TARGET_CELL_CONTENT_SIZE # e.g., 20
            # Calculate scaling factor, handle potential division by zero
            scale = min(target_content_size / w, target_content_size / h) if h > 0 and w > 0 else 0
            new_w, new_h = int(w * scale), int(h * scale)

            if new_w > 0 and new_h > 0: # Proceed only if ROI is valid and scale is positive
                try:
                    resized_roi = cv2.resize(digit_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

                    # Calculate padding needed to center this resized ROI in the 28x28 final canvas
                    pad_top = (target_h - new_h) // 2
                    pad_left = (target_w - new_w) // 2

                    # Place the resized ROI onto the final 28x28 canvas
                    # Ensure slicing doesn't go out of bounds
                    end_y = min(pad_top + new_h, target_h)
                    end_x = min(pad_left + new_w, target_w)
                    roi_h_slice = end_y - pad_top
                    roi_w_slice = end_x - pad_left

                    # Ensure the slice dimensions match the part of resized_roi we take
                    final_canvas[pad_top:end_y, pad_left:end_x] = resized_roi[:roi_h_slice, :roi_w_slice]

                except cv2.error as e:
                    # This might happen if contour/ROI is degenerate after calculations
                    # print(f"[Warning] cv2 error during resize/paste: {e}. Cell remains blank.")
                    pass # final_canvas remains black

        # 4. Normalize the final 28x28 canvas
        processed = final_canvas.astype("float32") / 255.0
        return processed # Should now always be (28, 28)

    def _build_cnn_model(self):
        """Defines and compiles the CNN architecture with augmentation."""
        # Data Augmentation Layers
        augment = keras.Sequential([
            layers.RandomRotation(0.1, fill_mode="constant", fill_value=0.0),
            layers.RandomTranslation(0.1, 0.1, fill_mode="constant", fill_value=0.0),
            layers.RandomZoom(0.1, 0.1, fill_mode="constant", fill_value=0.0),
            # layers.GaussianNoise(0.05) # Optional
        ], name="augmentation")

        model = models.Sequential([
            layers.Input(shape=MODEL_INPUT_SHAPE), # (28, 28, 1)
            augment, # Applied only during training
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)), # -> (14, 14, 32)
            layers.Dropout(0.25),

            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)), # -> (7, 7, 64)
            layers.Dropout(0.25),

            layers.Flatten(), # -> (3136)
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(NUM_CLASSES, activation='softmax') # 11 classes
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
        print("Calculating class weights from initial sample...")
        temp_renderer = SudokuRenderer()
        # Generate a decent sample size to estimate weights
        initial_sample_target = batch_size * 50 # Target ~50 batches worth
        sample_labels = []
        temp_gen = sudoku_data_generator(temp_renderer, batch_size, self._preprocess_cell_for_model, self._model_input_size)
        generated_count = 0
        max_gen_attempts = initial_sample_target * 2 # Safety break

        # Collect initial sample labels robustly
        while len(sample_labels) < initial_sample_target and generated_count < max_gen_attempts:
             try:
                 _, y_sample_batch = next(temp_gen)
                 sample_labels.extend(y_sample_batch.tolist())
                 generated_count += len(y_sample_batch) # Count actual items generated
             except StopIteration:
                 print("[Warning] Generator stopped unexpectedly during sample collection.")
                 break
             except Exception as e:
                 print(f"[Warning] Error in generator during sample collection: {e}")
                 generated_count += batch_size # Assume a batch worth of attempts failed if error

        del temp_gen
        del temp_renderer
        gc.collect()

        if not sample_labels:
             print("[Error] Failed to collect any labels for class weight calculation. Aborting.")
             return

        sample_labels = np.array(sample_labels)
        unique_labels_in_sample, counts_in_sample = np.unique(sample_labels, return_counts=True)
        print(f"Labels found in initial sample ({len(sample_labels)} items): {dict(zip(unique_labels_in_sample, counts_in_sample))}")

        if len(unique_labels_in_sample) == 0:
             print("[Error] No valid labels found in sample. Aborting.")
             return

        # --- Corrected Class Weight Calculation ---
        # Calculate weights ONLY for the classes present in the sample
        weights_calculated = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=unique_labels_in_sample, # Use labels actually present
            y=sample_labels
        )
        # Create a temporary dictionary mapping present labels to their calculated weights
        temp_weights_dict = dict(zip(unique_labels_in_sample, weights_calculated))

        # Create the final dictionary, ensuring all NUM_CLASSES (0-10) are present
        # Use a default weight of 1.0 for any class completely missed in the sample
        class_weights_dict = {}
        all_possible_labels = np.arange(NUM_CLASSES) # 0 to 10
        for label in all_possible_labels:
            class_weights_dict[label] = temp_weights_dict.get(label, 1.0) # Default to 1.0 if missing

        print(f"Final Class Weights (defaults applied for missing): {class_weights_dict}")
        # --- End Corrected Class Weight Calculation ---


        # Create separate renderers for train/val
        train_renderer = SudokuRenderer()
        val_renderer = SudokuRenderer() # Could use different params/seed if desired

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
                class_weight=class_weights_dict, # Use the final dict with defaults
                verbose=1 # Show progress bar per epoch
            )
        except Exception as e:
             print(f"\n[Error] An exception occurred during model.fit with generator: {e}")
             import traceback
             traceback.print_exc()
             print("Aborting training.")
             # Clean up generators explicitly? Might help release resources.
             del train_generator
             del val_generator
             gc.collect()
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
                    verbose=0 # Set to 1 for progress bar
                )
                print(f"\nFinal Validation Loss: {loss:.4f}")
                print(f"Final Validation Accuracy: {accuracy:.4f}")
                del eval_val_generator # Clean up generator
            except Exception as e:
                 print(f"[Error] Failed to evaluate model: {e}")
        else:
            print("[Error] Model object is None after training. Cannot evaluate.")
            # Clean up main generators if evaluation fails here
            del train_generator
            del val_generator
            gc.collect()
            return

        # Explicitly save the final best model (might be redundant but safe)
        if self.model:
            try:
                self.model.save(self.model_path)
                print(f"Final best model saved successfully to {self.model_path}")
            except Exception as e:
                print(f"[Error] Failed to save the final model: {e}")

        # Clean up generators after successful training/evaluation
        del train_generator
        del val_generator
        gc.collect()


    @torch.no_grad() # Disable gradient calculations for inference (PyTorch backend)
    def recognise(self, cell_image, confidence_threshold=0.7):
        """
        Recognises the digit in a single cell image.

        Returns:
            tuple: (int: digit, float: confidence)
                   Digit is 0-9 (0 for empty/low confidence).
                   Confidence is the probability of the returned class (0.0 to 1.0).
        """
        if self.model is None:
            print("[Error] Model not loaded or trained. Cannot recognise.")
            return 0, 0.0 # Return digit 0, confidence 0

        # 1. Preprocess (pass is_training=False for debug saving)
        processed_cell = self._preprocess_cell_for_model(cell_image, is_training=False)

        # --- Verification Step ---
        if processed_cell.shape != self._model_input_size:
             print(f"[ERROR in recognise] Preprocessed cell has wrong shape: {processed_cell.shape}. Expected {self._model_input_size}. Returning 0.")
             return 0, 0.0
        # ---

        # 2. Prepare for model input (add batch and channel dimensions)
        model_input = np.expand_dims(processed_cell, axis=(0, -1)) # Add batch & channel -> (1, 28, 28, 1)

        # Convert to PyTorch tensor if using torch backend
        if keras.backend.backend() == 'torch':
            try:
                model_input_tensor = torch.from_numpy(model_input).float()
                # If using GPU: model_input_tensor = model_input_tensor.to(next(self.model.parameters()).device)
            except Exception as e:
                 print(f"[Error] Failed converting NumPy to Torch tensor: {e}")
                 return 0, 0.0
        else: # TensorFlow or other backend
             model_input_tensor = model_input # Keras handles NumPy directly

        # 3. Predict probabilities
        try:
            probabilities = self.model(model_input_tensor, training=False)[0] # Get probabilities for the first (only) item in batch
        except Exception as e:
             print(f"[Error] Exception during model prediction: {e}")
             # Potentially log input shape: print(f"Input tensor shape: {model_input_tensor.shape}")
             return 0, 0.0

        # Convert back to NumPy if it's a tensor
        if isinstance(probabilities, torch.Tensor):
            # If using GPU: probabilities = probabilities.cpu()
            probabilities = probabilities.numpy()

        # 4. Interpret results
        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])

        # 5. Decision Logic
        if predicted_class == EMPTY_LABEL:
            # If it predicts 'empty', return 0 with that confidence
            return 0, confidence

        if confidence < confidence_threshold:
            # If confidence for a digit (0-9) is too low, return 0
            # Confidence returned is still the (low) confidence of the predicted digit class
            return 0, confidence

        # Otherwise, return the predicted digit (0-9) and its confidence
        return predicted_class, confidence

# --- Example Usage (for training) ---
if __name__ == "__main__":
    print("Testing DigitClassifier training (v3 - Generator)...")
    # Force training by setting training_required=True or deleting the model file
    force_train = False # Set to True to force retraining
    model_file = Path(MODEL_FILENAME) # Use v3 filename
    if force_train and model_file.exists():
        print("Forcing retraining, removing existing model file...")
        try:
            model_file.unlink()
        except OSError as e:
            print(f"Error removing model file: {e}")

    classifier = DigitClassifier() # Will load or announce training needed

    if classifier.model is None: # Check if training is needed
        # Train using the generator approach with default steps/epochs
        # Use slightly reduced params for a quicker test if needed:
        # classifier.train(epochs=10, steps_per_epoch=100, validation_steps=20)
        classifier.train() # Use defaults defined in constants
    else:
        print("Model already exists and loaded. Skipping training.")
        # Optional: Test recognition on a sample cell if needed
        # test_cell_path = "extracted_cells/cell_0_0.png" # Example cell path
        # if Path(test_cell_path).exists():
        #      test_cell = cv2.imread(test_cell_path)
        #      if test_cell is not None:
        #          digit, conf = classifier.recognise(test_cell)
        #          print(f"Test recognition on {test_cell_path}: Digit={digit}, Conf={conf:.3f}")
        # else:
        #      print(f"Test cell image not found at {test_cell_path}")

    print("\nClassifier test complete.")