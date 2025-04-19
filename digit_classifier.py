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
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# Import necessary components from other modules
from sudoku_renderer import SudokuRenderer
# Import specific functions needed for training data generation
from digit_extractor import rectify_grid, split_into_cells, _order_points # _order_points might not be needed here but safe to keep

# --- Constants ---
MODEL_FILENAME = "sudoku_digit_classifier_cnn_v2.keras" # Keep v2 name
MODEL_INPUT_SHAPE = (28, 28, 1) # Grayscale images, 28x28 pixels
NUM_CLASSES = 11 # Digits 0-9 + Empty class
EMPTY_LABEL = 10 # Label for the empty class
DEFAULT_TRAIN_SAMPLES = 10000 # Number of synthetic Sudokus for training
DEFAULT_VAL_SPLIT = 0.15
GRID_SIZE = 9 # Size of the Sudoku grid
TARGET_CELL_CONTENT_SIZE = 20 # Target size of digit within the 28x28 frame

class DigitClassifier:
    """
    Trains and uses a CNN model to classify Sudoku cell images (0-9 or empty).
    """
    def __init__(self, model_path=None, training_required=False):
        """
        Initializes the classifier. Loads an existing model or prepares for training.

        Args:
            model_path (str | Path | None): Path to the pre-trained model file.
                If None, defaults to MODEL_FILENAME in the same directory.
            training_required (bool): If True, forces training even if a model file exists.
        """
        self.model_path = Path(model_path or Path(__file__).parent / MODEL_FILENAME)
        self.model = None
        self._model_input_size = MODEL_INPUT_SHAPE[:2] # (height, width) -> (28, 28)

        if not training_required and self.model_path.exists():
            print(f"Loading existing model from: {self.model_path}")
            try:
                self.model = keras.saving.load_model(self.model_path)
                # Verify input shape compatibility (optional but good practice)
                if self.model.input_shape[1:3] != self._model_input_size:
                     print(f"[Warning] Loaded model input shape {self.model.input_shape[1:3]} "
                           f"differs from expected {self._model_input_size}. Mismatches may occur.")
                print("Model loaded successfully.")
            except Exception as e:
                print(f"[Error] Failed to load model: {e}. Will attempt training.")
                self.model = None # Ensure model is None if loading failed
        else:
            if training_required:
                print("Training explicitly required.")
            else:
                print(f"Model not found at {self.model_path}. Training is required.")
            # Model will be created during training

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
            # Return a blank 28x28 image
            return np.zeros((target_h, target_w), dtype=np.float32)

        # 1. Grayscale
        if cell_image.ndim == 3: gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        else: gray = cell_image.copy()

        # --- Adaptive Thresholding Robustness ---
        cell_h_orig, cell_w_orig = gray.shape[:2]
        # Ensure block size is odd and less than min dimension, prevent errors on small cells
        # Calculate a reasonable block size, ensuring it's odd and >= 3
        min_dim = min(cell_h_orig, cell_w_orig)
        block_size = min(19, min_dim - 1 if min_dim % 2 == 0 else min_dim - 2) # Max block size 19 or less
        block_size = max(3, block_size) # Min block size 3
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
                # Filter tiny noise and contours touching border (adjust thresholds if needed)
                # Check area and ensure it's not touching the border (x>0, y>0 etc.)
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
                    # Ensure slicing doesn't go out of bounds (important!)
                    # Calculate end coordinates, clamping to canvas dimensions
                    end_y = min(pad_top + new_h, target_h)
                    end_x = min(pad_left + new_w, target_w)

                    # Calculate ROI slice dimensions based on available space
                    roi_h_slice = end_y - pad_top
                    roi_w_slice = end_x - pad_left

                    # Ensure the slice dimensions match the part of resized_roi we take
                    final_canvas[pad_top:end_y, pad_left:end_x] = resized_roi[:roi_h_slice, :roi_w_slice]

                except cv2.error as e:
                    # This might happen if contour/ROI is degenerate after calculations
                    # print(f"[Warning] cv2 error during resize/paste: {e}. Cell remains blank.")
                    pass # final_canvas remains black

        # DEBUG saving (optional)
        # if is_training and random.random() < 0.001:
        #     cv2.imwrite(f"debug_train_preprocessed_{random.randint(1000,9999)}.png", final_canvas)
        # elif not is_training:
        #      cv2.imwrite(f"debug_infer_preprocessed_{random.randint(1000,9999)}.png", final_canvas)

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
            # layers.GaussianNoise(0.05) # Optional: Add noise augmentation
        ], name="augmentation")

        model = models.Sequential([
            layers.Input(shape=MODEL_INPUT_SHAPE), # Should be (28, 28, 1)
            # Apply augmentation *only* during training (handled by model.fit)
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

            layers.Flatten(), # Flattens (None, 7, 7, 64) -> (None, 3136) if input is 28x28
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(NUM_CLASSES, activation='softmax') # 11 classes
        ])

        # Use Adam optimizer with potentially adjusted learning rate
        optimizer = keras.optimizers.Adam(learning_rate=0.001)

        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        print("CNN Model Summary:")
        model.summary() # Print summary to verify layer output shapes
        return model

    def train(self, num_samples=DEFAULT_TRAIN_SAMPLES, epochs=25, batch_size=128, val_split=DEFAULT_VAL_SPLIT):
        """
        Trains the classifier using synthetic data, extracting cells via ground truth corners.
        """
        print(f"\n--- Starting Training (v2) ---")
        print(f"Generating data from ~{num_samples} synthetic Sudokus...")

        renderer = SudokuRenderer() # Use default settings
        all_cells_processed = []
        all_labels = []
        processed_sudokus = 0
        target_cells = num_samples * GRID_SIZE * GRID_SIZE

        # --- Use Ground Truth for Training Data Extraction ---
        while len(all_cells_processed) < target_cells:
            processed_sudokus += 1
            if processed_sudokus % 200 == 0:
                 print(f"  Generated {processed_sudokus} images, processed {len(all_cells_processed)} cells...")

            allow_empty = random.random() < 0.8
            # Get image, ground truth grid, AND the warped corners
            rendered_img, gt_grid, warped_corners = renderer.render_sudoku(allow_empty=allow_empty)

            if rendered_img is None or warped_corners is None:
                print("[Warning] Renderer failed to produce image or corners. Skipping.")
                continue

            # --- Bypass find_sudoku_grid_contour ---
            # 1. Rectify using known corners
            try:
                # Use default rectified size (450) for consistency
                rectified_grid = rectify_grid(rendered_img, warped_corners)
            except Exception as e:
                 print(f"[Warning] Failed to rectify grid using known corners: {e}. Skipping.")
                 continue

            # 2. Split into cells
            extracted_cells, _ = split_into_cells(rectified_grid)
            # --- End Bypass ---

            if len(extracted_cells) != GRID_SIZE * GRID_SIZE:
                 print(f"[Warning] Extracted {len(extracted_cells)} cells instead of {GRID_SIZE*GRID_SIZE} after rectify/split. Skipping.")
                 continue

            gt_labels_flat = gt_grid.flatten()

            for i, cell_img in enumerate(extracted_cells):
                label = gt_labels_flat[i]
                if label == 0: label = EMPTY_LABEL

                # Preprocess cell (pass is_training=True for debug saving if enabled)
                # This should now return a (28, 28) array
                processed_cell = self._preprocess_cell_for_model(cell_img, is_training=True)

                # --- Verification Step (Optional but recommended) ---
                if processed_cell.shape != self._model_input_size:
                     print(f"[CRITICAL ERROR] Preprocessed cell has wrong shape: {processed_cell.shape}. Expected {self._model_input_size}. Skipping cell.")
                     continue # Skip this cell to avoid crashing training
                # --- End Verification ---


                all_cells_processed.append(processed_cell)
                all_labels.append(label)

            # Optional: Stop slightly early if we overshoot significantly
            if len(all_cells_processed) >= target_cells * 1.05:
                 break
        # --- End Data Generation Loop ---

        print(f"\nGenerated {len(all_cells_processed)} cell samples from {processed_sudokus} Sudoku images.")

        if not all_cells_processed:
             print("[Error] No training data could be generated or processed correctly. Aborting training.")
             return

        X = np.array(all_cells_processed).astype('float32')
        y = np.array(all_labels).astype('int64')

        # --- CRITICAL: Ensure correct shape before adding channel dimension ---
        if X.ndim != 3 or X.shape[1:] != self._model_input_size:
             print(f"[CRITICAL ERROR] Final training data array X has wrong shape: {X.shape}. Expected (N, {self._model_input_size[0]}, {self._model_input_size[1]}). Aborting.")
             return
        # ---

        X = np.expand_dims(X, -1) # Add channel dimension -> (N, 28, 28, 1)

        print(f"Dataset shape: X={X.shape}, y={y.shape}")
        unique_labels, counts = np.unique(y, return_counts=True)
        print(f"Label distribution: {dict(zip(unique_labels, counts))}")

        # --- Calculate Class Weights ---
        weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y),
            y=y
        )
        class_weights_dict = dict(zip(np.unique(y), weights))
        print(f"Calculated Class Weights: {class_weights_dict}")
        # ---

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_split, random_state=42, stratify=y
        )
        print(f"Training set size: {len(X_train)}")
        print(f"Validation set size: {len(X_val)}")

        # Build model
        self.model = self._build_cnn_model()

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(self.model_path, monitor='val_loss', save_best_only=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
        ]

        # Train
        print("\nStarting model fitting...")
        try:
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                class_weight=class_weights_dict, # Apply class weights here
                verbose=1 # Show progress bar
            )
        except Exception as e:
             print(f"\n[Error] An exception occurred during model.fit: {e}")
             import traceback
             traceback.print_exc()
             print("Aborting training.")
             return # Stop if fitting fails

        print("\nTraining finished.")
        # Ensure model is loaded from the best checkpoint
        if self.model_path.exists():
             print(f"Reloading best model weights from {self.model_path}")
             try:
                 self.model = keras.saving.load_model(self.model_path)
             except Exception as e:
                  print(f"[Error] Failed to reload best model: {e}. Using final weights (might be suboptimal).")
                  # Keep the model object as it was at the end of training
        else:
             print("[Warning] Best model checkpoint file not found. Using final weights.")


        # Evaluate the final (best loaded or last) model
        if self.model:
            try:
                loss, accuracy = self.model.evaluate(X_val, y_val, verbose=0)
                print(f"\nFinal Validation Loss: {loss:.4f}")
                print(f"Final Validation Accuracy: {accuracy:.4f}")
            except Exception as e:
                 print(f"[Error] Failed to evaluate model: {e}")
        else:
            print("[Error] Model object is None after training. Cannot evaluate.")
            return # Cannot proceed without a model


        # Save the final best model explicitly (if loaded successfully)
        if self.model and self.model_path.exists(): # Only save if we successfully loaded the best one
            try:
                # Re-saving might be redundant if ModelCheckpoint worked perfectly, but safe
                self.model.save(self.model_path)
                print(f"Best model re-saved successfully to {self.model_path}")
            except Exception as e:
                print(f"[Error] Failed to re-save the final best model: {e}")


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
            # If it confidently predicts 'empty', return 0 with that confidence
            # We might still apply the threshold if we only want *very* confident empty cells
            # if confidence >= confidence_threshold:
            #      return 0, confidence
            # else:
            #      return 0, confidence # Return 0 anyway, but maybe lower confidence? Or just return 0, conf
            return 0, confidence # Simplest: return 0 if predicted empty

        if confidence < confidence_threshold:
            # If confidence for a digit (0-9) is too low, return 0
            # Confidence returned is still the (low) confidence of the predicted digit class
            return 0, confidence

        # Otherwise, return the predicted digit (0-9) and its confidence
        return predicted_class, confidence

# --- Example Usage (for training) ---
if __name__ == "__main__":
    print("Testing DigitClassifier training (v2)...")
    # Force training by setting training_required=True or deleting the model file
    force_train = False # Set to True to retrain even if model exists
    model_file = Path(MODEL_FILENAME) # Use new filename
    if force_train and model_file.exists():
        print("Forcing retraining, removing existing model file...")
        try:
            model_file.unlink()
        except OSError as e:
            print(f"Error removing model file: {e}")


    classifier = DigitClassifier() # Will load or announce training needed

    if classifier.model is None: # Check if training is needed
        # Train with fewer samples/epochs for a quick test
        # classifier.train(num_samples=1000, epochs=10, batch_size=64) # Reduced for testing
        # For real training, use defaults or larger values:
        classifier.train() # Use defaults: 10k samples, 25 epochs, batch 128
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