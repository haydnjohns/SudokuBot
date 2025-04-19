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
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# Import necessary components from other modules
from sudoku_renderer import SudokuRenderer
# Import specific functions needed for training data generation
from digit_extractor import rectify_grid, split_into_cells, _order_points

# --- Constants ---
MODEL_FILENAME = "sudoku_digit_classifier_cnn_v2.keras" # New model name
MODEL_INPUT_SHAPE = (28, 28, 1) # Grayscale images, 28x28 pixels
NUM_CLASSES = 11 # Digits 0-9 + Empty class
EMPTY_LABEL = 10 # Label for the empty class
DEFAULT_TRAIN_SAMPLES = 10000 # Increase number of training Sudokus
DEFAULT_VAL_SPLIT = 0.15
TARGET_CELL_CONTENT_SIZE = 20 # Target size of digit within the 28x28 frame

class DigitClassifier:
    def __init__(self, model_path=None, training_required=False):
        self.model_path = Path(model_path or Path(__file__).parent / MODEL_FILENAME)
        self.model = None
        self._model_input_size = MODEL_INPUT_SHAPE[:2]

        if not training_required and self.model_path.exists():
            print(f"Loading existing model from: {self.model_path}")
            try:
                # Ensure custom objects (if any) are handled if needed, though unlikely here
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

    def _preprocess_cell_for_model(self, cell_image, is_training=False):
        """
        Prepares a single extracted cell image for model input.
        - Converts to grayscale
        - Applies adaptive threshold
        - Finds largest contour (digit) within the cell
        - Resizes digit with padding to target size (28x28)
        - Normalizes
        """
        if cell_image is None or cell_image.size < 10: # Ignore very small cells
            return np.zeros(self._model_input_size, dtype=np.float32)

        # 1. Grayscale
        if cell_image.ndim == 3: gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        else: gray = cell_image.copy()

        # 2. Adaptive Threshold (Robust settings)
        thresh = cv2.adaptiveThreshold(gray, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, # Digit white, bg black
                                       19, 5) # Slightly larger block, C=5

        # 3. Find Largest Contour within the cell (to isolate digit)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        digit_contour = None
        max_area = 0
        if contours:
            for cnt in contours:
                area = cv2.contourArea(cnt)
                # Filter tiny noise contours and contours touching the border excessively
                x, y, w, h = cv2.boundingRect(cnt)
                if area > 10 and area > max_area and (x > 1 and y > 1 and x+w < thresh.shape[1]-1 and y+h < thresh.shape[0]-1):
                     max_area = area
                     digit_contour = cnt

        # Create a clean canvas centered with the digit
        canvas = np.zeros_like(thresh)
        if digit_contour is not None:
            # Get bounding box of the digit contour
            x, y, w, h = cv2.boundingRect(digit_contour)
            digit_roi = thresh[y:y+h, x:x+w]

            # Calculate padding to place this ROI centered in a TARGET_CELL_CONTENT_SIZE box
            target_size = TARGET_CELL_CONTENT_SIZE
            pad_top = max(0, (target_size - h) // 2)
            pad_bottom = max(0, target_size - h - pad_top)
            pad_left = max(0, (target_size - w) // 2)
            pad_right = max(0, target_size - w - pad_left)

            # Pad the digit ROI
            try:
                padded_roi = cv2.copyMakeBorder(digit_roi, pad_top, pad_bottom, pad_left, pad_right,
                                                cv2.BORDER_CONSTANT, value=0) # Black padding
                # Ensure it's exactly target_size x target_size (handle rounding errors)
                padded_roi = cv2.resize(padded_roi, (target_size, target_size), interpolation=cv2.INTER_AREA)
            except cv2.error as e:
                 # Handle cases where ROI might be empty or invalid after slicing
                 # print(f"[Warning] cv2 error during padding: {e}. Using blank cell.")
                 padded_roi = np.zeros((target_size, target_size), dtype=np.uint8)


            # Place this padded ROI onto the final 28x28 canvas
            final_canvas_size = self._model_input_size[0] # 28
            start_x = (final_canvas_size - target_size) // 2
            start_y = (final_canvas_size - target_size) // 2
            canvas[start_y:start_y+target_size, start_x:start_x+target_size] = padded_roi

        # If no suitable contour found, canvas remains black (empty)

        # DEBUG: Save preprocessed cell before normalization
        # if is_training and random.random() < 0.001: # Save occasionally during training
        #     cv2.imwrite(f"debug_train_preprocessed_{random.randint(1000,9999)}.png", canvas)
        # elif not is_training: # Save during inference
        #      cv2.imwrite(f"debug_infer_preprocessed_{random.randint(1000,9999)}.png", canvas)


        # 4. Normalize to [0, 1] float
        processed = canvas.astype("float32") / 255.0
        return processed

    def _build_cnn_model(self):
        """Defines and compiles the CNN architecture with augmentation."""
        # Data Augmentation Layers
        augment = keras.Sequential([
            layers.RandomRotation(0.1, fill_mode="constant", fill_value=0.0),
            layers.RandomTranslation(0.1, 0.1, fill_mode="constant", fill_value=0.0),
            layers.RandomZoom(0.1, 0.1, fill_mode="constant", fill_value=0.0),
            # layers.GaussianNoise(0.05) # Less noise maybe?
        ], name="augmentation")

        model = models.Sequential([
            layers.Input(shape=MODEL_INPUT_SHAPE),
            # Apply augmentation *only* during training (handled by model.fit)
            augment,
            # Rescaling can happen after augmentation if preferred
            # layers.Rescaling(1./255), # Not needed if input is already 0-1

            layers.Conv2D(32, (3, 3), activation='relu', padding='same'), # Added padding
            layers.BatchNormalization(), # Added Batch Norm
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25), # Added Dropout

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
            layers.Dense(NUM_CLASSES, activation='softmax') # 11 classes
        ])

        # Use Adam optimizer with potentially adjusted learning rate
        optimizer = keras.optimizers.Adam(learning_rate=0.001)

        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        print("CNN Model Summary:")
        model.summary()
        return model

    def train(self, num_samples=DEFAULT_TRAIN_SAMPLES, epochs=25, batch_size=128, val_split=DEFAULT_VAL_SPLIT): # Increased epochs, batch size
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
                rectified_grid = rectify_grid(rendered_img, warped_corners) # Use known corners
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
                processed_cell = self._preprocess_cell_for_model(cell_img, is_training=True)

                all_cells_processed.append(processed_cell)
                all_labels.append(label)

            # Optional: Stop slightly early if we overshoot significantly
            if len(all_cells_processed) >= target_cells * 1.05:
                 break
        # --- End Data Generation Loop ---

        print(f"\nGenerated {len(all_cells_processed)} cell samples from {processed_sudokus} Sudoku images.")

        if not all_cells_processed:
             print("[Error] No training data could be generated. Aborting training.")
             return

        X = np.array(all_cells_processed).astype('float32')
        y = np.array(all_labels).astype('int64')
        X = np.expand_dims(X, -1) # Add channel dimension

        print(f"Dataset shape: X={X.shape}, y={y.shape}")
        unique_labels, counts = np.unique(y, return_counts=True)
        print(f"Label distribution: {dict(zip(unique_labels, counts))}")

        # --- Calculate Class Weights ---
        # Handles imbalance: gives more importance to less frequent classes (digits)
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
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1) # Adjust LR
        ]

        # Train
        print("\nStarting model fitting...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weights_dict, # Apply class weights here
            verbose=1 # Show progress bar
        )

        print("\nTraining finished.")
        if self.model_path.exists():
             print(f"Reloading best model weights from {self.model_path}")
             self.model = keras.saving.load_model(self.model_path)
        else:
             print("[Warning] Best model checkpoint file not found. Using final weights.")

        loss, accuracy = self.model.evaluate(X_val, y_val, verbose=0)
        print(f"\nFinal Validation Loss: {loss:.4f}")
        print(f"Final Validation Accuracy: {accuracy:.4f}")

        try:
            self.model.save(self.model_path)
            print(f"Model saved successfully to {self.model_path}")
        except Exception as e:
            print(f"[Error] Failed to save the final model: {e}")


    @torch.no_grad()
    def recognise(self, cell_image, confidence_threshold=0.7): # Slightly lower default threshold?
        """
        Recognises the digit in a single cell image.

        Returns:
            tuple: (int: digit, float: confidence)
                   Digit is 0-9 (0 for empty/low confidence).
                   Confidence is the probability of the returned class (0.0 to 1.0).
        """
        if self.model is None:
            print("[Error] Model not loaded or trained.")
            return 0, 0.0 # Return digit 0, confidence 0

        # 1. Preprocess (pass is_training=False for debug saving)
        processed_cell = self._preprocess_cell_for_model(cell_image, is_training=False)

        # 2. Prepare for model
        model_input = np.expand_dims(processed_cell, axis=(0, -1)) # Add batch & channel

        # Convert to Tensor if needed
        if keras.backend.backend() == 'torch':
            model_input_tensor = torch.from_numpy(model_input).float()
            # Add GPU transfer if applicable: .to(next(self.model.parameters()).device)
        else:
            model_input_tensor = model_input

        # 3. Predict
        probabilities = self.model(model_input_tensor, training=False)[0]

        if isinstance(probabilities, torch.Tensor):
            # Add CPU transfer if applicable: .cpu()
            probabilities = probabilities.numpy()

        # 4. Interpret
        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])

        # 5. Decision Logic
        if predicted_class == EMPTY_LABEL:
            # If it confidently predicts 'empty', return 0 with that confidence
            return 0, confidence

        if confidence < confidence_threshold:
            # If confidence for a digit (0-9) is too low, return 0
            # Confidence returned is still the (low) confidence of the predicted digit class
            return 0, confidence

        # Otherwise, return the predicted digit (0-9) and its confidence
        return predicted_class, confidence

# (Keep __main__ block, ensure it uses the new MODEL_FILENAME)
if __name__ == "__main__":
    print("Testing DigitClassifier training (v2)...")
    force_train = False # Set to True to force retraining
    model_file = Path(MODEL_FILENAME) # Use new filename
    if force_train and model_file.exists():
        print("Forcing retraining, removing existing model file...")
        model_file.unlink()

    classifier = DigitClassifier()

    if classifier.model is None:
        # Train with more samples/epochs for a better result
        # classifier.train(num_samples=10000, epochs=25, batch_size=128) # Example real training
        # For a quicker test:
        classifier.train(num_samples=1000, epochs=10, batch_size=64) # Reduced for testing
    else:
        print("Model already exists and loaded. Skipping training.")
        # Optional: Test recognition
        # test_cell_path = "extracted_cells/cell_0_0.png" # Example cell path
        # if Path(test_cell_path).exists():
        #      test_cell = cv2.imread(test_cell_path)
        #      if test_cell is not None:
        #          digit, conf = classifier.recognise(test_cell)
        #          print(f"Test recognition on {test_cell_path}: Digit={digit}, Conf={conf:.3f}")
        # else:
        #      print(f"Test cell image not found at {test_cell_path}")


    print("\nClassifier test complete.")