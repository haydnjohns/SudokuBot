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

# Import necessary components from other modules
from sudoku_renderer import SudokuRenderer
from digit_extractor import extract_cells_from_image

# --- Constants ---
MODEL_FILENAME = "sudoku_digit_classifier_cnn.keras"
MODEL_INPUT_SHAPE = (28, 28, 1) # Grayscale images, 28x28 pixels
NUM_CLASSES = 11 # Digits 0-9 + Empty class
EMPTY_LABEL = 10 # Label for the empty class
DEFAULT_TRAIN_SAMPLES = 5000 # Number of synthetic Sudokus to generate for training
DEFAULT_VAL_SPLIT = 0.15
GRID_SIZE = 9

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
        self._model_input_size = MODEL_INPUT_SHAPE[:2] # (height, width)

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

    def _preprocess_cell_for_model(self, cell_image):
        """
        Prepares a single extracted cell image for model input.
        Input should be a raw cell image (likely BGR or Gray).
        """
        if cell_image is None or cell_image.size == 0:
            # print("[Debug] Preprocessing None or empty cell image.")
            # Return a blank image of the correct size/type
            return np.zeros(self._model_input_size, dtype=np.float32)

        # 1. Convert to Grayscale
        if cell_image.ndim == 3:
            gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell_image

        # 2. Adaptive Thresholding (Crucial for normalizing contrast and isolating digit)
        # Use settings robust to noise and varying line thickness
        thresh = cv2.adaptiveThreshold(gray, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, # Invert: Digit is white, background black
                                       15, 5) # Block size and C value - may need tuning

        # 3. Optional: Noise Reduction / Morphology
        # A small opening operation can remove salt-and-pepper noise
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # 4. Resize to Model Input Size
        # Ensure interpolation handles shrinking well
        resized = cv2.resize(thresh, self._model_input_size, interpolation=cv2.INTER_AREA)

        # 5. Normalize to [0, 1] float
        processed = resized.astype("float32") / 255.0

        return processed

    def _build_cnn_model(self):
        """Defines and compiles the CNN architecture."""
        model = models.Sequential([
            layers.Input(shape=MODEL_INPUT_SHAPE),

            # Data Augmentation (applied during training only)
            # Note: Augmentation is often placed *after* Rescaling if applied on GPU
            # layers.RandomRotation(0.1, fill_mode="constant", fill_value=0.0),
            # layers.RandomTranslation(0.1, 0.1, fill_mode="constant", fill_value=0.0),
            # layers.RandomZoom(0.1, 0.1, fill_mode="constant", fill_value=0.0),

            # Consider adding augmentation directly in the training loop if more control is needed
            # or if using tf.data pipelines. For simplicity here, we might skip it initially
            # or add it later.

            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'), # Added another conv layer
            layers.Flatten(),
            layers.Dense(128, activation='relu'), # Increased dense layer size
            layers.Dropout(0.5),
            layers.Dense(NUM_CLASSES, activation='softmax') # 11 classes (0-9 + empty)
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        print("CNN Model Summary:")
        model.summary()
        return model

    def train(self, num_samples=DEFAULT_TRAIN_SAMPLES, epochs=15, batch_size=64, val_split=DEFAULT_VAL_SPLIT):
        """
        Trains the digit classifier using synthetically generated Sudoku data.

        Args:
            num_samples: Number of synthetic Sudoku images to generate.
            epochs: Number of training epochs.
            batch_size: Training batch size.
            val_split: Fraction of generated data to use for validation.
        """
        print(f"\n--- Starting Training ---")
        print(f"Generating {num_samples} synthetic Sudoku images for training data...")

        renderer = SudokuRenderer() # Use default settings
        all_cells = []
        all_labels = []
        processed_count = 0
        generated_count = 0

        while processed_count < num_samples * (GRID_SIZE * GRID_SIZE):
            generated_count += 1
            if generated_count % 100 == 0:
                 print(f"  Generated {generated_count} images, processed {processed_count} cells...")

            # 1. Generate a synthetic Sudoku image and its ground truth
            # Ensure a good mix of empty and filled cells
            allow_empty = random.random() < 0.8 # Generate grids with empties 80% of the time
            rendered_img, gt_grid, _ = renderer.render_sudoku(allow_empty=allow_empty)

            # 2. Extract cells from the rendered image using the extractor
            # This simulates the actual process the recognizer will use
            extracted_cells, _, _ = extract_cells_from_image(rendered_img)

            if extracted_cells is None:
                print("[Warning] Failed to extract cells from a generated image. Skipping.")
                continue

            if len(extracted_cells) != GRID_SIZE * GRID_SIZE:
                 print(f"[Warning] Extracted {len(extracted_cells)} cells instead of {GRID_SIZE*GRID_SIZE}. Skipping.")
                 continue

            # 3. Pair extracted cells with ground truth labels
            gt_labels_flat = gt_grid.flatten() # Shape (81,)

            for i, cell_img in enumerate(extracted_cells):
                label = gt_labels_flat[i]
                if label == 0:
                    label = EMPTY_LABEL # Assign the special label for empty cells

                # 4. Preprocess the extracted cell for the model
                processed_cell = self._preprocess_cell_for_model(cell_img)

                all_cells.append(processed_cell)
                all_labels.append(label)
                processed_count += 1

            # Stop if we have enough samples (can overshoot slightly)
            if processed_count >= num_samples * (GRID_SIZE * GRID_SIZE):
                break

        print(f"\nGenerated {processed_count} cell samples from {generated_count} Sudoku images.")

        if not all_cells:
             print("[Error] No training data could be generated. Aborting training.")
             return

        # Convert to NumPy arrays
        X = np.array(all_cells).astype('float32')
        y = np.array(all_labels).astype('int64')

        # Add channel dimension for CNN input
        X = np.expand_dims(X, -1)

        print(f"Dataset shape: X={X.shape}, y={y.shape}")
        print(f"Label distribution: {np.unique(y, return_counts=True)}")

        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_split, random_state=42, stratify=y # Stratify helps with class imbalance
        )
        print(f"Training set size: {len(X_train)}")
        print(f"Validation set size: {len(X_val)}")

        # Build the model
        self.model = self._build_cnn_model()

        # Define callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(self.model_path, monitor='val_loss', save_best_only=True, verbose=1)
            # ReduceLROnPlateau could also be useful
            # keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001, verbose=1)
        ]

        # Train the model
        print("\nStarting model fitting...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=2 # Use 1 for progress bar, 2 for one line per epoch
        )

        print("\nTraining finished.")

        # Load the best weights saved by ModelCheckpoint
        if self.model_path.exists():
             print(f"Reloading best model weights from {self.model_path}")
             self.model = keras.saving.load_model(self.model_path)
        else:
             print("[Warning] Best model checkpoint file not found. Using final weights.")


        # Evaluate the final (best) model on the validation set
        loss, accuracy = self.model.evaluate(X_val, y_val, verbose=0)
        print(f"\nFinal Validation Loss: {loss:.4f}")
        print(f"Final Validation Accuracy: {accuracy:.4f}")

        # Save the final best model explicitly (redundant if ModelCheckpoint worked, but safe)
        try:
            self.model.save(self.model_path)
            print(f"Model saved successfully to {self.model_path}")
        except Exception as e:
            print(f"[Error] Failed to save the final model: {e}")


    @torch.no_grad() # Disable gradient calculations for inference (PyTorch backend)
    def recognise(self, cell_image, confidence_threshold=0.8):
        """
        Recognises the digit in a single cell image.

        Args:
            cell_image (np.ndarray): The image of the cell (BGR or Grayscale).
            confidence_threshold (float): Minimum probability for a prediction to be accepted.

        Returns:
            int: The recognised digit (1-9), or 0 if predicted as empty or below threshold.
        """
        if self.model is None:
            print("[Error] Model not loaded or trained. Cannot recognise.")
            # Attempt to load or train? For now, just return 0.
            # Or raise an exception: raise RuntimeError("Model is not available.")
            return 0

        # 1. Preprocess the cell
        processed_cell = self._preprocess_cell_for_model(cell_image)

        # DEBUG: Save preprocessed cell
        # cv2.imwrite(f"debug_processed_cell_{random.randint(1000,9999)}.png", (processed_cell * 255).astype(np.uint8))

        # 2. Prepare for model input (add batch and channel dimensions)
        model_input = np.expand_dims(processed_cell, axis=0) # Add batch dim -> (1, H, W)
        model_input = np.expand_dims(model_input, axis=-1)   # Add channel dim -> (1, H, W, 1)

        # Convert to PyTorch tensor if using torch backend
        if keras.backend.backend() == 'torch':
            model_input_tensor = torch.from_numpy(model_input).float()
            # If using GPU: model_input_tensor = model_input_tensor.to(next(self.model.parameters()).device)
        else: # TensorFlow or other backend
             model_input_tensor = model_input # Keras handles NumPy directly

        # 3. Predict probabilities
        probabilities = self.model(model_input_tensor, training=False)[0] # Get probabilities for the first (only) item in batch

        # Convert back to NumPy if it's a tensor
        if isinstance(probabilities, torch.Tensor):
            # If using GPU: probabilities = probabilities.cpu()
            probabilities = probabilities.numpy()

        # 4. Interpret results
        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])

        # print(f"[Debug] Pred Class: {predicted_class}, Conf: {confidence:.3f}, Probs: {np.round(probabilities, 2)}") # Debug

        # 5. Decision Logic
        if predicted_class == EMPTY_LABEL:
            # print(f"  -> Predicted Empty (Class {EMPTY_LABEL})")
            return 0 # Return 0 for the empty class

        if confidence < confidence_threshold:
            # print(f"  -> Low Confidence ({confidence:.3f} < {confidence_threshold})")
            return 0 # Return 0 if confidence is too low

        # Otherwise, return the predicted digit (predicted_class is 0-9 here)
        # print(f"  -> Predicted Digit: {predicted_class}")
        return predicted_class

# --- Example Usage (for training) ---
if __name__ == "__main__":
    print("Testing DigitClassifier training...")
    # Force training by setting training_required=True or deleting the model file
    force_train = False # Set to True to retrain even if model exists
    model_file = Path(MODEL_FILENAME)
    if force_train and model_file.exists():
        print("Forcing retraining, removing existing model file...")
        model_file.unlink()

    classifier = DigitClassifier() # Will load or announce training needed

    if classifier.model is None: # Check if training is needed
        # Train with fewer samples/epochs for a quick test
        classifier.train(num_samples=500, epochs=5, batch_size=32) # Reduced for testing
        # For real training, use defaults or larger values:
        # classifier.train()
    else:
        print("Model already exists and loaded. Skipping training.")
        # Optional: Test recognition on a sample cell if needed
        # test_cell = cv2.imread("extracted_cells/cell_0_0.png") # Example cell
        # if test_cell is not None:
        #     prediction = classifier.recognise(test_cell)
        #     print(f"Test recognition on cell_0_0.png: {prediction}")

    print("\nClassifier test complete.")