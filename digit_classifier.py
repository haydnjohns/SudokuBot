# digit_classifier.py
import os
# Set Keras backend *before* importing Keras
os.environ["KERAS_BACKEND"] = "torch" # Or "tensorflow"

import cv2
import numpy as np
from pathlib import Path
import keras
from keras import layers, models
import torch # Explicit import for backend functions like torch.no_grad
import random
import time

# Try importing necessary components from sibling modules
try:
    from sudoku_renderer import SudokuRenderer
    from digit_extractor import extract_digits
except ImportError:
    print("[ERROR] Cannot import SudokuRenderer or digit_extractor.")
    print("        Ensure sudoku_renderer.py and digit_extractor.py are in the same directory.")
    # Define dummy classes/functions if imports fail, to allow script loading
    # This is not ideal but prevents immediate crash if run standalone without siblings
    class SudokuRenderer: pass
    def extract_digits(*args, **kwargs): return [], None, None


# --- Constants ---
_MODEL_FNAME = "sudoku_digit_classifier_cnn.keras"
_INPUT_SHAPE = (28, 28) # Target size for classification input
_NUM_CLASSES = 11 # Digits 0-9 + class 10 for empty
_EMPTY_CLASS_LABEL = 10

class DigitClassifier:
    """
    Trains and uses a CNN model to classify Sudoku cell images (0-9 or empty).

    Uses SudokuRenderer and digit_extractor to generate training data.
    """
    def __init__(self, model_filename=_MODEL_FNAME, input_shape=_INPUT_SHAPE):
        self.model_file = Path(__file__).resolve().parent / model_filename
        self.input_shape = input_shape
        self.num_classes = _NUM_CLASSES
        self.model = None
        self._load_model() # Attempt to load existing model

    def _load_model(self):
        """Loads the Keras model from the specified file."""
        if self.model_file.exists():
            print(f"[INFO] Loading existing model: {self.model_file}")
            try:
                self.model = keras.saving.load_model(self.model_file)
                # Verify input shape compatibility (optional but good practice)
                if self.model.input_shape[1:3] != self.input_shape:
                     print(f"[WARN] Loaded model input shape {self.model.input_shape[1:3]} "
                           f"differs from expected {self.input_shape}. Errors may occur.")
            except Exception as e:
                print(f"[ERROR] Failed to load model {self.model_file}: {e}")
                self.model = None
        else:
            print(f"[INFO] Model file not found: {self.model_file}")
            self.model = None

    def _build_cnn(self):
        """Builds the Keras CNN model."""
        model = models.Sequential([
            layers.Input(shape=(self.input_shape[0], self.input_shape[1], 1)), # Grayscale input
            # Normalize input: Apply thresholding/rescaling *before* model if needed,
            # or use Rescaling layer if model expects 0-1 floats.
            # Let's assume preprocessing handles normalization/thresholding,
            # so the model just gets the processed 28x28 image.
            # layers.Rescaling(1./255), # Optional: if input is 0-255 uint8

            # Data Augmentation (applied during training)
            # Note: Augmentation is less critical if training data is diverse enough
            # layers.RandomRotation(0.1, fill_mode="constant", fill_value=0),
            # layers.RandomTranslation(0.1, 0.1, fill_mode="constant", fill_value=0),

            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'), # Added another conv layer
            layers.Flatten(),
            layers.Dense(128, activation='relu'), # Increased dense layer size
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax') # 11 classes
        ], name="SudokuDigitCNN")

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def preprocess_cell_for_model(self, cell_image):
        """
        Prepares a single cell image (BGR) for model input.
        MUST match the preprocessing used during training data generation.
        """
        if cell_image is None or cell_image.size == 0:
            # Return a blank image of the correct size if input is invalid
            return np.zeros(self.input_shape, dtype=np.uint8)

        # 1. Convert to Grayscale
        if cell_image.ndim == 3 and cell_image.shape[2] == 3:
            gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        elif cell_image.ndim == 2:
            gray = cell_image # Already grayscale
        else: # Invalid shape, return blank
             print(f"Warning: Invalid cell image shape {cell_image.shape}, returning blank.")
             return np.zeros(self.input_shape, dtype=np.uint8)

        # 2. Apply Adaptive Thresholding (to binarize and handle lighting)
        # This helps normalize the appearance. Tune parameters if needed.
        thresh = cv2.adaptiveThreshold(gray, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, # Invert: digit is white, bg is black
                                       15, 5) # Block size, C value - adjust these

        # 3. Optional: Noise removal / Morphological Ops on thresholded image
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # 4. Resize to target input shape (e.g., 28x28)
        # Use INTER_AREA for shrinking, INTER_LINEAR/CUBIC for enlarging
        h, w = gray.shape
        if h > self.input_shape[0] or w > self.input_shape[1]:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LINEAR
        resized = cv2.resize(thresh, self.input_shape, interpolation=interpolation)

        # 5. Ensure output is uint8 (model might expect float32 0-1 later)
        return resized.astype(np.uint8)


    def _generate_training_data(self, num_samples, renderer_options=None, extractor_options=None):
        """Generates training data using renderer and extractor."""
        if not callable(getattr(SudokuRenderer, 'render_sudoku', None)) or \
           not callable(extract_digits):
            print("[ERROR] Renderer or Extractor not available/functional. Cannot generate data.")
            return None, None

        renderer_options = renderer_options if renderer_options else {}
        extractor_options = extractor_options if extractor_options else {}

        # Initialize renderer (could be passed in)
        renderer = SudokuRenderer(**renderer_options)

        X_data = [] # List to hold processed cell images (28x28)
        y_data = [] # List to hold corresponding labels (0-9, or 10 for empty)

        print(f"[INFO] Generating {num_samples} training samples...")
        start_time = time.time()
        generated_count = 0
        attempts = 0
        max_attempts = num_samples * 3 # Stop if generation is too inefficient

        while generated_count < num_samples and attempts < max_attempts:
            attempts += 1
            if attempts % 100 == 0:
                 print(f"    Attempt {attempts}, Generated {generated_count}/{num_samples}...")

            # 1. Render a Sudoku image with known ground truth
            # Vary difficulty to get mix of filled/empty cells
            difficulty = random.uniform(0.2, 0.7)
            rendered_img, truth_grid, _ = renderer.render_sudoku(difficulty=difficulty)

            # 2. Extract cells from the rendered image
            # Use the same options as likely used during inference
            extracted_cells, _, _ = extract_digits(rendered_img, **extractor_options)

            if len(extracted_cells) != 81:
                # print(f"    Skipping image: Expected 81 cells, got {len(extracted_cells)}")
                continue # Skip if extraction failed

            # 3. Process each cell and pair with ground truth label
            for i, cell_img in enumerate(extracted_cells):
                row, col = divmod(i, 9)
                true_label = truth_grid[row, col]

                # Assign the 'empty' label if the ground truth is 0
                label = true_label if true_label != 0 else _EMPTY_CLASS_LABEL

                # Preprocess the cell exactly as done for prediction
                processed_cell = self.preprocess_cell_for_model(cell_img)

                # Optional: Check if cell is mostly blank after processing
                # This helps filter out poorly extracted cells or truly empty ones
                # if np.count_nonzero(processed_cell) < 10 and label != _EMPTY_CLASS_LABEL:
                #      # print(f"    Skipping cell {i}: Low content but not labeled empty.")
                #      continue # Skip potentially bad extractions mislabeled as digits

                X_data.append(processed_cell)
                y_data.append(label)
                generated_count += 1

                if generated_count >= num_samples:
                    break
            if generated_count >= num_samples:
                    break

        if generated_count < num_samples:
             print(f"[WARN] Could only generate {generated_count} samples after {attempts} attempts.")

        end_time = time.time()
        print(f"[INFO] Data generation took {end_time - start_time:.2f} seconds.")

        if not X_data:
             return None, None

        return np.array(X_data), np.array(y_data)


    def train(self,
              num_train_samples=10000,
              num_val_samples=2000,
              epochs=15,
              batch_size=64,
              force_retrain=False,
              renderer_options=None,
              extractor_options=None):
        """
        Trains the CNN model. Generates data if X_train/y_train not provided.

        Args:
            num_train_samples: Number of training samples to generate.
            num_val_samples: Number of validation samples to generate.
            epochs: Training epochs.
            batch_size: Training batch size.
            force_retrain: If True, retrain even if a model file exists.
            renderer_options: Dict of options passed to SudokuRenderer.
            extractor_options: Dict of options passed to extract_digits.
        """
        if self.model and not force_retrain:
            print("[INFO] Model already loaded and force_retrain=False. Skipping training.")
            return

        # --- Generate Data ---
        print("\n--- Generating Training Data ---")
        X_train, y_train = self._generate_training_data(num_train_samples, renderer_options, extractor_options)
        if X_train is None:
            print("[ERROR] Failed to generate training data. Aborting training.")
            return

        print("\n--- Generating Validation Data ---")
        X_val, y_val = self._generate_training_data(num_val_samples, renderer_options, extractor_options)
        if X_val is None:
            print("[ERROR] Failed to generate validation data. Aborting training.")
            return

        # --- Prepare Data for Keras ---
        # Add channel dimension (for grayscale)
        X_train = np.expand_dims(X_train, axis=-1).astype('float32') / 255.0 # Normalize 0-1
        X_val = np.expand_dims(X_val, axis=-1).astype('float32') / 255.0   # Normalize 0-1

        print(f"\n[INFO] Training data shape: {X_train.shape}, Labels shape: {y_train.shape}")
        print(f"[INFO] Validation data shape: {X_val.shape}, Labels shape: {y_val.shape}")
        print(f"[INFO] Label distribution (Train): {np.bincount(y_train, minlength=self.num_classes)}")
        print(f"[INFO] Label distribution (Val):   {np.bincount(y_val, minlength=self.num_classes)}")


        # --- Build and Train Model ---
        print("\n--- Building and Training Model ---")
        self.model = self._build_cnn()
        self.model.summary()

        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
            keras.callbacks.ModelCheckpoint(str(self.model_file), monitor='val_loss', save_best_only=True, verbose=0)
            # ReduceLROnPlateau could also be useful
        ]

        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=2 # Use 1 for progress bar, 2 for one line per epoch
        )

        print(f"\n[INFO] Training complete. Best model saved to {self.model_file}")
        # Reload the best saved model explicitly
        if self.model_file.exists():
             self._load_model()
        else:
             print("[WARN] Best model checkpoint file not found after training.")


    @torch.no_grad() # Disable gradient calculations for inference
    def predict(self, cell_image, confidence_threshold=0.7):
        """
        Predicts the digit (0-9) or empty (returns 0) for a single cell image.

        Args:
            cell_image (np.ndarray): BGR image of the cell.
            confidence_threshold (float): Minimum confidence required to predict a digit (1-9).
                                          Predictions below this (or of the empty class) return 0.

        Returns:
            int: Predicted digit (1-9) or 0 for empty/uncertain.
        """
        if self.model is None:
            print("[ERROR] Model not loaded. Cannot predict.")
            # Attempt to load? Or just return 0?
            self._load_model()
            if self.model is None:
                 print("[ERROR] Failed to load model for prediction.")
                 return 0 # Return 0 if model cannot be loaded

        # 1. Preprocess the cell image
        processed_cell = self.preprocess_cell_for_model(cell_image)

        # DEBUG: Save preprocessed cell occasionally
        # if random.random() < 0.01:
        #     cv2.imwrite(f"debug_predict_cell_{time.time()}.png", processed_cell)

        # 2. Prepare for model input (batch dim, channel dim, normalize)
        model_input = np.expand_dims(processed_cell, axis=[0, -1]).astype('float32') / 255.0

        # 3. Get prediction probabilities
        # Keras handles backend automatically, but ensure tensor conversion if needed
        # For torch backend, Keras usually handles numpy input fine.
        probabilities = self.model(model_input, training=False)[0] # Use training=False

        # Convert to numpy if it's a tensor (depends on exact Keras/backend version)
        if hasattr(probabilities, 'numpy'): # Check if it's a TF tensor
             probabilities = probabilities.numpy()
        elif hasattr(probabilities, 'cpu'): # Check if it's a Torch tensor
             probabilities = probabilities.cpu().numpy()
        # If it's already numpy, this does nothing

        # 4. Determine predicted class and confidence
        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])

        # 5. Apply logic: return 0 for empty class or low confidence
        if predicted_class == _EMPTY_CLASS_LABEL:
            # print(f"  -> Predicted Empty (Class {predicted_class}), Conf: {confidence:.2f}")
            return 0
        elif confidence < confidence_threshold:
            # print(f"  -> Predicted {predicted_class}, Low Conf: {confidence:.2f} < {confidence_threshold}")
            return 0
        else:
            # print(f"  -> Predicted {predicted_class}, Conf: {confidence:.2f}")
            # Ensure prediction is within 0-9 range if not empty class
            return predicted_class if 0 <= predicted_class <= 9 else 0


# --- Example Usage ---
if __name__ == "__main__":
    print("[INFO] Initializing Digit Classifier...")
    classifier = DigitClassifier()

    # --- Option 1: Train the model ---
    # Set force_retrain=True to train even if a model file exists
    # Adjust num_samples for quicker testing
    train_model = True # Set to False to skip training if model exists
    force = False     # Set to True to force retraining

    if train_model:
        print("\n--- Starting Training Process ---")
        # Optional: Customize renderer/extractor for training data generation
        render_opts = {'warp_intensity': 0.2, 'digit_source_ratio': (0.6, 0.4)}
        extract_opts = {'cell_border_frac': 0.08}
        classifier.train(
            num_train_samples=5000, # Reduce for faster testing
            num_val_samples=1000,  # Reduce for faster testing
            epochs=10,             # Reduce for faster testing
            batch_size=128,
            force_retrain=force,
            renderer_options=render_opts,
            extractor_options=extract_opts
        )
    elif classifier.model is None:
         print("[ERROR] No model loaded and training skipped. Cannot proceed with prediction test.")
         exit()

    # --- Option 2: Test prediction on a sample image ---
    print("\n--- Testing Prediction ---")
    # Generate a test image or use an existing one
    try:
        print("[INFO] Generating a test image for prediction...")
        renderer = SudokuRenderer(warp_intensity=0.15)
        test_img, test_truth, _ = renderer.render_sudoku(difficulty=0.6)
        cv2.imwrite("temp_classifier_test.png", test_img)
        img_to_predict = "temp_classifier_test.png"
    except Exception as e:
        print(f"[WARN] Failed to generate test image ({e}). Using fallback path.")
        img_to_predict = "sample_images/digit_5_img_0.png" # Fallback

    if not Path(img_to_predict).exists():
         print(f"[ERROR] Test image {img_to_predict} not found. Cannot test prediction.")
         exit()

    print(f"[INFO] Extracting digits from {img_to_predict} for prediction test...")
    test_cells, _, _ = extract_digits(img_to_predict, cell_border_frac=0.08) # Use same options as training if specified

    if test_cells and len(test_cells) == 81:
        print("[INFO] Predicting digits for extracted cells...")
        predicted_grid = np.zeros((9, 9), dtype=int)
        correct_predictions = 0
        total_digits = 0

        for i, cell in enumerate(test_cells):
            row, col = divmod(i, 9)
            prediction = classifier.predict(cell, confidence_threshold=0.6) # Adjust threshold
            predicted_grid[row, col] = prediction

            # Compare with ground truth if available (only for generated images)
            if 'test_truth' in locals():
                 true_val = test_truth[row, col]
                 # Treat predicted 0 as matching true 0 (empty)
                 is_correct = (prediction == true_val)
                 if true_val != 0: # Only count accuracy on actual digits
                      total_digits += 1
                      if is_correct:
                           correct_predictions += 1
                 # Print mismatch details
                 # if not is_correct:
                 #      print(f"  Mismatch at ({row},{col}): True={true_val}, Pred={prediction}")


        print("\n[INFO] Predicted Grid (0 = empty/uncertain):")
        print(predicted_grid)

        if 'test_truth' in locals():
            print("\n[INFO] Ground Truth Grid:")
            print(test_truth)
            if total_digits > 0:
                 accuracy = (correct_predictions / total_digits) * 100
                 print(f"\n[INFO] Accuracy on non-empty cells: {accuracy:.2f}% ({correct_predictions}/{total_digits})")
            else:
                 print("\n[INFO] No non-empty cells in ground truth to calculate accuracy.")

    else:
        print("[ERROR] Failed to extract cells from the test image. Cannot test prediction.")

    # Clean up temp file
    if Path("temp_classifier_test.png").exists():
        os.remove("temp_classifier_test.png")

    print("\n[INFO] Classifier testing complete.")