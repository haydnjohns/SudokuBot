import os
# ------------------------------------------------------------------ #
# We want Keras but *not* TensorFlow.  Tell Keras to use the torch
# backend *before* importing keras:
os.environ["KERAS_BACKEND"] = "torch"
# ------------------------------------------------------------------ #

from pathlib import Path
import urllib.request
import itertools
import cv2
import numpy as np
import keras
from keras import layers, models
import torch # Explicit import often good practice with Keras backend switching

_DIGITS_URL = ("https://raw.githubusercontent.com/opencv/opencv/"
               "master/samples/data/digits.png")
_MODEL_FNAME = "digits_cnn_11class.keras" # New model name
_EMPTY_LABEL = 10 # Label for the "empty/not-a-digit" class

class DigitClassifier:
    def __init__(self, model_filename: str = _MODEL_FNAME):
        self.model_file = Path(__file__).with_name(model_filename)
        if self.model_file.exists():
            print(f"Loading existing model: {self.model_file}")
            self.model = keras.saving.load_model(self.model_file)
        else:
            print(f"Model not found. Training new model: {self.model_file}")
            self.model = self._train_cnn()

    # ------------------------------------------------------------------ #
    # --------------------------  TRAINING  ---------------------------- #
    # ------------------------------------------------------------------ #
    def _train_cnn(self):
        X_digits, y_digits = self._load_digits_png() # 5000 clean 20×20 images (0-9)

        # !!! --- VITAL STEP: Load or Generate Empty Samples --- !!!
        # This is the most critical part requiring external data or generation
        # Let's assume we have a function `_load_empty_samples()` that returns
        # an array of empty/noise images (shape N, 20, 20)
        # X_empty = self._load_empty_samples() # Placeholder
        # For demonstration, let's generate some simple noise/blank images
        print("Generating synthetic empty samples for training demo...")
        X_empty = self._generate_synthetic_empty(num_samples=1000, size=20)
        y_empty = np.full(len(X_empty), _EMPTY_LABEL, dtype=np.int64)
        print(f"Generated {len(X_empty)} empty samples.")
        # !!! --- End of Empty Sample Handling --- !!!

        # Combine digits and empty samples
        X = np.concatenate((X_digits, X_empty), axis=0)
        y = np.concatenate((y_digits.astype("int64"), y_empty), axis=0)

        # Prepare images (deskew, pad to 28x28) - Apply to ALL samples
        X = self._prepare_images(X) # -> (N, 28, 28) uint8

        # Shuffle combined data
        rng = np.random.default_rng(2023)
        idx = rng.permutation(len(X))
        X, y = X[idx], y[idx]

        # Split train / val
        split = int(0.9 * len(X))
        (x_train, y_train), (x_val, y_val) = (X[:split], y[:split]), (X[split:], y[split:])

        # Add channel dimension for Keras
        x_train = np.expand_dims(x_train, -1)
        x_val = np.expand_dims(x_val, -1)

        # Data augmentation pipeline (remains the same)
        augment = keras.Sequential([
            layers.RandomRotation(0.15, fill_mode="constant", fill_value=0.0), # Use 0 for black bg
            layers.RandomTranslation(0.10, 0.10, fill_mode="constant", fill_value=0.0),
            layers.RandomZoom(0.10, 0.10, fill_mode="constant", fill_value=0.0),
            layers.RandomContrast(0.2),
            layers.GaussianNoise(0.1),
        ], name="augmentation")

        # Build the CNN - **Modify the final layer**
        model = models.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Rescaling(1./255),   # 0–1 float
            augment,
            layers.Conv2D(32, 3, activation="relu"),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(2),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(2),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            # --- MODIFIED LINE ---
            layers.Dense(11, activation="softmax") # 11 classes: 0-9 + empty
        ])
        model.compile(optimizer="adam",
                      loss="sparse_categorical_crossentropy", # Still correct
                      metrics=["accuracy"])

        print("Training model with 11 classes (0-9 digits + 10 empty)...")
        model.summary() # Print model structure

        # Consider adding callbacks like EarlyStopping
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ]

        model.fit(x_train, y_train,
                  epochs=25, # Might need more epochs
                  batch_size=128,
                  validation_data=(x_val, y_val),
                  callbacks=callbacks,
                  verbose=2)

        # Evaluate final model on validation set
        loss, accuracy = model.evaluate(x_val, y_val, verbose=0)
        print(f"\nValidation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")

        # Cache for future runs
        model.save(self.model_file)
        print(f"Digit CNN (11-class) trained and saved to {self.model_file}")
        return model

    # -------------------------- helpers ------------------------------ #
    def _load_digits_png(self):
        # ... (same as before) ...
        here = Path(__file__).resolve().parent
        local_digits = here / "digits.png"
        if not local_digits.exists():
            print("digits.png not found – downloading it ...")
            urllib.request.urlretrieve(_DIGITS_URL, str(local_digits))

        img = cv2.imread(str(local_digits), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError("Could not load digits.png")

        cells = [np.hsplit(r, 100) for r in np.vsplit(img, 50)]
        cells = np.array(cells, dtype=np.uint8).reshape(-1, 20, 20)
        labels = np.repeat(np.arange(10), 500)
        return cells, labels

    def _generate_synthetic_empty(self, num_samples=1000, size=20):
        """Generates simple synthetic empty/noise images."""
        samples = []
        rng = np.random.default_rng(42)

        # Type 1: Blank images
        blank_count = num_samples // 4
        samples.extend([np.zeros((size, size), dtype=np.uint8)] * blank_count)

        # Type 2: Random noise
        noise_count = num_samples // 4
        samples.extend([rng.integers(0, 50, size=(size, size), dtype=np.uint8) for _ in range(noise_count)])

        # Type 3: Simple lines (like grid lines)
        line_count = num_samples // 4
        for _ in range(line_count):
            img = np.zeros((size, size), dtype=np.uint8)
            num_lines = rng.integers(1, 3)
            for _ in range(num_lines):
                x1, y1 = rng.integers(0, size, size=2)
                x2, y2 = rng.integers(0, size, size=2)
                # --- FIX: Cast to standard Python int ---
                color = int(rng.integers(100, 255))
                thickness = int(rng.integers(1, 3))
                # --- End of FIX ---
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            samples.append(img)

        # Type 4: Gaussian blur of noise/lines
        blur_count = num_samples - len(samples) # Remaining samples
        base_indices = rng.choice(len(samples), size=blur_count)
        for i in base_indices:
             # Ensure base image is copied
            img_to_blur = samples[i].copy()
             # Apply Gaussian blur with random kernel size
            ksize = rng.choice([3, 5]) * 2 + 1 # Odd kernel size (e.g., 7, 11)
            blurred_img = cv2.GaussianBlur(img_to_blur, (ksize, ksize), 0)
            samples.append(blurred_img)

        return np.array(samples[:num_samples], dtype=np.uint8)


    @staticmethod
    def _prepare_images(imgs):
        """
        Pad to 28×28, preserving aspect & centring the glyph.
        Also deskew each image.
        Applies to both digits and empty samples during training.
        Also used during inference.
        """
        def deskew(im):
            # Only deskew if there's significant content
            if cv2.countNonZero(im) < 0.1 * im.size: # Heuristic: don't deskew near-empty
                 return im
            m = cv2.moments(im)
            if abs(m["mu02"]) < 1e-2:
                return im
            skew = m["mu11"] / m["mu02"]
            # Use the target size (20x20) for deskewing calculation
            M = np.float32([[1, skew, -0.5 * 20 * skew], [0, 1, 0]])
            # Ensure output size is 20x20
            return cv2.warpAffine(im, M, (20, 20),
                                  flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)

        out = []
        for im in imgs:
            # Deskewing might not be ideal for all 'empty' types, but let's try
            # It should handle grid lines okay. Noise might become weirder.
            # Original images are 20x20 before this step
            h_orig, w_orig = im.shape
            if h_orig != 20 or w_orig != 20:
                 # If input isn't 20x20 (e.g., from recognise), resize first
                 im = cv2.resize(im, (20, 20), interpolation=cv2.INTER_AREA)

            im_deskewed = deskew(im)

            # Pad to 28x28, centering the 20x20 content
            canvas = np.zeros((28, 28), np.uint8)
            # Calculate padding amounts
            pad_y = (28 - 20) // 2
            pad_x = (28 - 20) // 2
            canvas[pad_y:pad_y + 20, pad_x:pad_x + 20] = im_deskewed
            out.append(canvas)

        return np.stack(out)


    # ------------------------------------------------------------------ #
    # -------------------------  INFERENCE  ---------------------------- #
    # ------------------------------------------------------------------ #
    @torch.no_grad() # Disable gradient calculation for inference
    def recognise(self, cell):
        """
        Recognise the digit in a single Sudoku cell using the 11-class model.
        Returns 0–9 for a detected digit, or 0 if the model predicts the
        'empty' class or has very low confidence.
        """
        if cell is None or cell.size == 0:
            return 0
        if cell.ndim == 3:
            cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

        # 1. Basic Preprocessing: Adaptive Threshold
        # Keep thresholding to get a binary image, helps normalize contrast
        # Use slightly larger block size, maybe less sensitive to noise
        thr = cv2.adaptiveThreshold(cell, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 19, 5) # Larger block, adjust C if needed

        # Optional: Minimal noise removal (can be helpful)
        # Keep this minimal, as the model should handle some noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)

        # --- REMOVED: Contour finding, area filtering, bounding box extraction ---

        # 2. Prepare for Model: Resize/Pad the *entire thresholded cell*
        # Use the same logic as _prepare_images (deskew + pad)
        # We need to resize the potentially large cell `thr` down to 20x20 first,
        # then let _prepare_images handle deskewing and padding to 28x28.
        # Note: _prepare_images expects a list/array of images.
        prepared_img_array = self._prepare_images(np.array([thr])) # Pass as array
        canvas = prepared_img_array[0] # Get the single processed image

        # DEBUG: Optionally save the canvas fed to the model
        # cv2.imwrite(f"debug_canvas_{np.random.randint(1000)}.png", canvas)

        # 3. Model Prediction
        # Ensure input is float32, add batch and channel dims, normalize
        sample = canvas.astype("float32")[None, ..., None] / 255.0

        # Use torch tensor for torch backend
        sample_tensor = torch.from_numpy(sample).float()
        # If using GPU, move tensor to the correct device
        # device = next(self.model.parameters()).device # Get model's device
        # sample_tensor = sample_tensor.to(device)

        prob = self.model(sample_tensor, training=False)[0] # Use training=False

        # Convert probabilities back to numpy if needed (e.g., for argmax)
        # If using GPU, move back to CPU first
        prob_np = prob.cpu().numpy() # Use .cpu() if tensor was moved to GPU

        pred_class = int(prob_np.argmax())
        confidence = float(prob_np[pred_class])

        # 4. Decision Logic
        # Check if the predicted class is the 'empty' label
        if pred_class == _EMPTY_LABEL:
            # print(f"Predicted Empty (Class {_EMPTY_LABEL}), Conf: {confidence:.2f}")
            return 0 # Return 0 for empty

        # Optional: Add a confidence threshold even for digit predictions
        min_confidence = 0.7 # Increase threshold slightly? Tunable.
        if confidence < min_confidence:
            # print(f"Predicted {pred_class}, Low Conf: {confidence:.2f} < {min_confidence}")
            return 0 # Reject low-confidence predictions

        # Otherwise, return the predicted digit (0-9)
        # print(f"Predicted {pred_class}, Conf: {confidence:.2f}")
        return pred_class # pred_class is already 0-9 here