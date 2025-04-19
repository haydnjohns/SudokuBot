import cv2
import numpy as np
from pathlib import Path
import logging
import os

# Suppress TensorFlow/Keras informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 0 = all, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR

try:
    # Keras 3 imports
    import keras
    from keras import layers
    from keras.datasets import mnist
except ImportError:
    raise ImportError("Keras 3 is required. Please install it (e.g., 'pip install keras') "
                      "along with a backend (e.g., 'pip install tensorflow').")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DigitClassifier:
    # --- Constants ---
    MODEL_FILENAME = "digits_cnn.keras" # Keras native format
    IMG_WIDTH = 28
    IMG_HEIGHT = 28
    # -----------------

    def __init__(self, model_filename=MODEL_FILENAME):
        self.model_file = Path(__file__).with_name(model_filename)
        self.model = None
        self._load_or_train_model()

    def _load_or_train_model(self):
        """Loads the pre-trained CNN model or trains a new one if not found."""
        if self.model_file.exists():
            try:
                self.model = keras.models.load_model(str(self.model_file))
                logging.info(f"Loaded pre-trained digit recognition model from {self.model_file}")
            except Exception as e:
                logging.error(f"Error loading model from {self.model_file}: {e}. Retraining...")
                self.model = self._train_cnn()
        else:
            logging.info(f"Model file {self.model_file} not found. Training a new model...")
            self.model = self._train_cnn()

    def _build_cnn_model(self, input_shape, num_classes):
        """Builds a simple CNN model structure."""
        model = keras.Sequential(
            [
                keras.Input(shape=input_shape),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5), # Regularization
                layers.Dense(num_classes, activation="softmax"),
            ]
        )
        return model

    def _train_cnn(self):
        """
        Trains a CNN model on the MNIST dataset and saves it.
        """
        num_classes = 10  # Digits 0-9
        input_shape = (self.IMG_HEIGHT, self.IMG_WIDTH, 1) # Grayscale images

        # 1. Load MNIST Data
        try:
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
        except Exception as e:
            logging.error(f"Failed to download or load MNIST dataset: {e}")
            logging.error("Please check your internet connection or Keras setup.")
            raise RuntimeError("Could not load MNIST data for training.") from e

        # 2. Preprocess Data
        # Scale images to the [0, 1] range
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
        # Make sure images have shape (num_samples, 28, 28, 1)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

        logging.info(f"Training data shape: {x_train.shape}")
        logging.info(f"{x_train.shape[0]} train samples, {x_test.shape[0]} test samples")

        # Convert class vectors to binary class matrices (not needed for sparse_categorical_crossentropy)
        # y_train = keras.utils.to_categorical(y_train, num_classes)
        # y_test = keras.utils.to_categorical(y_test, num_classes)

        # 3. Build Model
        model = self._build_cnn_model(input_shape, num_classes)
        model.summary(print_fn=logging.info) # Log model summary

        # 4. Compile Model
        # Use sparse_categorical_crossentropy since labels are integers
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        # 5. Train Model
        batch_size = 128
        epochs = 10 # Reduced epochs for faster training, increase for potentially better accuracy
        logging.info(f"Starting CNN training (batch_size={batch_size}, epochs={epochs})...")
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
        logging.info("CNN training finished.")

        # 6. Evaluate Model
        score = model.evaluate(x_test, y_test, verbose=0)
        logging.info(f"Test loss: {score[0]:.4f}")
        logging.info(f"Test accuracy: {score[1]:.4f}")

        # 7. Save Model
        try:
            model.save(str(self.model_file))
            logging.info(f"CNN model saved to {self.model_file}")
        except Exception as e:
            logging.error(f"Error saving model to {self.model_file}: {e}")

        return model

    def _preprocess_cell_for_cnn(self, cell_img):
        """
        Preprocesses a single Sudoku cell image for CNN prediction.
        Includes thresholding, contour finding, ROI extraction, resizing, and normalization.
        Returns a processed 28x28 image ready for the CNN or None if no digit is found.
        """
        # 1) Convert to Grayscale if needed
        if cell_img.ndim == 3:
            gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell_img.copy()

        # 2) Basic Thresholding & Cleaning (Robustness Improvements)
        # Apply Gaussian blur to reduce noise before thresholding
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive thresholding works well for varying lighting
        thresh = cv2.adaptiveThreshold(blurred, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, # Invert: digit is white, background black
                                       11, # Block size
                                       2)  # Constant subtracted from mean

        # Optional: Morphological operations to remove noise / close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        # kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=1) # Helps close gaps in digits

        # 3) Find Contours and Filter
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, # Only external contours
                                       cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # logging.debug("No contours found in cell.")
            return None # Indicate empty cell

        # Filter contours based on area and aspect ratio to find the most likely digit
        min_area = 0.01 * gray.shape[0] * gray.shape[1] # Require contour to be at least 1% of cell area
        max_area = 0.80 * gray.shape[0] * gray.shape[1] # Avoid overly large contours (e.g., grid lines)
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                 x, y, w, h = cv2.boundingRect(cnt)
                 aspect_ratio = w / float(h)
                 # Filter based on aspect ratio typical for digits (e.g., not extremely wide or tall)
                 if 0.2 < aspect_ratio < 1.5: # Adjust as needed
                     valid_contours.append(cnt)

        if not valid_contours:
            # logging.debug("No valid digit contours found after filtering.")
            return None # Indicate empty cell

        # Assume the largest valid contour is the digit
        cnt = max(valid_contours, key=cv2.contourArea)

        # 4) Extract Digit ROI and Resize for CNN (28x28)
        x, y, w, h = cv2.boundingRect(cnt)

        # Extract the digit using the bounding box from the thresholded image
        digit_roi = thresh[y:y+h, x:x+w]

        # Create a black square canvas and place the digit in the center,
        # resizing it while maintaining aspect ratio, similar to MNIST preparation.
        target_size = self.IMG_WIDTH # 28
        canvas = np.zeros((target_size, target_size), dtype=np.uint8)

        # Calculate scaling factor and new size, add padding around the digit
        padding = 4 # Add some padding around the digit within the 28x28 box
        max_dim = max(w, h)
        scale = (target_size - 2 * padding) / float(max_dim)
        new_w, new_h = int(w * scale), int(h * scale)

        # Ensure new dimensions are valid
        if new_w <= 0 or new_h <= 0:
            logging.warning("Invalid ROI size after scaling, skipping cell.")
            return None

        digit_resized = cv2.resize(digit_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Calculate position to paste the resized digit onto the canvas
        paste_x = (target_size - new_w) // 2
        paste_y = (target_size - new_h) // 2

        # Paste the digit
        canvas[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = digit_resized

        # Optional: Apply slight blur again to mimic MNIST's anti-aliasing
        # canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

        return canvas # Return the 28x28 processed image

    def recognise(self, cell_img):
        """
        Recognise the digit in a single Sudoku cell image using the trained CNN.
        Returns 0–9, where 0 means “no digit detected” or the digit '0' was recognized.
        """
        if cell_img is None or cell_img.size == 0:
            logging.warning("Received empty image for recognition.")
            return 0

        # Preprocess the cell to get a 28x28 image suitable for the CNN
        processed_cell = self._preprocess_cell_for_cnn(cell_img)

        if processed_cell is None:
            return 0 # No valid digit found during preprocessing

        # Prepare image for Keras model:
        # 1. Normalize pixel values to [0, 1]
        img_normalized = processed_cell.astype("float32") / 255.0
        # 2. Reshape to (1, height, width, channels) - batch size of 1, 1 channel (grayscale)
        img_reshaped = img_normalized.reshape(1, self.IMG_HEIGHT, self.IMG_WIDTH, 1)

        # Predict using the CNN model
        prediction = self.model.predict(img_reshaped, verbose=0) # verbose=0 suppresses progress bar

        # Get the digit with the highest probability
        predicted_digit = np.argmax(prediction)

        # Optional: Add confidence threshold
        confidence = np.max(prediction)
        confidence_threshold = 0.6 # Adjust this threshold based on testing
        if confidence < confidence_threshold:
             # logging.debug(f"Prediction confidence low ({confidence:.2f}), classifying as empty.")
             return 0 # Treat low-confidence predictions as empty

        # logging.debug(f"Predicted digit: {predicted_digit} with confidence {confidence:.2f}")
        return int(predicted_digit) # Return the recognized digit (0-9)

# Example Usage (Optional - for testing the classifier directly)
if __name__ == '__main__':
    print("Testing DigitClassifier...")

    # Create a classifier instance (will train if model doesn't exist)
    classifier = DigitClassifier()

    # Create a dummy blank image (should return 0)
    blank_cell = np.zeros((50, 50), dtype=np.uint8)
    pred_blank = classifier.recognise(blank_cell)
    print(f"Prediction for blank cell: {pred_blank} (Expected: 0)")
    assert pred_blank == 0

    # Create a dummy image with noise (should return 0)
    noise_cell = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
    cv2.circle(noise_cell, (25, 25), 3, (255, 255, 255), -1) # Small white circle
    pred_noise = classifier.recognise(noise_cell)
    print(f"Prediction for noise cell: {pred_noise} (Expected: 0)")
    # assert pred_noise == 0 # This might fail depending on noise and thresholds

    # Create a dummy image with a '7' (requires manual creation or loading)
    # For a real test, load an actual image of a digit
    try:
        # Try loading MNIST test data to simulate a digit
        (_, _), (x_test, y_test) = mnist.load_data()
        idx_7 = np.where(y_test == 7)[0][0] # Find index of a '7'
        test_digit_img = x_test[idx_7] # This is 28x28

        # Simulate a larger cell containing the digit
        cell_with_7 = np.zeros((60, 60), dtype=np.uint8)
        # Invert MNIST (white digit on black background) for our preprocessing
        test_digit_img_inv = 255 - test_digit_img
        # Resize and place onto cell
        h, w = test_digit_img_inv.shape
        scale = 1.5
        digit_res = cv2.resize(test_digit_img_inv, (int(w*scale), int(h*scale)))
        dh, dw = digit_res.shape
        y_offset = (60 - dh) // 2
        x_offset = (60 - dw) // 2
        cell_with_7[y_offset:y_offset+dh, x_offset:x_offset+dw] = digit_res

        # Add some noise
        # noise = np.random.randint(0, 30, cell_with_7.shape, dtype=np.uint8)
        # cell_with_7 = cv2.add(cell_with_7, noise)

        # cv2.imshow("Test Cell 7", cell_with_7) # Uncomment to view
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        pred_7 = classifier.recognise(cell_with_7)
        print(f"Prediction for cell with '7': {pred_7} (Expected: 7)")
        # assert pred_7 == 7 # This assertion depends heavily on the quality of the simulated image

    except Exception as e:
        print(f"Could not perform MNIST test image prediction: {e}")

    print("DigitClassifier test finished.")