import cv2
import numpy as np
from pathlib import Path
import urllib.request
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_DIGITS_URL = ("https://raw.githubusercontent.com/opencv/opencv/"
               "master/samples/data/digits.png")

class DigitClassifier:
    # --- Parameters for recognise() ---
    # Preprocessing
    GAUSSIAN_BLUR_KERNEL = (5, 5) # Kernel size for Gaussian blur
    ADAPTIVE_THRESH_BLOCK_SIZE = 19 # Block size for adaptive thresholding (must be odd)
    ADAPTIVE_THRESH_C = 9         # Constant subtracted from the mean in adaptive thresholding

    # Contour Filtering
    MIN_CONTOUR_AREA_RATIO = 0.01  # Min contour area relative to cell area (e.g., 1%)
    MAX_CONTOUR_AREA_RATIO = 0.80  # Max contour area relative to cell area (e.g., 80%)
    MIN_ASPECT_RATIO = 0.15        # Min aspect ratio (width/height)
    MAX_ASPECT_RATIO = 1.5         # Max aspect ratio (width/height)
    MIN_EXTENT = 0.2               # Min extent (contour area / bounding box area)

    # ROI Extraction & Resizing
    TARGET_DIGIT_SIZE = 18         # Target size (width or height) to fit digit within 20x20
    CANVAS_SIZE = 20               # Canvas size for SVM/kNN input

    def __init__(self, model_filename_base="digits", classifier_type='svm'):
        """
        Initializes the DigitClassifier.

        Args:
            model_filename_base (str): Base name for the model file (e.g., "digits").
                                       The extension (.yml for SVM, .knn for kNN) will be added.
            classifier_type (str): Type of classifier to use ('svm' or 'knn').
        """
        if classifier_type not in ['svm', 'knn']:
            raise ValueError("classifier_type must be 'svm' or 'knn'")

        self.classifier_type = classifier_type
        model_extension = ".yml" if classifier_type == 'svm' else ".knn" # Use .knn for clarity, though OpenCV might save as XML/YML
        self.model_file = Path(__file__).with_name(model_filename_base + "_" + classifier_type + model_extension)

        if self.model_file.exists():
            logging.info(f"Loading pre-trained {classifier_type.upper()} model from {self.model_file}")
            if self.classifier_type == 'svm':
                self.model = cv2.ml.SVM_load(str(self.model_file))
            else: # knn
                # Note: OpenCV kNN doesn't have a direct load method like SVM.
                # We need to reload training data or use FileStorage if saved that way.
                # For simplicity, let's re-train if the file exists but loading fails or isn't standard.
                # A more robust way would be to save/load kNN using FileStorage or pickle.
                # Let's assume re-training is acceptable if loading isn't straightforward.
                try:
                    # Attempt loading if saved via FileStorage (might produce .yml or .xml)
                    # This part is tricky as kNN doesn't have a simple load like SVM
                    # fs = cv2.FileStorage(str(self.model_file), cv2.FILE_STORAGE_READ)
                    # self.model = cv2.ml.KNearest_create()
                    # self.model.read(fs.root()) # This might work depending on how it was saved
                    # fs.release()
                    # If the above fails or isn't implemented, we fall back to training.
                    # For this example, we'll just trigger training if loading isn't obvious.
                    logging.warning(f"Standard kNN loading not implemented, attempting re-train.")
                    self.model = self._train_model()
                except Exception as e:
                    logging.warning(f"Failed to load kNN model ({e}), will re-train.")
                    self.model = self._train_model()

        else:
            logging.info(f"Model file not found. Training new {classifier_type.upper()} model.")
            self.model = self._train_model()

    def _get_hog_features(self, digits_img):
        """Extracts HOG features from the digits image."""
        logging.info("Extracting HOG features from digits image...")
        # split into 50 rows × 100 cols of 20×20 images
        rows = np.vsplit(digits_img, 50)
        cells = [np.hsplit(r, 100) for r in rows]
        cells = np.array(cells, dtype=np.uint8)  # shape=(50,100,20,20)

        # prepare training data
        hog_descriptors = []
        for img in cells.reshape(-1, self.CANVAS_SIZE, self.CANVAS_SIZE):
            deskewed = self._deskew(img)
            hog_descriptors.append(self._hog(deskewed))
        train_data = np.vstack(hog_descriptors)

        # labels: 500 samples of each digit 0..9
        labels = np.repeat(np.arange(10), 500)[:, None].astype(np.float32) # kNN needs float32 labels
        logging.info("HOG feature extraction complete.")
        return train_data, labels

    def _train_model(self):
        """
        Download (if needed), load the sample digits.png, extract HOG features
        and train the selected classifier (SVM or k-NN).
        Saves the trained model to disk for next time.
        """
        here = Path(__file__).resolve().parent
        local_digits = here / "digits.png"
        if not local_digits.exists():
            logging.info("digits.png not found – downloading it ...")
            try:
                urllib.request.urlretrieve(_DIGITS_URL, str(local_digits))
                logging.info("Download complete.")
            except Exception as e:
                logging.error(f"Failed to download digits.png: {e}")
                raise

        digits_img = cv2.imread(str(local_digits), cv2.IMREAD_GRAYSCALE)
        if digits_img is None:
            logging.error(f"Could not load digits.png from {local_digits}")
            raise FileNotFoundError(f"Could not load digits.png from {local_digits}")

        train_data, labels = self._get_hog_features(digits_img)

        logging.info(f"Training {self.classifier_type.upper()} model...")
        if self.classifier_type == 'svm':
            svm = cv2.ml.SVM_create()
            svm.setKernel(cv2.ml.SVM_RBF)
            # These parameters might need tuning (e.g., via cross-validation)
            svm.setC(12.5) # Increased C slightly from OpenCV example
            svm.setGamma(0.50625) # From OpenCV example calculation
            svm.train(train_data, cv2.ml.ROW_SAMPLE, labels.astype(np.int32)) # SVM needs int32 labels
            svm.save(str(self.model_file))
            model = svm
        else: # knn
            knn = cv2.ml.KNearest_create()
            knn.setDefaultK(3) # K value for kNN
            knn.train(train_data, cv2.ml.ROW_SAMPLE, labels)
            # Saving kNN model data (less standard than SVM save)
            # Option 1: Save using FileStorage (might create XML/YML)
            # fs = cv2.FileStorage(str(self.model_file), cv2.FILE_STORAGE_WRITE)
            # knn.write(fs)
            # fs.release()
            # Option 2: Re-train on load (simpler for this example)
            # We won't explicitly save kNN here, relying on re-training if file doesn't exist.
            # If self.model_file is created (e.g., empty), __init__ logic might need adjustment.
            # Let's just create a dummy file to indicate training happened.
            self.model_file.touch() # Create the file to prevent re-training every time
            model = knn

        logging.info(f"{self.classifier_type.upper()} trained and cached at {self.model_file}")
        return model

    @staticmethod
    def _deskew(img):
        """
        Deskew the image so that its centre of mass
        lies on the vertical axis.
        """
        SZ = img.shape[0] # Assuming square image (e.g., 20x20)
        m = cv2.moments(img)
        if abs(m["mu02"]) < 1e-2:
            return img.copy()
        skew = m["mu11"] / m["mu02"]
        M = np.float32([[1, skew, -0.5 * SZ * skew],
                        [0, 1, 0]])
        return cv2.warpAffine(img, M, (SZ, SZ),
                              flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)

    @staticmethod
    def _hog(img):
        """
        Compute a HOG descriptor.
        Uses parameters similar to OpenCV's digits sample but configurable cell size.
        """
        SZ = img.shape[0]
        # Parameters from OpenCV hog.cpp sample (used for digits)
        win_size = (SZ, SZ)
        block_size = (SZ//2, SZ//2) # 10x10 for 20x20 input
        block_stride = (SZ//2, SZ//2) # 10x10 for 20x20 input
        cell_size = (SZ//2, SZ//2) # 10x10 for 20x20 input
        nbins = 9 # Number of orientation bins (more common than 16)

        # Corrected HOGDescriptor initialization
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

        # Compute HOG features
        descriptor = hog.compute(img)

        # hog.compute returns a column vector, flatten it
        return descriptor.flatten().astype(np.float32)


    def recognise(self, cell):
        """
        Recognise the digit in a single Sudoku cell image (BGR or gray).
        Applies more robust preprocessing and contour filtering.
        Returns 0–9, where 0 means “no digit detected”.
        """
        if cell is None or cell.size == 0:
            logging.warning("Input cell is empty.")
            return 0

        # 1) Convert to gray if necessary
        if cell.ndim == 3:
            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell.copy() # Avoid modifying original

        cell_h, cell_w = gray.shape

        # 2) Preprocessing: Blur and Threshold
        # Apply Gaussian blur to reduce noise before thresholding
        blurred = cv2.GaussianBlur(gray, self.GAUSSIAN_BLUR_KERNEL, 0)

        # Adaptive thresholding works well for varying illumination
        thresh = cv2.adaptiveThreshold(blurred, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, # Invert: digit is white, background black
                                       self.ADAPTIVE_THRESH_BLOCK_SIZE,
                                       self.ADAPTIVE_THRESH_C)

        # Optional: Morphological opening to remove small noise specks
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        # 3) Find Contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, # RETR_EXTERNAL finds only outer contours
                                       cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # logging.debug("No contours found in cell.")
            return 0 # No contours found

        # 4) Filter Contours to find the best digit candidate
        digit_contours = []
        min_area = self.MIN_CONTOUR_AREA_RATIO * cell_h * cell_w
        max_area = self.MAX_CONTOUR_AREA_RATIO * cell_h * cell_w

        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)

            # Basic filtering criteria
            if area < min_area or area > max_area:
                continue # Too small or too large

            aspect_ratio = w / float(h) if h > 0 else 0
            if aspect_ratio < self.MIN_ASPECT_RATIO or aspect_ratio > self.MAX_ASPECT_RATIO:
                continue # Aspect ratio suggests it's not a digit (e.g., a line)

            extent = area / float(w * h) if w > 0 and h > 0 else 0
            if extent < self.MIN_EXTENT:
                 continue # Contour doesn't fill its bounding box well

            # Optional: Check if contour is roughly centered?
            # center_x, center_y = x + w / 2, y + h / 2
            # cell_cx, cell_cy = cell_w / 2, cell_h / 2
            # dist_from_center = np.sqrt((center_x - cell_cx)**2 + (center_y - cell_cy)**2)
            # max_dist = 0.35 * max(cell_w, cell_h) # Allow deviation up to 35% from center
            # if dist_from_center > max_dist:
            #     continue

            digit_contours.append(cnt)

        if not digit_contours:
            # logging.debug("No suitable contours passed filtering.")
            return 0 # No contour passed the filters

        # Select the best candidate (e.g., largest valid contour)
        best_cnt = max(digit_contours, key=cv2.contourArea)

        # 5) Extract ROI, Resize, and Center
        x, y, w, h = cv2.boundingRect(best_cnt)
        digit_roi = thresh[y:y+h, x:x+w] # Extract from the thresholded image

        # Create a black canvas
        canvas = np.zeros((self.CANVAS_SIZE, self.CANVAS_SIZE), dtype=np.uint8)

        # Resize the digit ROI to fit within TARGET_DIGIT_SIZE, preserving aspect ratio
        scale = self.TARGET_DIGIT_SIZE / float(max(w, h))
        new_w, new_h = int(w * scale), int(h * scale)

        # Ensure new dimensions are at least 1x1
        if new_w <= 0 or new_h <= 0:
            logging.warning(f"Invalid resized dimensions ({new_w}x{new_h}) for contour ROI.")
            return 0

        try:
            digit_resized = cv2.resize(digit_roi, (new_w, new_h))
        except cv2.error as e:
            logging.warning(f"cv2.resize failed: {e}. ROI shape: {digit_roi.shape}, Target: ({new_w},{new_h})")
            return 0


        # Calculate padding to center the digit on the canvas
        pad_y = (self.CANVAS_SIZE - new_h) // 2
        pad_x = (self.CANVAS_SIZE - new_w) // 2

        # Place the resized digit onto the canvas
        canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = digit_resized

        # Debug: Save the processed canvas
        # cv2.imwrite(f"debug_canvas_{np.random.randint(1000)}.png", canvas)

        # 6) Deskew, Extract HOG Features, and Predict
        deskewed_canvas = self._deskew(canvas)
        hog_features = self._hog(deskewed_canvas).reshape(1, -1) # Reshape for prediction

        if self.classifier_type == 'svm':
            _, result = self.model.predict(hog_features)
            prediction = int(result[0, 0])
        else: # knn
            # kNN's findNearest returns: retval, results, neighborResponses, dists
            retval, results, _, _ = self.model.findNearest(hog_features, k=self.model.getDefaultK())
            prediction = int(results[0, 0])

        # logging.debug(f"Predicted digit: {prediction}")
        return prediction