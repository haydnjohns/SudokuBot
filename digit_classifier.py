import cv2
import numpy as np
from pathlib import Path
import urllib.request

_DIGITS_URL = ("https://raw.githubusercontent.com/opencv/opencv/"
               "master/samples/data/digits.png")

class DigitClassifier:
    def __init__(self, model_filename="digits_svm.yml"):
        # Where we cache/load the trained SVM:
        self.model_file = Path(__file__).with_name(model_filename)
        if self.model_file.exists():
            self.svm = cv2.ml.SVM_load(str(self.model_file))
        else:
            self.svm = self._train_svm()

    def _train_svm(self):
        """
        Download (if needed), load the sample digits.png, extract HOG features
        and train an RBF‐SVM.  Saves the trained model to disk for next time.
        """
        here = Path(__file__).resolve().parent
        local_digits = here / "digits.png"
        if not local_digits.exists():
            print("digits.png not found – downloading it ...")
            urllib.request.urlretrieve(_DIGITS_URL, str(local_digits))

        digits_img = cv2.imread(str(local_digits), cv2.IMREAD_GRAYSCALE)
        if digits_img is None:
            raise FileNotFoundError("Could not load digits.png")

        # split into 50 rows × 100 cols of 20×20 images
        rows = np.vsplit(digits_img, 50)
        cells = [np.hsplit(r, 100) for r in rows]
        cells = np.array(cells, dtype=np.uint8)  # shape=(50,100,20,20)

        # prepare training data
        hog_descriptors = []
        for img in cells.reshape(-1, 20, 20):
            deskewed = self._deskew(img)
            hog_descriptors.append(self._hog(deskewed))
        train_data = np.vstack(hog_descriptors)

        # labels: 500 samples of each digit 0..9
        labels = np.repeat(np.arange(10), 500)[:, None]

        svm = cv2.ml.SVM_create()
        svm.setKernel(cv2.ml.SVM_RBF)
        svm.setC(2.5)
        svm.setGamma(0.05)
        svm.train(train_data, cv2.ml.ROW_SAMPLE, labels)
        svm.save(str(self.model_file))
        print(f"SVM trained and cached at {self.model_file}")
        return svm

    @staticmethod
    def _deskew(img):
        """
        Deskew the 20×20 image so that its centre of mass
        lies on the vertical axis.
        """
        m = cv2.moments(img)
        if abs(m["mu02"]) < 1e-2:
            return img.copy()
        skew = m["mu11"] / m["mu02"]
        M = np.float32([[1, skew, -0.5 * 20 * skew],
                        [0, 1, 0]])
        return cv2.warpAffine(img, M, (20, 20),
                              flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)

    @staticmethod
    def _hog(img):
        """
        Compute a 16‐bin HOG descriptor identical to OpenCV's digits sample.
        Splits the 20×20 image into four 10×10 cells.
        """
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        bins = np.int32(16 * ang / 360)  # 16 orientation bins

        hist = []
        cell_size = 10
        for i in range(2):
            for j in range(2):
                bin_cell = bins[i*cell_size:(i+1)*cell_size,
                                j*cell_size:(j+1)*cell_size]
                mag_cell = mag[i*cell_size:(i+1)*cell_size,
                               j*cell_size:(j+1)*cell_size]
                hist.append(np.bincount(bin_cell.ravel(),
                                        mag_cell.ravel(),
                                        minlength=16))
        return np.hstack(hist).astype(np.float32)

    def recognise(self, cell):
        """
        Recognise the digit in a single Sudoku cell image (BGR or gray).
        Returns 0–9, where 0 means “no digit detected”.
        """
        # 1) to gray
        if cell.ndim == 3:
            cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

        h, w = cell.shape
        margin = int(0.12 * min(h, w))
        roi = cell[margin:h-margin, margin:w-margin]

        # 2) binarise & clean
        thresh = cv2.adaptiveThreshold(roi, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV,
                                       11, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        # 3) find largest contour
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 0.02 * h * w:
            return 0

        # 4) extract digit ROI and resize to fit 20×20 (keeping aspect)
        x, y, w0, h0 = cv2.boundingRect(cnt)
        digit_roi = thresh[y:y+h0, x:x+w0]
        canvas = np.zeros((20, 20), dtype=np.uint8)
        roi_h, roi_w = digit_roi.shape
        scale = 18.0 / max(roi_h, roi_w)
        digit_resized = cv2.resize(digit_roi,
                                   (int(roi_w * scale), int(roi_h * scale)))
        dy = (20 - digit_resized.shape[0]) // 2
        dx = (20 - digit_resized.shape[1]) // 2
        canvas[dy:dy+digit_resized.shape[0],
               dx:dx+digit_resized.shape[1]] = digit_resized

        # 5) deskew, HOG, predict
        desk = self._deskew(canvas)
        sample = self._hog(desk).reshape(1, -1)
        _, result = self.svm.predict(sample)
        return int(result[0, 0])