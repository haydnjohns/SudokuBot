from __future__ import annotations
import os
import random
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

# ------------------------------------------------------------------ #
# 1.  Use TensorFlow/Keras if we can – fall back to the old SVM       #
# ------------------------------------------------------------------ #
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False
    import warnings
    warnings.warn("TensorFlow not found – falling back to HoG + SVM")

# ------------------------------------------------------------------ #
# 2.  Public class                                                    #
# ------------------------------------------------------------------ #
class DigitClassifier:
    """
    recognise(cell)  -> 0 … 9      (0 == no digit detected)

    If TensorFlow is present a small CNN is trained once on MNIST +
    strong augmentation and stored as <module>/digit_cnn.h5.
    Otherwise the legacy HoG + SVM code path is used.
    """

    # ------------------------------------------------------------------ #
    # Constructor / model bootstrap                                      #
    # ------------------------------------------------------------------ #
    def __init__(self,
                 cnn_filename: str = "digit_cnn.h5",
                 svm_filename: str = "digit_svm.yml",
                 conf_thresh: float = 0.6):

        self.conf_thresh = conf_thresh
        base_dir = Path(__file__).resolve().parent

        if _TF_AVAILABLE:
            self.cnn_path = base_dir / cnn_filename
            if self.cnn_path.exists():
                self.model = models.load_model(self.cnn_path)
            else:
                self.model = self._train_cnn()
                self.model.save(self.cnn_path)
        else:
            # ---------- fallback -------------
            self.svm_path = base_dir / svm_filename
            if self.svm_path.exists():
                self.svm = cv2.ml.SVM_load(str(self.svm_path))
            else:
                self.svm = self._train_svm()

    # ------------------------------------------------------------------ #
    # public API                                                         #
    # ------------------------------------------------------------------ #
    def recognise(self, cell: np.ndarray) -> int:
        """
        Parameters
        ----------
        cell  : ndarray
            1 Sudoku cell (BGR or gray), anything from 20×20 to 200×200 px.

        Returns
        -------
        int   : 0–9  (0 means “looks empty / unsure”)
        """

        img28, empty_like = self._preprocess(cell)

        # No meaningful foreground ⇒ definitely empty
        if empty_like:
            return 0

        if _TF_AVAILABLE:
            logits = self.model(img28[None, ...], training=False)[0].numpy()
            pred   = int(logits.argmax())
            if logits[pred] < self.conf_thresh:
                return 0
            return pred
        else:
            sample = self._hog(img28.squeeze()*255).reshape(1, -1)
            _, out = self.svm.predict(sample)
            return int(out[0][0])

    # ================================================================== #
    # 3.  CNN TRAINING  (runs once, 30‑60 s on CPU)                      #
    # ================================================================== #
    def _train_cnn(self):
        print("Training CNN on MNIST (with augmentation) – please wait …")

        (x_train, y_train), (x_val, y_val) = \
            tf.keras.datasets.mnist.load_data(path="mnist.npz")

        # normalise 0‑1
        x_train = x_train.astype("float32") / 255.
        x_val   = x_val.astype("float32")   / 255.
        x_train = x_train[..., None]          # (N,28,28,1)
        x_val   = x_val[..., None]

        def make_ds(x, y, training):
            ds = tf.data.Dataset.from_tensor_slices((x, y))
            if training:
                ds = ds.shuffle(10000).map(self._tf_augment,
                                           num_parallel_calls=4,
                                           deterministic=False)
            ds = ds.batch(256).prefetch(2)
            return ds

        train_ds = make_ds(x_train, y_train, True)
        val_ds   = make_ds(x_val,   y_val,   False)

        model = models.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, 3, activation='relu'),
            layers.MaxPool2D(),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPool2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(train_ds,
                  epochs=5,
                  validation_data=val_ds,
                  verbose=2)

        print("✓  CNN finished – accuracy on MNIST:",
              round(model.evaluate(val_ds, verbose=0)[1]*100, 2), "%")
        return model

    # -------- data augmentation (TensorFlow graph function) --------- #
    @tf.function
    def _tf_augment(self, x, y):
        # Random affine / elastic style distortions to approximate Sudoku noise
        x = tf.image.random_flip_left_right(x)          # seldom useful, but free
        x = tf.image.random_brightness(x, 0.2)
        x = tf.image.random_contrast(x, 0.8, 1.2)

        # small rotation / shear / translation
        deg = tf.random.uniform([], -20., 20.) * np.pi / 180
        tx  = tf.random.uniform([], -3., 3.)
        ty  = tf.random.uniform([], -3., 3.)
        M   = tf.stack([
            [ tf.cos(deg), -tf.sin(deg), tx],
            [ tf.sin(deg),  tf.cos(deg), ty],
            [ 0.,           0.,          1.]
        ])
        x = tfa.image.transform(x, M[:2, :].flatten(), fill_mode="nearest")

        # add synthetic line crossing out half the digit with 10 % prob
        if tf.random.uniform([]) < 0.1:
            h = tf.shape(x)[0]
            y0 = tf.random.uniform([], 0, h, dtype=tf.int32)
            y1 = y0 + tf.random.uniform([], 1, h//2, dtype=tf.int32)
            x = tf.tensor_scatter_nd_update(x,
                                            indices=tf.range(y0, y1)[:,None],
                                            updates=tf.zeros([y1-y0, 28, 1]))
        return x, y

    # ================================================================== #
    # 4.  Robust Sudoku cell PRE‑PROCESSING                              #
    # ================================================================== #
    def _preprocess(self, cell: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Converts `cell` to a (28,28,1) float32 image in [0,1] that is
        centred and sized like MNIST digits.  Also returns a flag telling
        whether the cell *looked* empty.
        """

        # ---- to gray --------------------------------------------------- #
        if cell.ndim == 3:
            cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        h, w = cell.shape

        # ---- kill the thick Sudoku grid lines by eroding & dilating ---- #
        # grid lines are the dark, long, almost‑straight segments that
        # usually touch the border of the cell.
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(4, h//8)))
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(4, w//8), 1))
        no_v = cv2.morphologyEx(cell, cv2.MORPH_OPEN, v_kernel)
        no_h = cv2.morphologyEx(cell, cv2.MORPH_OPEN, h_kernel)
        cleaned = cv2.max(cell, cv2.max(no_v, no_h))

        # ---- adaptive threshold (works in uneven lighting) ------------- #
        thresh = cv2.adaptiveThreshold(cleaned, 255,
                                       cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY_INV,
                                       11, 2)

        # ---- remove small speckles / blobs ----------------------------- #
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (2,2)),
                                  iterations=2)

        # empty ?  (almost no white pixels)
        if cv2.countNonZero(thresh) < 0.02 * h * w:
            return np.zeros((28, 28, 1), np.float32), True

        # ---- largest connected component → the digit ------------------- #
        num_lab, labels, stats, _ = cv2.connectedComponentsWithStats(thresh,
                                                                     connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA]        # skip background
        if len(areas) == 0:
            return np.zeros((28, 28, 1), np.float32), True
        biggest = 1 + int(areas.argmax())

        x, y, bw, bh, _ = stats[biggest]
        digit = thresh[y:y+bh, x:x+bw]

        # ---- keep aspect, pad to square, resize to 22×22 --------------- #
        side = max(bh, bw) + 8          # 4‑pixel margin
        square = np.zeros((side, side), dtype=np.uint8)
        y_off = (side - bh)//2
        x_off = (side - bw)//2
        square[y_off:y_off+bh, x_off:x_off+bw] = digit

        sq28 = cv2.resize(square, (28, 28), interpolation=cv2.INTER_LINEAR)

        # ---- final  (28,28,1) float32 in [0,1] ------------------------- #
        img28 = sq28.astype("float32")[..., None] / 255.
        return img28, False

    # ================================================================== #
    # 5.  Legacy HoG + SVM code (kept verbatim from the old file)        #
    # ================================================================== #
    _DIGITS_URL = ("https://raw.githubusercontent.com/opencv/opencv/"
                   "master/samples/data/digits.png")

    # -------------------- HoG + SVM TRAIN ------------------------------ #
    def _train_svm(self):
        print("Training fallback HoG+SVM – limited accuracy!")
        here = Path(__file__).resolve().parent
        local_digits = here / "digits.png"
        if not local_digits.exists():
            print("Downloading OpenCV digits.png …")
            import urllib.request
            urllib.request.urlretrieve(self._DIGITS_URL, str(local_digits))

        img = cv2.imread(str(local_digits), cv2.IMREAD_GRAYSCALE)
        rows = np.vsplit(img, 50)
        cells = np.array([np.hsplit(r, 100) for r in rows]).reshape(-1, 20, 20)

        hogs = [self._hog(self._deskew(c)) for c in cells]
        train_data = np.vstack(hogs)
        labels = np.repeat(np.arange(10), 500)[:, None]

        svm = cv2.ml.SVM_create()
        svm.setKernel(cv2.ml.SVM_RBF)
        svm.setC(2.5)
        svm.setGamma(0.05)
        svm.train(train_data, cv2.ml.ROW_SAMPLE, labels)

        svm.save(str(self.svm_path))
        return svm

    # ---------------- deskew / HoG identical to original --------------- #
    @staticmethod
    def _deskew(img: np.ndarray) -> np.ndarray:
        m = cv2.moments(img)
        if abs(m["mu02"]) < 1e-2:
            return img.copy()
        skew = m["mu11"] / m["mu02"]
        M = np.float32([[1, skew, -0.5 * 20 * skew],
                        [0, 1, 0]])
        return cv2.warpAffine(img, M, (20, 20),
                              flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)

    @staticmethod
    def _hog(img: np.ndarray) -> np.ndarray:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        bins = np.int32(16 * ang / 360)

        hist = []
        for i in range(2):
            for j in range(2):
                bin_cell = bins[i*10:(i+1)*10, j*10:(j+1)*10]
                mag_cell = mag[i*10:(i+1)*10, j*10:(j+1)*10]
                hist.append(np.bincount(bin_cell.ravel(),
                                        mag_cell.ravel(),
                                        minlength=16))
        return np.hstack(hist).astype(np.float32)