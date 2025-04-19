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

_DIGITS_URL = ("https://raw.githubusercontent.com/opencv/opencv/"
               "master/samples/data/digits.png")

# name chosen so it never collides with the old OpenCV‑SVM model
_MODEL_FNAME = "digits_cnn.keras"


class DigitClassifier:
    def __init__(self, model_filename: str = _MODEL_FNAME):
        self.model_file = Path(__file__).with_name(model_filename)
        if self.model_file.exists():
            self.model = keras.saving.load_model(self.model_file)
        else:
            self.model = self._train_cnn()

    # ------------------------------------------------------------------ #
    # --------------------------  TRAINING  ---------------------------- #
    # ------------------------------------------------------------------ #
    def _train_cnn(self):
        X, y = self._load_digits_png()        # 5 000 clean 20×20 images
        X = self._prepare_images(X)           # -> (N, 28, 28, 1)  uint8
        y = y.astype("int64")

        # split train / val
        rng = np.random.default_rng(2023)
        idx = rng.permutation(len(X))
        X, y = X[idx], y[idx]
        split = int(0.9 * len(X))
        (x_train, y_train), (x_val, y_val) = (X[:split], y[:split]), (X[split:], y[split:])

        # data augmentation pipeline (run on GPU/CPU in the model graph)
        augment = keras.Sequential([
            layers.RandomRotation(0.15, fill_mode="constant"),
            layers.RandomTranslation(0.10, 0.10, fill_mode="constant"),
            layers.RandomZoom(0.10, 0.10, fill_mode="constant"),
            layers.RandomContrast(0.2),
            layers.GaussianNoise(8.0),
        ])

        def add_channel(x):         # Keras likes (h,w,1) not (h,w)
            return np.expand_dims(x, -1)

        x_train, x_val = add_channel(x_train), add_channel(x_val)

        # build a very small CNN – trains in < 1 min on CPU
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
            layers.Dense(10, activation="softmax")
        ])
        model.compile(optimizer="adam",
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])

        model.fit(x_train, y_train,
                  epochs=12, batch_size=128,
                  validation_data=(x_val, y_val),
                  verbose=2)

        # cache for future runs
        model.save(self.model_file)
        print(f"Digit CNN trained and saved to {self.model_file}")
        return model

    # -------------------------- helpers ------------------------------ #
    def _load_digits_png(self):
        """Returns (images, labels) from OpenCV’s digits.png."""
        here = Path(__file__).resolve().parent
        local_digits = here / "digits.png"
        if not local_digits.exists():
            print("digits.png not found – downloading it ...")
            urllib.request.urlretrieve(_DIGITS_URL, str(local_digits))

        img = cv2.imread(str(local_digits), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError("Could not load digits.png")

        # Split into 50×100 grid of 20×20 cells
        cells = [np.hsplit(r, 100) for r in np.vsplit(img, 50)]
        cells = np.array(cells, dtype=np.uint8).reshape(-1, 20, 20)
        labels = np.repeat(np.arange(10), 500)
        return cells, labels

    @staticmethod
    def _prepare_images(imgs):
        """
        Pad to 28×28, preserving aspect & centring the glyph.
        Also deskew each image (same routine as before).
        """
        def deskew(im):
            m = cv2.moments(im)
            if abs(m["mu02"]) < 1e-2:
                return im
            skew = m["mu11"] / m["mu02"]
            M = np.float32([[1, skew, -0.5 * 20 * skew], [0, 1, 0]])
            return cv2.warpAffine(im, M, (20, 20),
                                  flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)

        out = []
        for im in imgs:
            im = deskew(im)
            canvas = np.zeros((28, 28), np.uint8)
            scale = 22. / 20
            resized = cv2.resize(im, (0, 0), fx=scale, fy=scale,
                                 interpolation=cv2.INTER_AREA)
            dy = (28 - resized.shape[0]) // 2
            dx = (28 - resized.shape[1]) // 2
            canvas[dy:dy+resized.shape[0], dx:dx+resized.shape[1]] = resized
            out.append(canvas)
        return np.stack(out)


    # ------------------------------------------------------------------ #
    # -------------------------  INFERENCE  ---------------------------- #
    # ------------------------------------------------------------------ #
    def recognise(self, cell):
        """
        Recognise the digit in a single Sudoku cell.
        Returns 0–9, where 0 means “no confident digit detected”.
        """
        if cell.ndim == 3:
            cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        h, w = cell.shape

        # 1) Big margin trimming (gets rid of grid lines)
        margin = int(0.12 * min(h, w))
        roi = cell[margin:h - margin, margin:w - margin]
        if roi.size == 0:
            return 0

        # 2) adaptive threshold -> white glyph on black bg
        thr = cv2.adaptiveThreshold(roi, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)

        # 3) Kill very small speckles
        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
        # 4) find biggest blob
        cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return 0
        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < 0.02 * h * w:          # too small to be a digit
            return 0

        x, y, w0, h0 = cv2.boundingRect(cnt)
        digit = thr[y:y + h0, x:x + w0]

        # 5) place on 28×28 canvas
        canvas = np.zeros((28, 28), np.uint8)
        # keep aspect ratio; leave a 2‑px margin
        scale = 24.0 / max(h0, w0)
        resized = cv2.resize(digit, (int(w0 * scale), int(h0 * scale)),
                             interpolation=cv2.INTER_NEAREST)
        dy = (28 - resized.shape[0]) // 2
        dx = (28 - resized.shape[1]) // 2
        canvas[dy:dy + resized.shape[0], dx:dx + resized.shape[1]] = resized

        # 6) model prediction
        sample = canvas.astype("float32")[None, ..., None] / 255.0
        prob = self.model(sample)[0].cpu().detach().numpy()
        digit_pred = int(prob.argmax())
        confidence = float(prob[digit_pred])

        # 7) reject if low confidence
        return digit_pred if confidence >= 0.6 else 0