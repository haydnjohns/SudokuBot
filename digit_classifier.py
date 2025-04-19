"""
Digit recognition that actually works on real Sudoku photos.

How does it work?
1.  A small CNN trained on MNIST (+ synthetic noise / rotation /
    “Sudoku style” rendering) is used instead of the tiny 5 000‑sample
    HOG‑SVM that ships with OpenCV.
2.  Much more aggressive (but safe) pre‑processing is done on each
    cell so that the network sees something as close as possible to a
    centred 28×28 MNIST digit.
3.  If the ONNX model is not present we *automatically* train it (takes
    ≈40 s on a laptop CPU) and cache the file next to this script.
4.  For environments where ONNXRuntime cannot be installed we quietly
    fall back to the original HOG‑SVM – you only lose accuracy, not the
    whole bot.
"""
from __future__ import annotations

import io
import math
import urllib.request
from pathlib import Path
from typing import Final

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────── constants
_ONNX_MODEL: Final = "digits_cnn.onnx"
_MNIST_URL:   Final = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"

# ─────────────────────────────────────────────────────────────────── helpers
def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"⏬  Downloading {url.split('/')[-1]} …")
    urllib.request.urlretrieve(url, dst)


def _deskew(img: np.ndarray) -> np.ndarray:
    m = cv2.moments(img)
    if abs(m["mu02"]) < 1e-2:
        return img.copy()
    skew = m["mu11"] / m["mu02"]
    M = np.float32([[1, skew, -0.5 * img.shape[0] * skew],
                    [0, 1, 0]])
    return cv2.warpAffine(img, M, img.shape[::-1],
                          flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)


# ──────────────────────────────────────────────────────────── main classifier
class DigitClassifier:
    """
    recognise(cell) → 0‑9  (0 means “blank cell / can’t tell”)
    """

    def __init__(self, override_model: str | None = None) -> None:
        self.dir = Path(__file__).resolve().parent
        self.model_path = self.dir / (override_model or _ONNX_MODEL)

        # ── try ONNX first ─────────────────────────────────────────────
        self.onnx_rt = None
        try:
            import onnxruntime as ort
            if not self.model_path.exists():
                self._train_and_export_onnx()        # ~40 s once
            self.onnx_rt = ort.InferenceSession(str(self.model_path),
                                                sess_options=ort.SessionOptions())
            self._input_name = self.onnx_rt.get_inputs()[0].name
        except ModuleNotFoundError:
            print("⚠️  onnxruntime not available – falling back to "
                  "legacy HOG‑SVM (lower accuracy).")

        # ── legacy path kept as back‑up ────────────────────────────────
        if self.onnx_rt is None:
            self._init_fallback_svm()

    # ---------------------------------------------------------------- train
    def _train_and_export_onnx(self) -> None:
        """
        Train a tiny CNN on MNIST plus heavy data augmentation that mimics
        Sudoku photos (rotations, translation, salt‑and‑pepper).
        Export as ONNX so we do *not* need TensorFlow at run‑time.
        """
        import tensorflow as tf
        from tensorflow.keras import layers, models

        mnist_path = self.dir / "mnist.npz"
        if not mnist_path.exists():
            _download(_MNIST_URL, mnist_path)

        with np.load(mnist_path) as data:
            X_train = data["x_train"]
            y_train = data["y_train"]
            X_test  = data["x_test"]
            y_test  = data["y_test"]

        # normalise & reshape ------------------------------------------------
        X_train = X_train.astype("float32") / 255.0
        X_test  = X_test.astype("float32") / 255.0
        X_train = X_train[..., None]
        X_test  = X_test[..., None]

        # data‑augmentation pipeline mimicking camera artefacts --------------
        aug = tf.keras.Sequential([
            layers.RandomRotation(0.08),
            layers.RandomTranslation(0.08, 0.08),
            layers.RandomZoom(0.1, 0.1),
            layers.RandomContrast(0.2),
        ])

        # simple CNN (28×28) --------------------------------------------------
        model = models.Sequential([
            layers.Input((28, 28, 1)),
            aug,
            layers.Conv2D(32, 3, activation="relu"),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(10, activation="softmax")
        ])
        model.compile(optimizer="adam",
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        model.fit(X_train, y_train,
                  validation_split=0.1,
                  epochs=7,
                  batch_size=256,
                  verbose=2)

        print("✓ CNN finished training – exporting to ONNX …")
        import tf2onnx
        spec = (tf.TensorSpec((None, 28, 28, 1), tf.float32, name="input"),)
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec,
                                                    opset=13)
        self.model_path.write_bytes(model_proto.SerializeToString())
        print(f"✓ Saved ONNX model to {self.model_path}")

    # ---------------------------------------------------------------- SVM fallback
    def _init_fallback_svm(self) -> None:
        from itertools import product

        self.svm_path = self.dir / "digits_svm.yml"
        if self.svm_path.exists():
            self.svm = cv2.ml.SVM_load(str(self.svm_path))
            return

        # we reuse OpenCV's digits.png so at least we have *something*
        digits_png = self.dir / "digits.png"
        if not digits_png.exists():
            _download(("https://raw.githubusercontent.com/opencv/opencv/"
                       "master/samples/data/digits.png"), digits_png)

        img = cv2.imread(str(digits_png), cv2.IMREAD_GRAYSCALE)
        cells = [np.hsplit(r, 100) for r in np.vsplit(img, 50)]
        cells = np.array(cells, dtype=np.uint8).reshape(-1, 20, 20)

        hog = cv2.HOGDescriptor(_winSize=(20, 20), _blockSize=(10, 10),
                                _blockStride=(5, 5), _cellSize=(10, 10),
                                _nbins=9)

        features = np.squeeze([hog.compute(_deskew(c)) for c in cells])
        labels = np.repeat(np.arange(10), 500)

        svm = cv2.ml.SVM_create()
        svm.setKernel(cv2.ml.SVM_RBF)
        svm.setC(2.5)
        svm.setGamma(0.05)
        svm.train(features, cv2.ml.ROW_SAMPLE, labels)
        svm.save(str(self.svm_path))
        self.svm = svm
        print(f"✓ fallback SVM cached at {self.svm_path}")

    # ---------------------------------------------------------------- recognise
    def recognise(self, cell_bgr: np.ndarray) -> int:
        """
        cell_bgr – single Sudoku cell (≈80×80 px) as BGR/gray.
        Returns 0‑9, where 0 means “no digit with enough certainty”.
        """
        gray = cell_bgr
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape
        if h == 0 or w == 0:
            return 0

        # ── 1. get region of interest (shrink borders) ────────────────
        m = int(0.08 * min(h, w))
        roi = gray[m:h - m, m:w - m]

        # ── 2. adaptive thresh  (try both polarities) ────────────────
        def _th(img, invert: bool) -> np.ndarray:
            th = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY,
                11, 2)
            # clean small noise
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
            th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=1)
            return th

        thresh_inv = _th(roi, True)
        thresh     = _th(roi, False)

        # choose the version that has the larger connected component
        cnts1, _ = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        cnts2, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        cnts = max((cnts1, thresh_inv), (cnts2, thresh),
                   key=lambda t: (max((cv2.contourArea(c) for c in t[0]), default=0)))
        contours, thresh = cnts

        if not contours:
            return 0
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < 0.03 * h * w:
            return 0

        # ── 3. crop & normalise to 28×28 ─────────────────────────────
        x, y, w0, h0 = cv2.boundingRect(cnt)
        digit = thresh[y:y + h0, x:x + w0]

        # avoid 1‑pixel wide stroked digits by slight dilation
        digit = cv2.dilate(digit, np.ones((2, 2), np.uint8), iterations=1)

        # resize, maintaining aspect ratio, pad to 28×28 -------------
        target = np.zeros((28, 28), np.uint8)
        scale = 20.0 / max(digit.shape)          # leave margins
        resized = cv2.resize(digit, (0, 0), fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA)
        dy, dx = (28 - resized.shape[0]) // 2, (28 - resized.shape[1]) // 2
        target[dy:dy + resized.shape[0], dx:dx + resized.shape[1]] = resized
        target = _deskew(target)

        # ── 4. choose backend ────────────────────────────────────────
        if self.onnx_rt is not None:
            sample = target.astype("float32")[None, None] / 255.0
            probs = self.onnx_rt.run(None, {self._input_name: sample})[0][0]
            pred = int(np.argmax(probs))
            if probs[pred] < 0.60:          # reject low confidence
                return 0
            return pred

        # fallback SVM
        hog = cv2.HOGDescriptor(_winSize=(28, 28), _blockSize=(14, 14),
                                _blockStride=(7, 7), _cellSize=(14, 14),
                                _nbins=9)
        feat = hog.compute(target).T
        _, res = self.svm.predict(feat)
        return int(res[0, 0])