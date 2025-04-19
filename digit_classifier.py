# digit_classifier.py
"""
Convolutional‑NN based digit classifier used by the Sudoku recogniser.

The module can be executed directly to (re‑)train a model and run a
quick smoke‑test afterwards.
"""
from __future__ import annotations

import gc
import math
import os
import random
from pathlib import Path
from typing import Callable, Generator, Tuple

import cv2
import keras
import numpy as np
import torch
from keras import callbacks, layers, models

# --------------------------------------------------------------------------- #
#  Configure Keras backend – keep this on top to make sure it applies early.  #
# --------------------------------------------------------------------------- #
os.environ["KERAS_BACKEND"] = "torch"

# --------------------------------------------------------------------------- #
#  Local imports (kept late to avoid circular / backend initialisation woes). #
# --------------------------------------------------------------------------- #
from sudoku_renderer import SudokuRenderer, generate_and_save_test_example
from digit_extractor import (
    GRID_SIZE,
    extract_cells_from_image,
    rectify_grid,
    split_into_cells,
)
from sudoku_recogniser import (
    FINAL_CONFIDENCE_THRESHOLD,
    print_sudoku_grid,
)

# --------------------------------------------------------------------------- #
#  Module constants                                                           #
# --------------------------------------------------------------------------- #
MODEL_FILENAME = "sudoku_digit_classifier_cnn.keras"
MODEL_INPUT_SHAPE = (28, 28, 1)

NUM_CLASSES = 11          # 0‑9 + one “empty” class
EMPTY_LABEL = 10          # index of the empty class
TARGET_CELL_CONTENT_SIZE = 24
TARGET_DIGIT_RATIO = 1.5  # digits : empties within a batch

EPOCHS = 40
STEPS_PER_EPOCH = 150
BATCH_SIZE = 128
VALIDATION_STEPS = 50

# --------------------------------------------------------------------------- #
#  Data generator                                                             #
# --------------------------------------------------------------------------- #
DataBatch = Tuple[np.ndarray, np.ndarray]


def sudoku_data_generator(
    renderer: SudokuRenderer,
    batch_size: int,
    preprocess_func: Callable[[np.ndarray], np.ndarray | None],
    input_size: tuple[int, int, int],
    target_digit_ratio: float = TARGET_DIGIT_RATIO,
) -> Generator[DataBatch, None, None]:
    """
    Yields balanced batches of *single‑cell* images and labels generated on‑the‑fly.
    """
    grid_size_sq = GRID_SIZE * GRID_SIZE
    target_digits = int(batch_size * (target_digit_ratio / (1 + target_digit_ratio)))
    target_empty = batch_size - target_digits
    input_h, input_w = input_size[:2]

    while True:
        x_batch, y_batch = [], []
        n_digits = n_empty = 0
        attempts, max_attempts = 0, batch_size * 4

        while len(x_batch) < batch_size and attempts < max_attempts:
            attempts += 1
            allow_empty = random.random() < 0.8
            img, gt_grid, warped_corners = renderer.render_sudoku(allow_empty=allow_empty)
            if img is None or warped_corners is None:
                continue

            try:
                rectified = rectify_grid(img, warped_corners)
                if rectified is None:
                    continue

                cells, _ = split_into_cells(rectified)
                if len(cells) != grid_size_sq:
                    continue
            except Exception:
                continue

            gt_flat = gt_grid.flatten()
            cell_indices = list(range(grid_size_sq))
            random.shuffle(cell_indices)

            for idx in cell_indices:
                cell_img = cells[idx]
                label = gt_flat[idx]
                is_empty = label == 0
                model_label = EMPTY_LABEL if is_empty else label

                if is_empty and n_empty >= target_empty:
                    continue
                if not is_empty and n_digits >= target_digits:
                    continue

                processed = preprocess_func(cell_img)
                if processed is None or processed.shape != (input_h, input_w):
                    continue

                x_batch.append(processed)
                y_batch.append(model_label)
                if is_empty:
                    n_empty += 1
                else:
                    n_digits += 1

                if len(x_batch) >= batch_size:
                    break

        if not x_batch:
            continue  # try again

        x_arr = np.expand_dims(np.asarray(x_batch, dtype="float32"), -1)
        y_arr = np.asarray(y_batch, dtype="int64")
        perm = np.random.permutation(len(y_arr))

        yield x_arr[perm], y_arr[perm]
        del x_batch, y_batch, x_arr, y_arr
        gc.collect()


# --------------------------------------------------------------------------- #
#  Epoch‑end callback                                                         #
# --------------------------------------------------------------------------- #
class EpochTestCallback(callbacks.Callback):
    """Evaluate the model on a fixed Sudoku after every *frequency* epochs."""

    def __init__(
        self,
        test_img: str | Path,
        gt_grid: np.ndarray,
        classifier: "DigitClassifier",
        frequency: int = 1,
    ) -> None:
        super().__init__()
        self.frequency = max(1, frequency)
        self.gt_grid = gt_grid
        self.classifier = classifier

        cells, *_ = extract_cells_from_image(test_img, debug=False)
        if not cells or len(cells) != GRID_SIZE * GRID_SIZE:
            self.preprocessed = None
            print("[Callback] Test image preparation failed – callback disabled.")
            return

        processed = []
        for cell in cells:
            proc = classifier._preprocess_cell_for_model(cell)
            if proc is None:
                proc = np.zeros(classifier._model_input_size, dtype="float32")
            processed.append(proc)

        self.preprocessed = np.expand_dims(np.asarray(processed, dtype="float32"), -1)

    # --------------------------------------------------------------------- #
    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if self.preprocessed is None or (epoch + 1) % self.frequency:
            return

        preds = self.model.predict(self.preprocessed, verbose=0)
        idxs = np.argmax(preds, axis=1)
        confs = np.max(preds, axis=1)

        final = []
        for i, c in zip(idxs, confs):
            digit = 0
            if i != EMPTY_LABEL and c >= FINAL_CONFIDENCE_THRESHOLD:
                digit = i
            final.append(digit)

        pred_grid = np.asarray(final).reshape(GRID_SIZE, GRID_SIZE)
        conf_grid = confs.reshape(GRID_SIZE, GRID_SIZE)

        print(f"\n--- Epoch {epoch + 1} test example ---")
        print("Ground truth:")
        print_sudoku_grid(self.gt_grid, threshold=1.1)

        print("\nPrediction:")
        print_sudoku_grid(pred_grid, conf_grid, threshold=FINAL_CONFIDENCE_THRESHOLD)

        correct = np.sum(pred_grid == self.gt_grid)
        print(f"Accuracy: {correct}/{GRID_SIZE**2} ({correct / GRID_SIZE**2:.4f})")
        print("--- end ---\n")


# --------------------------------------------------------------------------- #
#  DigitClassifier                                                             #
# --------------------------------------------------------------------------- #
class DigitClassifier:
    """
    Wraps model loading, training and inference.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        training_required: bool = False,
    ) -> None:
        self.model_path = Path(model_path or MODEL_FILENAME)
        self.model: keras.Model | None = None
        self._model_input_size = MODEL_INPUT_SHAPE[:2]

        if not training_required and self.model_path.exists():
            try:
                self.model = keras.saving.load_model(self.model_path)
                loaded_shape = self.model.input_shape[1:3]
                if loaded_shape != self._model_input_size:
                    print(
                        f"[Warning] Model input shape {loaded_shape} "
                        f"!= expected {self._model_input_size}"
                    )
                print("Model loaded.")
            except Exception as exc:
                print(f"[Error] Could not load model: {exc}")

    # --------------------------------------------------------------------- #
    #  Pre‑processing                                                       #
    # --------------------------------------------------------------------- #
    def _preprocess_cell_for_model(
        self, cell: np.ndarray
    ) -> np.ndarray | None:
        """
        Binarise, crop, centre, resize and normalise a single cell image.
        """
        tgt_h, tgt_w = self._model_input_size
        if cell is None or cell.size < 10:
            return np.zeros((tgt_h, tgt_w), dtype="float32")

        gray = (
            cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            if cell.ndim == 3
            else cell.copy()
        )

        blk = max(3, min(gray.shape) // 4) | 1  # ensure odd
        try:
            thresh = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                blk,
                7,
            )
        except cv2.error:
            return np.zeros((tgt_h, tgt_w), dtype="float32")

        coords = cv2.findNonZero(thresh)
        if coords is None:
            return np.zeros((tgt_h, tgt_w), dtype="float32")

        x, y, w, h = cv2.boundingRect(coords)
        roi = thresh[y : y + h, x : x + w]
        if roi.size == 0:
            return np.zeros((tgt_h, tgt_w), dtype="float32")

        scale = min(
            TARGET_CELL_CONTENT_SIZE / max(1, w),
            TARGET_CELL_CONTENT_SIZE / max(1, h),
        )
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        roi_resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((tgt_h, tgt_w), dtype=np.uint8)
        top = (tgt_h - new_h) // 2
        left = (tgt_w - new_w) // 2
        canvas[top : top + new_h, left : left + new_w] = roi_resized

        return canvas.astype("float32") / 255.0

    # --------------------------------------------------------------------- #
    #  Model architecture                                                   #
    # --------------------------------------------------------------------- #
    def _build_cnn_model(self) -> keras.Model:
        inp = keras.Input(shape=MODEL_INPUT_SHAPE)

        aug = keras.Sequential(
            [
                layers.RandomRotation(0.08, fill_mode="constant"),
                layers.RandomTranslation(0.08, 0.08, fill_mode="constant"),
                layers.RandomZoom(0.08, 0.08, fill_mode="constant"),
            ],
            name="augmentation",
        )
        x = aug(inp)

        # block 1
        for _ in range(2):
            x = layers.Conv2D(32, (3, 3), padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)

        # block 2
        for _ in range(2):
            x = layers.Conv2D(64, (3, 3), padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Flatten()(x)
        x = layers.Dense(128)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.5)(x)

        out = layers.Dense(NUM_CLASSES, activation="softmax")(x)

        model = models.Model(inp, out)
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    # --------------------------------------------------------------------- #
    #  Training                                                             #
    # --------------------------------------------------------------------- #
    def train(
        self,
        epochs: int = EPOCHS,
        steps_per_epoch: int = STEPS_PER_EPOCH,
        batch_size: int = BATCH_SIZE,
        validation_steps: int = VALIDATION_STEPS,
    ) -> None:
        print(
            f"\nTraining parameters: {epochs=}, {steps_per_epoch=}, "
            f"{batch_size=}, {validation_steps=}"
        )

        try:
            test_img, test_gt = generate_and_save_test_example()
            cb_epoch_test = EpochTestCallback(test_img, test_gt, self)
        except Exception as exc:
            print(f"[Warn] Epoch test example unavailable: {exc}")
            cb_epoch_test = None

        train_gen = sudoku_data_generator(
            SudokuRenderer(),
            batch_size,
            self._preprocess_cell_for_model,
            MODEL_INPUT_SHAPE,
        )
        val_gen = sudoku_data_generator(
            SudokuRenderer(),
            batch_size,
            self._preprocess_cell_for_model,
            MODEL_INPUT_SHAPE,
        )

        if self.model is None:
            self.model = self._build_cnn_model()

        callbacks_list: list[callbacks.Callback] = [
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=1,
            ),
            callbacks.ModelCheckpoint(
                str(self.model_path),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.2,
                patience=3,
                min_lr=1e-6,
                verbose=1,
            ),
        ]
        if cb_epoch_test and cb_epoch_test.preprocessed is not None:
            callbacks_list.append(cb_epoch_test)

        self.model.fit(
            train_gen,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=callbacks_list,
            verbose=1,
        )

        # final evaluation
        print("\nFinal evaluation:")
        loss, acc = self.model.evaluate(
            sudoku_data_generator(
                SudokuRenderer(),
                batch_size,
                self._preprocess_cell_for_model,
                MODEL_INPUT_SHAPE,
            ),
            steps=validation_steps,
            verbose=1,
        )
        print(f"val_loss={loss:.4f}  val_acc={acc:.4f}")

        self.model.save(self.model_path)
        del train_gen, val_gen
        gc.collect()

    # --------------------------------------------------------------------- #
    #  Inference                                                            #
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def recognise(
        self,
        cell: np.ndarray,
        confidence_threshold: float = 0.7,
    ) -> Tuple[int, float]:
        """
        Return (digit, confidence). 0 means “empty / uncertain”.
        """
        if self.model is None:
            return 0, 0.0

        proc = self._preprocess_cell_for_model(cell)
        if proc is None or proc.shape != self._model_input_size:
            return 0, 0.0

        x = np.expand_dims(proc, (0, -1))
        x_tensor = torch.from_numpy(x).float()

        probs = self.model(x_tensor, training=False)[0]
        if isinstance(probs, torch.Tensor):
            probs = probs.cpu().numpy()

        idx = int(np.argmax(probs))
        conf = float(probs[idx])

        if idx == EMPTY_LABEL or conf < confidence_threshold:
            return 0, conf
        return idx, conf


# --------------------------------------------------------------------------- #
#  CLI / quick test                                                           #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    FORCE_TRAIN = False
    if FORCE_TRAIN and Path(MODEL_FILENAME).exists():
        Path(MODEL_FILENAME).unlink()

    clf = DigitClassifier(training_required=FORCE_TRAIN)

    if clf.model is None:
        clf.train()

    if clf.model:
        print("\nQuick dummy test:")
        dummy = np.zeros((50, 50), dtype=np.uint8)
        cv2.line(dummy, (25, 10), (25, 40), 255, 3)
        d, c = clf.recognise(dummy, 0.5)
        print(f"Pred (vertical line) -> {d}  conf={c:.3f}")

        empty = np.zeros((50, 50), dtype=np.uint8)
        d, c = clf.recognise(empty, 0.5)
        print(f"Pred (blank)        -> {d}  conf={c:.3f}")
