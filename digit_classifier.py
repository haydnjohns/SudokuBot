"""
Convolutional‐NN based digit classifier for Sudoku recognition.
"""

import os
import random
import gc
from pathlib import Path
from typing import Callable, Generator, Tuple, Optional

import cv2
import numpy as np
import keras
from keras import callbacks, layers, models
import torch

# use Torch as Keras backend
os.environ["KERAS_BACKEND"] = "torch"

from sudoku_renderer import SudokuRenderer, generate_and_save_test_example
from digit_extractor import GRID_SIZE, extract_cells_from_image, rectify_grid, split_into_cells
from sudoku_recogniser import FINAL_CONFIDENCE_THRESHOLD, print_sudoku_grid

MODEL_FILENAME = "sudoku_digit_classifier_cnn.keras"
MODEL_INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 11        # digits 0–9 plus one “empty” class
EMPTY_LABEL = 10
TARGET_CELL_CONTENT_SIZE = 24
TARGET_DIGIT_RATIO = 1.5

EPOCHS = 20
STEPS_PER_EPOCH = 150
BATCH_SIZE = 256
VALIDATION_STEPS = 50

DataBatch = Tuple[np.ndarray, np.ndarray]


def sudoku_data_generator(
    renderer: SudokuRenderer,
    batch_size: int,
    preprocess_func: Callable[[np.ndarray], Optional[np.ndarray]],
    input_size: Tuple[int, int, int],
    target_digit_ratio: float = TARGET_DIGIT_RATIO,
) -> Generator[DataBatch, None, None]:
    """
    Yield balanced batches of (cell_image, label) for training.
    """
    total_cells = GRID_SIZE * GRID_SIZE
    target_digits = int(batch_size * target_digit_ratio / (1 + target_digit_ratio))
    target_empty = batch_size - target_digits
    input_h, input_w = input_size[:2]

    while True:
        x_list, y_list = [], []
        n_digits = n_empty = 0
        attempts = 0
        max_attempts = batch_size * 4

        while len(x_list) < batch_size and attempts < max_attempts:
            attempts += 1
            allow_empty = (random.random() < 0.8)
            img, gt_grid, corners = renderer.render_sudoku(allow_empty=allow_empty)
            if img is None or corners is None:
                continue

            rectified = rectify_grid(img, corners)
            if rectified is None:
                continue

            cells, _ = split_into_cells(rectified)
            if len(cells) != total_cells:
                continue

            gt_flat = gt_grid.flatten()
            indices = list(range(total_cells))
            random.shuffle(indices)

            for idx in indices:
                cell = cells[idx]
                label = EMPTY_LABEL if gt_flat[idx] == 0 else gt_flat[idx]
                is_empty = (label == EMPTY_LABEL)

                if is_empty and n_empty >= target_empty:
                    continue
                if not is_empty and n_digits >= target_digits:
                    continue

                processed = preprocess_func(cell)
                if processed is None or processed.shape != (input_h, input_w):
                    continue

                x_list.append(processed)
                y_list.append(label)
                if is_empty:
                    n_empty += 1
                else:
                    n_digits += 1

                if len(x_list) >= batch_size:
                    break

        if not x_list:
            continue

        x_arr = np.array(x_list, dtype="float32")[..., np.newaxis]
        y_arr = np.array(y_list, dtype="int64")
        perm = np.random.permutation(len(y_arr))
        yield x_arr[perm], y_arr[perm]

        del x_list, y_list, x_arr, y_arr
        gc.collect()


class EpochTestCallback(callbacks.Callback):
    """
    Evaluate the model on a fixed Sudoku example after every `frequency` epochs.
    """

    def __init__(
        self,
        test_img_path: Path,
        gt_grid: np.ndarray,
        classifier: "DigitClassifier",
        frequency: int = 1,
    ) -> None:
        super().__init__()
        self.frequency = max(1, frequency)
        self.gt_grid = gt_grid
        self.classifier = classifier

        cells, _, _ = extract_cells_from_image(test_img_path, debug=False)
        if not cells or len(cells) != GRID_SIZE * GRID_SIZE:
            self.preprocessed = None
            print("[Callback] Test image preparation failed—callback disabled.")
            return

        processed = []
        for cell in cells:
            proc = classifier._preprocess_cell_for_model(cell)
            if proc is None:
                proc = np.zeros(classifier._model_input_size, dtype="float32")
            processed.append(proc)

        self.preprocessed = np.array(processed, dtype="float32")[..., np.newaxis]

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        if self.preprocessed is None or (epoch + 1) % self.frequency != 0:
            return

        preds = self.model.predict(self.preprocessed, verbose=0)
        idxs = np.argmax(preds, axis=1)
        confs = np.max(preds, axis=1)

        final = []
        for idx, conf in zip(idxs, confs):
            digit = idx if (idx != EMPTY_LABEL and conf >= FINAL_CONFIDENCE_THRESHOLD) else 0
            final.append(digit)

        pred_grid = np.array(final).reshape(GRID_SIZE, GRID_SIZE)
        conf_grid = confs.reshape(GRID_SIZE, GRID_SIZE)

        print(f"\n--- Epoch {epoch+1} Test Example ---")
        print("Ground truth:")
        print_sudoku_grid(self.gt_grid, threshold=1.1)
        print("Prediction:")
        print_sudoku_grid(pred_grid, conf_grid, threshold=FINAL_CONFIDENCE_THRESHOLD)

        correct = np.sum(pred_grid == self.gt_grid)
        total = GRID_SIZE * GRID_SIZE
        print(f"Accuracy: {correct}/{total} ({correct/total:.4f})")
        print("--- End ---\n")


class DigitClassifier:
    """
    Handles loading, training, and inference of the digit classification model.
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        training_required: bool = False,
    ) -> None:
        self.model_path = Path(model_path) if model_path else Path(MODEL_FILENAME)
        self.model: Optional[keras.Model] = None
        self._model_input_size = MODEL_INPUT_SHAPE[:2]

        if not training_required and self.model_path.exists():
            try:
                self.model = keras.models.load_model(self.model_path)
                loaded_shape = self.model.input_shape[1:3]
                if loaded_shape != self._model_input_size:
                    print(
                        f"[Warning] Model input shape {loaded_shape} "
                        f"!= expected {self._model_input_size}"
                    )
                print("Model loaded.")
            except Exception as exc:
                print(f"[Error] Could not load model: {exc}")

    def _build_cnn_model(self) -> keras.Model:
        """13‑layer SimpleNet backbone (all 3×3 convs)."""
        cfg_filters   = [32, 32,     # block‑1
                         64, 64,     # block‑2
                         96, 96, 96, # block‑3
                         128,128,128,128,   # block‑4
                         192,192]    # block‑5 (inc. optional 1×1 later)
        pool_after_id = {1, 3, 6, 10}       # indices after which we pool

        inputs = keras.Input(shape=MODEL_INPUT_SHAPE)
        x = inputs
        for i, f in enumerate(cfg_filters):
            # ordinary 3×3 conv
            x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)

            # 2×2 max‑pool **after** the i‑th conv if requested
            if i in pool_after_id:
                x = layers.MaxPooling2D(pool_size=2)(x)

        # optional 1×1 bottleneck (kept trainable but parameter‑cheap)
        x = layers.Conv2D(256, 1, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        # global spatial pooling
        x = layers.GlobalAveragePooling2D()(x)
        # add two hidden dense layers before the final classification
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(64, activation="relu")(x)
        # final output layer
        outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

        model = keras.Model(inputs, outputs, name="simplenet_digits")
        model.compile( 
            optimizer=keras.optimizers.Adam(1e-3),
            loss      ="sparse_categorical_crossentropy",
            metrics   =["accuracy"],
        )
        model.summary()         # shows ~5 M parameters → on‑par with paper
        return model

    def _preprocess_cell_for_model(self, cell: np.ndarray) -> Optional[np.ndarray]:
        """
        Binarize, crop, center, resize, and normalize a single cell image.
        Returns a 2D float32 array of shape `_model_input_size`, or None.
        """
        if cell is None or cell.size < 10:
            return None

        gray = (
            cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            if cell.ndim == 3 else cell.copy()
        )

        blk = max(3, min(gray.shape) // 4) | 1
        try:
            thresh = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                blk, 7
            )
        except cv2.error:
            return None

        coords = cv2.findNonZero(thresh)
        if coords is None:
            return None

        x, y, w, h = cv2.boundingRect(coords)
        if w == 0 or h == 0:
            return None

        roi = thresh[y : y+h, x : x+w]
        scale = min(
            TARGET_CELL_CONTENT_SIZE / w,
            TARGET_CELL_CONTENT_SIZE / h,
        )
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.zeros(self._model_input_size, dtype=np.uint8)
        top = (self._model_input_size[0] - new_h) // 2
        left = (self._model_input_size[1] - new_w) // 2
        canvas[top:top+new_h, left:left+new_w] = resized

        return canvas.astype("float32") / 255.0

    def train(
        self,
        epochs: int = EPOCHS,
        steps_per_epoch: int = STEPS_PER_EPOCH,
        batch_size: int = BATCH_SIZE,
        validation_steps: int = VALIDATION_STEPS,
    ) -> None:
        """Train the model using synthetic Sudoku data."""
        print(
            f"\nTraining parameters: "
            f"epochs={epochs}, steps_per_epoch={steps_per_epoch}, "
            f"batch_size={batch_size}, validation_steps={validation_steps}"
        )

        try:
            test_img, test_gt = generate_and_save_test_example()
            epoch_test_cb = EpochTestCallback(test_img, test_gt, self)
        except Exception as exc:
            print(f"[Warning] Could not set up epoch test callback: {exc}")
            epoch_test_cb = None

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

        cbs = [
            callbacks.EarlyStopping(
                monitor="val_loss", patience=5,
                restore_best_weights=True, verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath=self.model_path,
                monitor="val_loss", save_best_only=True, verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.2,
                patience=3, min_lr=1e-6, verbose=1
            ),
        ]
        if epoch_test_cb and epoch_test_cb.preprocessed is not None:
            cbs.append(epoch_test_cb)

        self.model.fit(
            train_gen,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=cbs,
            verbose=1,
        )

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
        print(f"Validation loss={loss:.4f}, accuracy={acc:.4f}")

        self.model.save(self.model_path)
        del train_gen, val_gen
        gc.collect()

    @torch.no_grad()
    def recognise(
        self,
        cell: np.ndarray,
        confidence_threshold: float = 0.7,
    ) -> Tuple[int, float]:
        """
        Predict the digit in a single cell.
        Returns (digit, confidence). digit=0 indicates empty/uncertain.
        """
        if self.model is None:
            return 0, 0.0

        proc = self._preprocess_cell_for_model(cell)
        if proc is None:
            return 0, 0.0

        x = proc[np.newaxis, ..., np.newaxis]
        x_tensor = torch.from_numpy(x).float()
        probs = self.model(x_tensor, training=False)[0]
        if isinstance(probs, torch.Tensor):
            probs = probs.cpu().numpy()

        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        if idx == EMPTY_LABEL or conf < confidence_threshold:
            return 0, conf
        return idx, conf


if __name__ == "__main__":
    FORCE_TRAIN = False
    if FORCE_TRAIN and Path(MODEL_FILENAME).exists():
        Path(MODEL_FILENAME).unlink()

    classifier = DigitClassifier(training_required=FORCE_TRAIN)
    if classifier.model is None:
        classifier.train()

    if classifier.model:
        print("\nQuick dummy test:")
        dummy = np.zeros((50, 50), dtype=np.uint8)
        cv2.line(dummy, (25, 10), (25, 40), 255, 3)
        d, c = classifier.recognise(dummy, confidence_threshold=0.5)
        print(f"Pred (vertical line) -> {d}, conf={c:.3f}")

        empty = np.zeros((50, 50), dtype=np.uint8)
        d, c = classifier.recognise(empty, confidence_threshold=0.5)
        print(f"Pred (blank)         -> {d}, conf={c:.3f}")
