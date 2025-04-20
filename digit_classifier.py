"""
SudokuBot – digit classifier
Fixed version 2025‑04‑20
Async data generator version 2025‑04‑21
Serialization and Torch backend fit fix 2025-04-22
Sequence super init fix 2025-04-23

Major fixes
• data generator now really yields balanced batches
• preprocessing is tolerant – almost never rejects a cell
• data generator uses keras.utils.Sequence for asynchronous buffering
• model uses string activation for proper serialization
• removed workers/use_multiprocessing from fit/evaluate for torch backend compatibility
• Added super().__init__() to Sequence to address PyDataset warning
"""

# ------------------------------------------------------------------ #
# 1.  choose backend BEFORE importing keras
# ------------------------------------------------------------------ #
import os

os.environ["KERAS_BACKEND"] = "torch"  # must be first – do NOT move

# ------------------------------------------------------------------ #
# 2.  std‑lib & 3rd‑party imports
# ------------------------------------------------------------------ #
import gc
import random
import math
from pathlib import Path
from typing import Callable, Generator, Optional, Tuple

import cv2
import numpy as np
import torch
import keras
from keras import callbacks, layers, models, activations
from keras.utils import Sequence # Import Sequence

# ------------------------------------------------------------------ #
# 3.  project‑local imports
# ------------------------------------------------------------------ #
# Assuming these files exist in the same directory or are accessible via PYTHONPATH
try:
    from sudoku_renderer import SudokuRenderer, generate_and_save_test_example
    from digit_extractor import (
        GRID_SIZE,
        extract_cells_from_image,
        rectify_grid,
        split_into_cells,
    )
    import sudoku_recogniser
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print("Please ensure sudoku_renderer.py, digit_extractor.py, and sudoku_recogniser.py are available.")
    # Define dummy values/functions if imports fail, to allow script structure check
    GRID_SIZE = 9
    class SudokuRenderer:
        def render_sudoku(self, allow_empty=True): return None, np.zeros((9,9), dtype=int), None
    def generate_and_save_test_example(): return Path("epoch_test_sudoku.png"), np.zeros((9,9), dtype=int)
    def extract_cells_from_image(path, debug=False): return [np.zeros((30,30), dtype=np.uint8)]*(GRID_SIZE*GRID_SIZE), None, None
    def rectify_grid(img, corners): return np.zeros((200,200), dtype=np.uint8) if img is not None else None
    def split_into_cells(rect): return [np.zeros((20,20), dtype=np.uint8)]*(GRID_SIZE*GRID_SIZE), None
    class sudoku_recogniser:
        FINAL_CONFIDENCE_THRESHOLD = 0.9
        @staticmethod
        def print_sudoku_grid(grid, confs=None, threshold=0.0):
            print(grid)


# ------------------------------------------------------------------ #
# 4.  constants
# ------------------------------------------------------------------ #
MODEL_FILENAME = "sudoku_digit_classifier_cnn.keras"

MODEL_INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 11  # digits 0‑9 + “empty”
EMPTY_LABEL = 10

TARGET_CELL_CONTENT_SIZE = 24          # preprocessing
TARGET_DIGIT_RATIO = 1.5               # 60 % digits / 40 % empty

EPOCHS = 10
# Define steps explicitly for Sequence length calculation
TRAIN_STEPS_PER_EPOCH = 100
VALIDATION_STEPS = 50
BATCH_SIZE = 256


DataBatch = Tuple[np.ndarray, np.ndarray]

# ------------------------------------------------------------------ #
# 5.  balanced data generator (fixed + async via Sequence)
# ------------------------------------------------------------------ #
class SudokuDataSequence(Sequence):
    """
    Generates balanced batches of (cell, label) using keras.utils.Sequence
    for asynchronous data loading managed by Keras.
    """
    def __init__(
        self,
        renderer: SudokuRenderer,
        batch_size: int,
        steps_per_epoch: int, # Number of batches per epoch
        preprocess_func: Callable[[np.ndarray], Optional[np.ndarray]],
        input_size: Tuple[int, int, int],
        target_digit_ratio: float = TARGET_DIGIT_RATIO,
        name: str = "Generator", # For debug prints
    ):
        # --- Call super().__init__() ---
        super().__init__()
        # -----------------------------
        self.renderer = renderer
        self.batch_size = batch_size
        self.steps = steps_per_epoch
        self.preprocess_func = preprocess_func
        self.input_size = input_size
        self.target_digit_ratio = target_digit_ratio
        self.name = name

        self.total_cells = GRID_SIZE * GRID_SIZE
        self.want_digits = int(
            self.batch_size * self.target_digit_ratio / (1 + self.target_digit_ratio)
        )
        self.want_empty = self.batch_size - self.want_digits
        self.in_h, self.in_w = self.input_size[:2]

        # Debug flag check
        self.debug_print = (
            os.environ.get("SUDOKU_DEBUG_GENERATOR", "0") == "1"
        )

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return self.steps

    def __getitem__(self, index: int) -> DataBatch:
        """Generate one batch of data."""
        xs, ys = [], []
        n_dig = n_emp = 0

        while n_dig < self.want_digits or n_emp < self.want_empty:
            if not hasattr(self, 'renderer') or self.renderer is None:
                 raise RuntimeError("SudokuRenderer not initialized in Sequence.")

            img, gt_grid, corners = self.renderer.render_sudoku(allow_empty=True)
            if img is None or corners is None or gt_grid is None:
                continue
            rectified = rectify_grid(img, corners)
            if rectified is None:
                continue
            cells, _ = split_into_cells(rectified)
            if len(cells) != self.total_cells:
                continue

            idxs = list(range(self.total_cells))
            random.shuffle(idxs)

            for idx in idxs:
                if n_dig >= self.want_digits and n_emp >= self.want_empty:
                    break

                cell = cells[idx]
                flat_gt = gt_grid.flatten()
                label = EMPTY_LABEL if flat_gt[idx] == 0 else flat_gt[idx]

                proc = self.preprocess_func(cell)
                if proc is None:
                    continue

                if label == EMPTY_LABEL:
                    if n_emp < self.want_empty:
                        n_emp += 1
                        xs.append(proc)
                        ys.append(label)
                else:
                    if n_dig < self.want_digits:
                        n_dig += 1
                        xs.append(proc)
                        ys.append(label)

        x_arr = np.asarray(xs, dtype="float32")[..., np.newaxis]
        y_arr = np.asarray(ys, dtype="int64")

        p = np.random.permutation(self.batch_size)
        x_batch = x_arr[p]
        y_batch = y_arr[p]

        if self.debug_print and (index + 1) % 50 == 0:
             try: pid_info = f" worker {os.getpid()}"
             except AttributeError: pid_info = ""
             print(f"[{self.name}{pid_info}] Batch {index+1}/{self.steps} label histogram:",
                   np.bincount(y_batch, minlength=NUM_CLASSES))

        del xs, ys, x_arr, y_arr
        return x_batch, y_batch

    def on_epoch_end(self):
        """Called by Keras at the end of each epoch."""
        pass


# ------------------------------------------------------------------ #
# 6.  layer helpers (modified for serialization)
# ------------------------------------------------------------------ #
def _norm():
    """Returns a GroupNormalization layer instance."""
    return layers.GroupNormalization()

_ACTIVATION = "gelu"


# ------------------------------------------------------------------ #
# 7.  classifier object
# ------------------------------------------------------------------ #
class DigitClassifier:
    """
    Handles loading, training and inference of the CNN digit classifier.
    Uses SudokuDataSequence for efficient, asynchronous data generation.
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
            print(f"Attempting to load model from: {self.model_path}")
            try:
                self.model = keras.models.load_model(self.model_path)
                if self.model.input_shape[1:3] != self._model_input_size:
                    print(f"[Warning] Stored model input size {self.model.input_shape[1:3]} "
                          f"differs from expected {self._model_input_size}")
                print("Digit‑classifier model loaded from disk.")
            except Exception as e:
                print(f"[Error] failed to load model – will train from scratch.\nError details: {e}")
                self.model = None
        else:
            if training_required:
                print("Training required flag is set.")
            if not self.model_path.exists():
                print(f"Model file not found at {self.model_path}.")
            # Don't print "Model will be trained..." here, do it before calling train()
            # print("Model will be trained from scratch.")


    def _build_cnn_model(self) -> keras.Model:
        """Simple CNN using string activation for serialization."""
        cfg = [32, 32, 64, 64, 96, 96, 96, 128, 128, 128, 128, 192, 192]
        pool_at = {1, 3, 6, 10}

        x_in = keras.Input(shape=MODEL_INPUT_SHAPE)
        x = x_in
        for i, f in enumerate(cfg):
            x = layers.Conv2D(f, 3, padding="same", use_bias=False, name=f'conv_{i}')(x)
            x = layers.Activation(_ACTIVATION, name=f'act_{i}')(x)
            x = _norm()(x)
            if i in pool_at:
                x = layers.MaxPooling2D(2, name=f'pool_{i}')(x)

        x = layers.Conv2D(256, 1, use_bias=False, name='conv_bottleneck')(x)
        x = layers.Activation(_ACTIVATION, name='act_bottleneck')(x)
        x = _norm()(x)

        x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
        x = layers.Dense(128, activation=_ACTIVATION, name='dense_1')(x)
        x = layers.Dense(64, activation=_ACTIVATION, name='dense_2')(x)
        y_out = layers.Dense(NUM_CLASSES, activation="softmax", name='output_softmax')(x)

        model = models.Model(x_in, y_out, name="simplenet_digits_gn_gelu")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=3e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def _preprocess_cell_for_model(self, cell: np.ndarray) -> Optional[np.ndarray]:
        """
        Convert raw cell → 28×28 float32 in [0,1].
        Never raises; returns None only if `cell` itself is invalid.
        """
        if cell is None or cell.size < 10:
            return None

        if cell.ndim == 3 and cell.shape[2] == 3:
            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        elif cell.ndim == 2:
            gray = cell
        else:
             return None

        if gray.dtype != np.uint8:
             if np.issubdtype(gray.dtype, np.floating):
                 gray = np.clip(gray * 255, 0, 255).astype(np.uint8)
             else:
                 try: gray = gray.astype(np.uint8)
                 except (ValueError, TypeError): return None

        try:
            h, w = gray.shape[:2]
            blk = max(3, min(h, w) // 4)
            if blk % 2 == 0: blk += 1
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blk, 7
            )
        except cv2.error:
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        pts = cv2.findNonZero(thresh)
        if pts is None:
            return np.zeros(self._model_input_size, dtype="float32")

        x, y, w, h = cv2.boundingRect(pts)
        if w <= 0 or h <= 0:
            return np.zeros(self._model_input_size, dtype="float32")

        roi = thresh[y : y + h, x : x + w]

        scale = min(TARGET_CELL_CONTENT_SIZE / max(1, w), TARGET_CELL_CONTENT_SIZE / max(1, h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        try:
            resized = cv2.resize(roi, (new_w, new_h), interpolation=interp)
        except cv2.error:
             return np.zeros(self._model_input_size, dtype="float32")

        canvas = np.zeros(self._model_input_size, np.uint8)
        top = max(0, (self._model_input_size[0] - new_h) // 2)
        left = max(0, (self._model_input_size[1] - new_w) // 2)
        h_slice = slice(top, min(top + new_h, self._model_input_size[0]))
        w_slice = slice(left, min(left + new_w, self._model_input_size[1]))
        canvas_h = h_slice.stop - h_slice.start
        canvas_w = w_slice.stop - w_slice.start

        try:
            canvas[h_slice, w_slice] = resized[:canvas_h, :canvas_w]
        except ValueError:
            return np.zeros(self._model_input_size, dtype="float32")

        return canvas.astype("float32") / 255.0

    def train(
        self,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        train_steps: int = TRAIN_STEPS_PER_EPOCH,
        val_steps: int = VALIDATION_STEPS,
    ) -> None:
        print(f"\nStarting Training: epochs={epochs} batch={batch_size} (Parallelism handled by Keras/Backend)")
        try:
            test_img_path, test_gt = generate_and_save_test_example()
            if not Path(test_img_path).exists():
                 raise FileNotFoundError(f"Test image not found at {test_img_path}")
            epoch_cb = EpochTestCallback(test_img_path, test_gt, self)
        except Exception as e:
            print(f"[Warning] Epoch test callback disabled ({e})")
            epoch_cb = None

        train_seq = SudokuDataSequence(
            renderer=SudokuRenderer(), batch_size=batch_size, steps_per_epoch=train_steps,
            preprocess_func=self._preprocess_cell_for_model, input_size=MODEL_INPUT_SHAPE, name="TrainGen"
        )
        val_seq = SudokuDataSequence(
            renderer=SudokuRenderer(), batch_size=batch_size, steps_per_epoch=val_steps,
            preprocess_func=self._preprocess_cell_for_model, input_size=MODEL_INPUT_SHAPE, name="ValGen"
        )

        if self.model is None:
            self.model = self._build_cnn_model()
            print("Built new model:")
            self.model.summary(line_length=100)

        cbs: list[callbacks.Callback] = [
            callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
            callbacks.ModelCheckpoint(filepath=self.model_path, monitor="val_loss", save_best_only=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6, verbose=1),
        ]
        if epoch_cb and epoch_cb.preprocessed is not None:
            cbs.append(epoch_cb)

        print("Starting model.fit()...")
        history = self.model.fit(
            train_seq, epochs=epochs, validation_data=val_seq, callbacks=cbs, verbose=1,
        )

        print("\nFinal evaluation using best weights (restored by EarlyStopping):")
        eval_seq = SudokuDataSequence(
            renderer=SudokuRenderer(), batch_size=batch_size, steps_per_epoch=val_steps,
            preprocess_func=self._preprocess_cell_for_model, input_size=MODEL_INPUT_SHAPE, name="EvalGen"
        )
        loss, acc = self.model.evaluate(eval_seq, verbose=1)
        print(f"Final Validation Loss: {loss:.4f}")
        print(f"Final Validation Accuracy: {acc:.4f}")

        try:
            self.model.save(self.model_path)
            print(f"Model saved successfully to {self.model_path}")
        except Exception as e:
            print(f"[Error] Failed to save the final model: {e}")

        del train_seq, val_seq, eval_seq
        gc.collect()

    @torch.no_grad()
    def recognise(
        self,
        cell: np.ndarray,
        confidence_threshold: float = 0.7,
    ) -> Tuple[int, float]:
        """Recognise a single digit cell image."""
        if self.model is None:
            print("[Error] Recognise called but model is not loaded.")
            return 0, 0.0

        proc = self._preprocess_cell_for_model(cell)
        if proc is None:
            return 0, 0.0

        x_np = proc[np.newaxis, ..., np.newaxis]

        try:
            probs = self.model.predict(x_np, verbose=0)[0]
        except Exception as e:
             print(f"[Error] Model prediction failed: {e}")
             return 0, 0.0

        idx = int(np.argmax(probs))
        conf = float(probs[idx])

        if idx == EMPTY_LABEL or conf < confidence_threshold:
            return 0, conf
        return idx, conf


# ------------------------------------------------------------------ #
# 8.  epoch‑end callback
# ------------------------------------------------------------------ #
class EpochTestCallback(callbacks.Callback):
    """Callback to evaluate and print results on a fixed test image each epoch."""
    def __init__(
        self,
        test_img_path: Path | str,
        gt_grid: np.ndarray,
        classifier: "DigitClassifier",
        frequency: int = 1,
    ) -> None:
        super().__init__()
        self.frequency = max(1, frequency)
        self.gt_grid = gt_grid
        self.classifier = classifier
        self.test_img_path = str(test_img_path)
        self.preprocessed = None

        print(f"[Callback] Initializing with test image: {self.test_img_path}")
        try:
            cells, _, _ = extract_cells_from_image(self.test_img_path, debug=False)
            if not cells or len(cells) != GRID_SIZE * GRID_SIZE:
                print(f"[Callback] Failed to extract {GRID_SIZE*GRID_SIZE} cells from test image.")
                return

            buf = []
            for i, cell_img in enumerate(cells):
                proc = self.classifier._preprocess_cell_for_model(cell_img)
                if proc is None:
                    proc = np.zeros(self.classifier._model_input_size, dtype="float32")
                buf.append(proc)

            self.preprocessed = np.asarray(buf, dtype="float32")[..., np.newaxis]
            print("[Callback] Test example prepared successfully.")

        except FileNotFoundError:
             print(f"[Callback Error] Test image file not found at {self.test_img_path}. Callback disabled.")
        except Exception as e:
            print(f"[Callback Error] Failed preparing test example: {e}. Callback disabled.")
            self.preprocessed = None


    def on_epoch_end(self, epoch, logs=None):
        """Runs at the end of each epoch."""
        if not hasattr(self, 'model') or self.model is None or self.preprocessed is None or (epoch + 1) % self.frequency != 0:
            # Check self.model exists, Keras assigns it during training
            return

        print(f"\n--- Running Epoch {epoch+1} Test Callback ---")
        logs = logs or {}
        val_loss = logs.get('val_loss', -1)
        val_acc = logs.get('val_accuracy', -1)
        print(f"Epoch {epoch+1} - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")

        try:
            probs = self.model.predict(self.preprocessed, verbose=0)
            idxs = np.argmax(probs, axis=1)
            confs = np.max(probs, axis=1)

            final_preds = [
                i if (i != EMPTY_LABEL and c >= sudoku_recogniser.FINAL_CONFIDENCE_THRESHOLD) else 0
                for i, c in zip(idxs, confs)
            ]
            pred_grid = np.asarray(final_preds).reshape(GRID_SIZE, GRID_SIZE)
            conf_grid = confs.reshape(GRID_SIZE, GRID_SIZE)

            print("Ground Truth Grid:")
            sudoku_recogniser.print_sudoku_grid(self.gt_grid, threshold=1.1)
            print("Predicted Grid (Thresholded):")
            sudoku_recogniser.print_sudoku_grid(pred_grid, conf_grid)

            correct_cells = (pred_grid == self.gt_grid).sum()
            total_cells = self.gt_grid.size
            accuracy = correct_cells / total_cells if total_cells > 0 else 0
            print(f"Test Image Accuracy: {correct_cells}/{total_cells} = {accuracy:.4f}")
            print("--- End Callback ---\n")

        except Exception as e:
            print(f"[Callback Error] During prediction/evaluation: {e}")


# ------------------------------------------------------------------ #
# 9.  CLI helper
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    # --- Configuration ---
    FORCE_TRAIN = False # IMPORTANT: Set to False to load model if it exists
                        # Set to True to delete existing model and always retrain
    # ---------------------

    model_file = Path(MODEL_FILENAME)

    # Delete existing model only if FORCE_TRAIN is True
    if FORCE_TRAIN and model_file.exists():
        print(f"FORCE_TRAIN=True: Deleting existing model '{model_file}'")
        try:
            model_file.unlink()
            print("Model file deleted.")
        except OSError as e:
            print(f"Error deleting model file: {e}")

    # Instantiate the classifier. It will attempt to load if FORCE_TRAIN is False.
    clf = DigitClassifier(model_path=model_file, training_required=FORCE_TRAIN)

    # Train only if the model wasn't loaded successfully
    if clf.model is None:
        print("Model not loaded. Starting training...")
        try:
            clf.train(
                epochs=EPOCHS, batch_size=BATCH_SIZE,
                train_steps=TRAIN_STEPS_PER_EPOCH, val_steps=VALIDATION_STEPS,
            )
            # After successful training, the model is now in clf.model
            if clf.model is None:
                 # This shouldn't happen if train completes without error and saves
                 print("[Error] Training finished but classifier model is still None.")
                 exit(1)
            print("Training finished.")
        except Exception as train_error:
             print(f"\n--- Training Failed ---")
             print(f"An error occurred during training: {train_error}")
             exit(1)
    else:
        print("Model loaded successfully from file.")

    # Perform sanity check if model is available
    if clf.model:
        print("\n--- Running Sanity Check ---")
        try:
            dummy_line = np.zeros((50, 50), np.uint8)
            cv2.line(dummy_line, (25, 10), (25, 40), 255, 3)
            d1, c1 = clf.recognise(dummy_line, confidence_threshold=0.5)
            print(f"Vertical stroke → Predicted: {d1} (Confidence: {c1:.3f})")

            blank = np.zeros((50, 50), np.uint8)
            d0, c0 = clf.recognise(blank, confidence_threshold=0.1)
            print(f"Blank cell      → Predicted: {d0} (Confidence: {c0:.3f})")

            noisy_blank = np.random.randint(0, 30, size=(50, 50), dtype=np.uint8)
            dn, cn = clf.recognise(noisy_blank, confidence_threshold=0.1)
            print(f"Noisy blank     → Predicted: {dn} (Confidence: {cn:.3f})")

            # Optional: Test with a real digit image
            # sample_digit_path = Path("sample_digit_7.png")
            # if sample_digit_path.exists():
            #      sample_img = cv2.imread(str(sample_digit_path), cv2.IMREAD_GRAYSCALE)
            #      if sample_img is not None:
            #          ds, cs = clf.recognise(sample_img, confidence_threshold=0.5)
            #          print(f"Sample digit ({sample_digit_path.name}) → Predicted: {ds} (Confidence: {cs:.3f})")

        except Exception as sanity_error:
            print(f"\n--- Sanity Check Failed ---")
            print(f"An error occurred during sanity check: {sanity_error}")
    else:
        print("\nModel is not available. Cannot run sanity check.")

    print("\nScript finished.")