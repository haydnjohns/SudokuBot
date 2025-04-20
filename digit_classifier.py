"""
Convolution-NN based digit classifier for Sudoku recognition.
Fixed version:
  – backend is selected *before* importing Keras
  – Batch Normalisation → Layer Normalisation
    (gets rid of the big train/val accuracy gap)
  – Corrected label mapping: 0=empty, 1-9=digits (NUM_CLASSES=10)
"""

# ------------------------------------------------------------------ #
# 1.  Backend must be selected BEFORE importing keras
# ------------------------------------------------------------------ #
import os
os.environ["KERAS_BACKEND"] = "torch"          # HAS TO BE FIRST

# ------------------------------------------------------------------ #
# 2.  Standard imports
# ------------------------------------------------------------------ #
import random
import gc
from pathlib import Path
from typing import Callable, Generator, Tuple, Optional

import cv2
import numpy as np
import torch
import keras
from keras import callbacks, layers, models

# ------------------------------------------------------------------ #
# 3.  Project-local imports
# ------------------------------------------------------------------ #
# Assume these imports exist and are correct
try:
    from sudoku_renderer import SudokuRenderer, generate_and_save_test_example
    from digit_extractor import GRID_SIZE, extract_cells_from_image, rectify_grid, split_into_cells
    import sudoku_recogniser
except ImportError as e:
    print(f"[Warning] Could not import project files: {e}")
    # Define dummy values/classes if imports fail, to allow script execution
    GRID_SIZE = 9
    class SudokuRenderer:
        def render_sudoku(self, allow_empty=True): return None, None, None
    def generate_and_save_test_example(): return Path("dummy_test.png"), np.zeros((9,9), dtype=int)
    def extract_cells_from_image(path, debug=False): return [], None, None
    def rectify_grid(img, corners): return None
    def split_into_cells(rectified): return [], None
    class sudoku_recogniser:
        FINAL_CONFIDENCE_THRESHOLD = 0.7
        @staticmethod
        def print_sudoku_grid(grid, conf=None, threshold=0.0): print(grid)


# ------------------------------------------------------------------ #
# 4.  Constants
# ------------------------------------------------------------------ #
MODEL_FILENAME = "sudoku_digit_classifier_cnn.keras"
MODEL_INPUT_SHAPE = (28, 28, 1)
# --- FIXED ---
NUM_CLASSES = 10          # digits 1-9 (indices 1-9) + "empty" (index 0)
EMPTY_LABEL = 0           # Use index 0 for the "empty" class
# -------------
TARGET_CELL_CONTENT_SIZE = 24
TARGET_DIGIT_RATIO = 1.5 # Ratio of digit examples to empty examples in a batch

EPOCHS = 10
STEPS_PER_EPOCH = 100
BATCH_SIZE = 256
VALIDATION_STEPS = 50

DataBatch = Tuple[np.ndarray, np.ndarray]


# ------------------------------------------------------------------ #
# 5.  Data generator (logic unchanged, but uses new EMPTY_LABEL)
# ------------------------------------------------------------------ #
def sudoku_data_generator(
    renderer: SudokuRenderer,
    batch_size: int,
    preprocess_func: Callable[[np.ndarray], Optional[np.ndarray]],
    input_size: Tuple[int, int, int],
    target_digit_ratio: float = TARGET_DIGIT_RATIO,
) -> Generator[DataBatch, None, None]:
    """
    Yield balanced batches of (cell_image, label) for training / validation.
    Labels: 0 for empty, 1-9 for digits.
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
            if img is None or corners is None or gt_grid is None:
                continue

            rectified = rectify_grid(img, corners)
            if rectified is None:
                continue

            cells, _ = split_into_cells(rectified)
            if len(cells) != total_cells:
                continue

            gt_flat = gt_grid.flatten()
            idxs = list(range(total_cells))
            random.shuffle(idxs)

            for idx in idxs:
                cell = cells[idx]
                # --- Uses new EMPTY_LABEL=0 ---
                # If gt_flat[idx] is 0 (empty in Sudoku), label becomes 0.
                # If gt_flat[idx] is 1-9, label becomes 1-9.
                label = EMPTY_LABEL if gt_flat[idx] == 0 else gt_flat[idx]
                # -----------------------------
                is_empty = (label == EMPTY_LABEL) # Checks if label is 0

                if is_empty and n_empty >= target_empty:
                    continue
                if not is_empty and n_digits >= target_digits:
                    continue

                processed = preprocess_func(cell)
                if processed is None or processed.shape != (input_h, input_w):
                    # Treat failed preprocessing as potentially empty for balancing
                    if n_empty < target_empty:
                         # Add a blank image with empty label if needed
                         x_list.append(np.zeros((input_h, input_w), dtype="float32"))
                         y_list.append(EMPTY_LABEL)
                         n_empty += 1
                    continue # Skip the original cell if preprocessing failed

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

        x_arr = np.asarray(x_list, dtype="float32")[..., np.newaxis]
        # Labels are now correctly in the range [0, 9]
        y_arr = np.asarray(y_list, dtype="int64")
        p = np.random.permutation(len(y_arr))
        yield x_arr[p], y_arr[p]
        del x_list, y_list, x_arr, y_arr
        gc.collect()


# ------------------------------------------------------------------ #
# 6.  Fixed CNN backbone (LayerNorm + Correct Output Layer Size)
# ------------------------------------------------------------------ #
def _norm():                   # small helper → keeps the code tidy
    return layers.LayerNormalization(epsilon=1e-5)


class DigitClassifier:
    """
    Handles loading, training and inference of the digit-classification model.
    """

    # -------------------------------------------------------------- #
    # constructor
    # -------------------------------------------------------------- #
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
                # Ensure custom objects are known if needed (LayerNorm is standard)
                self.model = keras.models.load_model(self.model_path)
                if self.model.input_shape[1:3] != self._model_input_size:
                    print("[Warning] Stored model input size differs from expected.")
                if self.model.output_shape[-1] != NUM_CLASSES:
                     print(f"[Error] Stored model output size ({self.model.output_shape[-1]}) "
                           f"differs from expected ({NUM_CLASSES}). Re-training required.")
                     self.model = None # Force re-training
                else:
                    print("Model loaded from disk.")
            except Exception as e:
                print(f"[Error] Failed to load model – will re-train ({e})")
                self.model = None # Ensure model is None if loading fails

        # If model is still None (either not found, load failed, or size mismatch),
        # and training is required, it will be built in the train() method.
        if self.model is None and not training_required:
             print("[Info] Model not found or invalid, and training_required=False. "
                   "Classifier will not function until trained or a valid model is provided.")


    # -------------------------------------------------------------- #
    # backbone
    # -------------------------------------------------------------- #
    def _build_cnn_model(self) -> keras.Model:
        """Simple-Net backbone (LayerNorm) with corrected output size."""
        cfg_filters   = [32, 32,        # block-1
                         64, 64,        # block-2
                         96, 96, 96,    # block-3
                         128,128,128,128,  # block-4
                         192,192]       # block-5
        pool_after = {1, 3, 6, 10}

        x_in = keras.Input(shape=MODEL_INPUT_SHAPE)
        x = x_in
        for i, f in enumerate(cfg_filters):
            x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
            x = _norm()(x)              # LayerNorm – no train/infer mismatch
            x = layers.ReLU()(x)
            if i in pool_after:
                x = layers.MaxPooling2D(2)(x)

        # 1×1 bottleneck
        x = layers.Conv2D(256, 1, use_bias=False)(x)
        x = _norm()(x)
        x = layers.ReLU()(x)

        # classifier head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(64,  activation="relu")(x)
        # --- FIXED ---
        # Output layer now has NUM_CLASSES (10) units for classes 0-9
        y_out = layers.Dense(NUM_CLASSES, activation="softmax")(x)
        # -------------

        model = keras.Model(x_in, y_out, name="simplenet_digits_ln_fixed")
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy", # Expects labels 0 to N-1
            metrics=["accuracy"],
        )
        model.summary()
        return model

    # -------------------------------------------------------------- #
    # cell preprocessing (unchanged)
    # -------------------------------------------------------------- #
    def _preprocess_cell_for_model(self, cell: np.ndarray) -> Optional[np.ndarray]:
        if cell is None or cell.size < 10:
            return None

        # Ensure input is grayscale
        if cell.ndim == 3 and cell.shape[2] == 3:
            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        elif cell.ndim == 2:
            gray = cell.copy()
        else: # Unexpected shape
             return None

        # Adaptive thresholding
        blk = max(3, min(gray.shape)//4) | 1 # Ensure odd block size >= 3
        try:
            thresh = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                blk, 7
            )
        except cv2.error:
            # Handle cases where block size might be too large for small images
            # Fallback to simple Otsu thresholding? Or just return None?
            try:
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            except cv2.error:
                 return None # Give up if thresholding fails

        # Find bounding box of non-zero pixels
        pts = cv2.findNonZero(thresh)
        if pts is None:
            # If the cell is entirely black after inversion (i.e., originally white/blank)
            # Return an empty canvas, which should be classified as EMPTY_LABEL
            return np.zeros(self._model_input_size, dtype="float32")

        x, y, w, h = cv2.boundingRect(pts)
        if w <= 0 or h <= 0:
             # Should not happen if pts is not None, but check anyway
             return np.zeros(self._model_input_size, dtype="float32")

        # Extract ROI and resize
        roi = thresh[y:y+h, x:x+w]
        scale = min(
            TARGET_CELL_CONTENT_SIZE / w,
            TARGET_CELL_CONTENT_SIZE / h
        )
        new_w = max(1, int(w*scale))
        new_h = max(1, int(h*scale))
        try:
            resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except cv2.error:
            return None # Resize failed

        # Place resized ROI onto center of canvas
        canvas = np.zeros(self._model_input_size, np.uint8)
        top  = max(0, (self._model_input_size[0]-new_h)//2)
        left = max(0, (self._model_input_size[1]-new_w)//2)
        # Ensure slicing does not go out of bounds
        h_slice = slice(top, min(top + new_h, self._model_input_size[0]))
        w_slice = slice(left, min(left + new_w, self._model_input_size[1]))
        canvas_h = h_slice.stop - h_slice.start
        canvas_w = w_slice.stop - w_slice.start

        # Ensure resized image fits into the calculated slice
        if canvas_h > 0 and canvas_w > 0:
             canvas[h_slice, w_slice] = resized[:canvas_h, :canvas_w]

        # Normalize to [0, 1]
        return canvas.astype("float32")/255.0

    # -------------------------------------------------------------- #
    # training routine (uses corrected model builder)
    # -------------------------------------------------------------- #
    def train(
        self,
        epochs: int = EPOCHS,
        steps_per_epoch: int = STEPS_PER_EPOCH,
        batch_size: int = BATCH_SIZE,
        validation_steps: int = VALIDATION_STEPS,
    ) -> None:
        print(f"\nTraining: epochs={epochs}  batch={batch_size}")
        try:
            test_img_path, test_gt = generate_and_save_test_example()
            epoch_cb = EpochTestCallback(test_img_path, test_gt, self)
            if epoch_cb.preprocessed is None:
                 print("[Warning] EpochTestCallback disabled due to preprocessing issues.")
                 epoch_cb = None # Disable if setup failed
        except Exception as e:
            print(f"[Warning] Epoch-callback disabled ({e})")
            epoch_cb = None

        # Ensure SudokuRenderer is available
        try:
            renderer = SudokuRenderer()
        except NameError:
            print("[Error] SudokuRenderer not available. Cannot generate training data.")
            return
        except Exception as e:
            print(f"[Error] Failed to initialize SudokuRenderer: {e}")
            return

        train_gen = sudoku_data_generator(
            renderer, batch_size,
            self._preprocess_cell_for_model, MODEL_INPUT_SHAPE
        )
        val_gen = sudoku_data_generator(
            renderer, batch_size,
            self._preprocess_cell_for_model, MODEL_INPUT_SHAPE
        )

        if self.model is None:
            print("Building new model...")
            self.model = self._build_cnn_model()
        elif self.model.output_shape[-1] != NUM_CLASSES:
             print(f"Model output size mismatch ({self.model.output_shape[-1]} vs {NUM_CLASSES}). Rebuilding...")
             self.model = self._build_cnn_model()


        # Check if model exists before proceeding
        if self.model is None:
             print("[Error] Model could not be built. Aborting training.")
             return

        cbs: list[callbacks.Callback] = [
            callbacks.EarlyStopping(
                monitor="val_accuracy", # Monitor val_accuracy for improvement
                patience=7,             # Increase patience slightly
                restore_best_weights=True,
                mode='max',             # We want to maximize accuracy
                verbose=1),
            callbacks.ModelCheckpoint(
                filepath=self.model_path,
                monitor="val_accuracy", # Save based on best val_accuracy
                save_best_only=True,
                mode='max',             # We want to maximize accuracy
                verbose=1),
            callbacks.ReduceLROnPlateau(
                monitor="val_accuracy", # Reduce LR if val_accuracy plateaus
                factor=0.3,             # More aggressive reduction
                patience=3,
                min_lr=1e-7,            # Allow lower min_lr
                mode='max',             # We want to maximize accuracy
                verbose=1),
        ]
        if epoch_cb: # Add callback only if it was successfully initialized
            cbs.append(epoch_cb)

        print("Starting model training...")
        try:
            self.model.fit(
                train_gen,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=val_gen,
                validation_steps=validation_steps,
                callbacks=cbs,
                verbose=1,
            )
        except Exception as e:
             print(f"\n[Error] Training failed: {e}")
             # Optionally: clean up generators
             del train_gen, val_gen
             gc.collect()
             return # Stop execution here

        print("\nTraining finished. Final evaluation on validation set:")
        # Reload the best model saved by ModelCheckpoint
        if self.model_path.exists():
             print(f"Loading best model from {self.model_path} for final evaluation.")
             try:
                 self.model = keras.models.load_model(self.model_path)
             except Exception as e:
                  print(f"[Warning] Could not reload best model, evaluating current model state: {e}")

        if self.model: # Ensure model exists before evaluation
            loss, acc = self.model.evaluate(
                sudoku_data_generator( # Use a fresh generator instance
                    SudokuRenderer(), batch_size,
                    self._preprocess_cell_for_model, MODEL_INPUT_SHAPE
                ),
                steps=validation_steps,
                verbose=1,
            )
            print(f"Final val_loss={loss:.4f}  val_acc={acc:.4f}")

            # Save the final best model explicitly (redundant if ModelCheckpoint worked, but safe)
            print(f"Saving final model to {self.model_path}")
            self.model.save(self.model_path)
        else:
             print("[Error] No model available for final evaluation or saving.")

        del train_gen, val_gen
        gc.collect()

    # -------------------------------------------------------------- #
    # inference (logic unchanged, but uses new EMPTY_LABEL=0)
    # -------------------------------------------------------------- #
    @torch.no_grad()
    def recognise(
        self,
        cell: np.ndarray,
        confidence_threshold: float = 0.7,
    ) -> Tuple[int, float]:
        """
        Recognises a digit in a cell image.
        Returns: (predicted_digit, confidence).
                 predicted_digit is 0 for empty/uncertain, 1-9 for digits.
        """
        if self.model is None:
            print("[Warning] Recognise called but model is not loaded.")
            return 0, 0.0

        proc = self._preprocess_cell_for_model(cell)
        if proc is None:
            # Preprocessing failed, treat as empty/uncertain
            return 0, 0.0

        # Add batch and channel dimensions
        x = torch.from_numpy(proc[np.newaxis, ..., np.newaxis]).float()

        # Perform inference
        try:
            # Use Keras model's predict method for consistency, even with torch backend
            # Or call directly if using torch backend exclusively
            # probs = self.model.predict(x.cpu().numpy(), verbose=0)[0] # Keras predict
            probs = self.model(x, training=False)[0] # Direct call (might return tensor)
            if isinstance(probs, torch.Tensor):
                probs = probs.cpu().numpy() # Convert to numpy if needed
        except Exception as e:
             print(f"[Error] Model inference failed: {e}")
             return 0, 0.0

        # --- Uses new EMPTY_LABEL=0 ---
        idx = int(np.argmax(probs)) # Index 0 is empty, 1-9 are digits
        conf = float(probs[idx])

        # If predicted class is EMPTY_LABEL (0) or confidence is too low,
        # return 0 (Sudoku convention for empty).
        if idx == EMPTY_LABEL or conf < confidence_threshold:
            # Return 0, but retain the confidence of the top prediction (even if it was empty)
            return 0, conf
        # Otherwise, return the predicted digit (1-9) and its confidence.
        return idx, conf
        # -----------------------------


# ------------------------------------------------------------------ #
# 7.  Epoch-end sanity-check callback (logic unchanged, uses new EMPTY_LABEL=0)
# ------------------------------------------------------------------ #
class EpochTestCallback(callbacks.Callback):
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
        self.preprocessed = None # Initialize as None

        try:
            cells, _, _ = extract_cells_from_image(test_img_path, debug=False)
            if not cells or len(cells) != GRID_SIZE*GRID_SIZE:
                print("[Callback] Could not extract correct number of cells from test image.")
                return # Leave self.preprocessed as None

            buf = []
            valid_cells = 0
            for i, cell in enumerate(cells):
                proc = classifier._preprocess_cell_for_model(cell)
                if proc is None:
                    # If preprocessing fails, use a blank image
                    print(f"[Callback Warning] Preprocessing failed for cell {i}, using blank.")
                    proc = np.zeros(classifier._model_input_size, dtype="float32")
                else:
                     valid_cells += 1
                buf.append(proc)

            if valid_cells == 0:
                 print("[Callback Error] Preprocessing failed for all test cells.")
                 return # Leave self.preprocessed as None

            self.preprocessed = np.asarray(buf, dtype="float32")[..., np.newaxis]
            print(f"[Callback] Test example prepared successfully ({valid_cells}/{GRID_SIZE*GRID_SIZE} cells valid).")

        except NameError:
             print("[Callback Error] `extract_cells_from_image` not available.")
        except Exception as e:
            print(f"[Callback Error] Failed to prepare test example: {e}")


    def on_epoch_end(self, epoch, logs=None):
        # Check if callback is enabled and frequency matches
        if self.preprocessed is None or (epoch+1) % self.frequency != 0:
            return
        # Check if the model exists on the parent class (Keras requirement)
        if not hasattr(self, 'model') or self.model is None:
             print("[Callback Warning] Model not available in callback.")
             return

        print(f"\n--- Epoch {epoch+1} test example ---")
        try:
            probs = self.model.predict(self.preprocessed, verbose=0)
            idxs = np.argmax(probs, axis=1) # Index 0=empty, 1-9=digits
            confs = np.max(probs, axis=1)

            # --- Uses new EMPTY_LABEL=0 ---
            # If index `i` is not EMPTY_LABEL (0) and confidence is high, keep `i`.
            # Otherwise, map to 0 (Sudoku empty convention).
            final = [(i if (i != EMPTY_LABEL and c >= sudoku_recogniser.FINAL_CONFIDENCE_THRESHOLD) else 0)
                     for i, c in zip(idxs, confs)]
            # -----------------------------
            pred_grid = np.asarray(final).reshape(GRID_SIZE, GRID_SIZE)
            conf_grid = confs.reshape(GRID_SIZE, GRID_SIZE)

            print("Ground truth:")
            sudoku_recogniser.print_sudoku_grid(self.gt_grid, threshold=1.1) # Show all GT digits
            print("Prediction (Thresholded):")
            sudoku_recogniser.print_sudoku_grid(pred_grid, conf_grid) # Shows digits passing threshold
            ok = (pred_grid == self.gt_grid).sum()
            total = self.gt_grid.size
            print(f"Accuracy (Thresholded) {ok}/{total} = {ok/total:.4f}")

            # Optional: Show raw predictions before thresholding
            raw_pred_grid = idxs.reshape(GRID_SIZE, GRID_SIZE)
            print("Raw Prediction (0=empty):")
            sudoku_recogniser.print_sudoku_grid(raw_pred_grid, conf_grid, threshold=-0.1) # Show all raw preds
            raw_ok = (raw_pred_grid == (self.gt_grid.astype(int))).sum() # Compare raw prediction to GT (0 vs 0, 1 vs 1 etc)
            print(f"Accuracy (Raw) {raw_ok}/{total} = {raw_ok/total:.4f}\n---")


        except Exception as e:
            print(f"[Callback Error] Failed during prediction or display: {e}\n---")


# ------------------------------------------------------------------ #
# 8.  CLI-helper (unchanged, but uses fixed classifier)
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    FORCE_TRAIN = True # Set to True to force retraining even if model exists
    model_file = Path(MODEL_FILENAME)

    if FORCE_TRAIN and model_file.exists():
        print(f"FORCE_TRAIN is True. Deleting existing model: {model_file}")
        try:
            model_file.unlink()
        except OSError as e:
            print(f"Error deleting model file: {e}")

    # Instantiate classifier, requiring training if model doesn't exist or FORCE_TRAIN is True
    clf = DigitClassifier(training_required=(FORCE_TRAIN or not model_file.exists()))

    # Train if the model wasn't loaded successfully
    if clf.model is None:
        print("Model not loaded or invalid. Starting training...")
        clf.train() # This will build the model if needed
    else:
        print("Model loaded successfully. Skipping training unless forced.")
        # If training was forced but loading somehow succeeded (e.g., error during deletion),
        # we might still want to train. Check the flag again.
        if FORCE_TRAIN:
             print("FORCE_TRAIN is True. Retraining...")
             clf.train()


    # Perform sanity check only if a model exists after potential training
    if clf.model:
        print("\nQuick sanity check:")
        # Test case 1: Vertical line (should ideally be '1')
        dummy1 = np.zeros((50, 50), np.uint8)
        cv2.line(dummy1, (25, 10), (25, 40), 255, 4) # Thicker line
        d1, c1 = clf.recognise(dummy1, 0.1) # Lower threshold for testing
        print(f"Vertical stroke → {d1}  (conf {c1:.3f})")

        # Test case 2: Blank image (should be '0')
        blank = np.zeros((50, 50), np.uint8)
        d0, c0 = clf.recognise(blank, 0.1) # Lower threshold for testing
        print(f"Blank cell      → {d0}  (conf {c0:.3f})")

        # Test case 3: Horizontal line (should ideally be uncertain or maybe '7'/'1'?)
        dummy7 = np.zeros((50, 50), np.uint8)
        cv2.line(dummy7, (10, 25), (40, 25), 255, 4) # Thicker line
        d7, c7 = clf.recognise(dummy7, 0.1) # Lower threshold for testing
        print(f"Horiz stroke    → {d7}  (conf {c7:.3f})")
    else:
        print("\nSanity check skipped: No valid model available.")