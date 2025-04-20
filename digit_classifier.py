"""
SudokuBot – digit classifier
Fixed version 2025‑04‑20
Async data generator version 2025‑04‑21
Serialization and Torch backend fit fix 2025-04-22

Major fixes
• data generator now really yields balanced batches
• preprocessing is tolerant – almost never rejects a cell
• data generator uses keras.utils.Sequence for asynchronous buffering
• model uses string activation for proper serialization
• removed workers/use_multiprocessing from fit/evaluate for torch backend compatibility
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
        def render_sudoku(self, allow_empty=True): return None, None, None
    def generate_and_save_test_example(): return Path("dummy_test.png"), np.zeros((9,9), dtype=int)
    def extract_cells_from_image(path, debug=False): return [], None, None
    def rectify_grid(img, corners): return None
    def split_into_cells(rect): return [], None
    class sudoku_recogniser:
        FINAL_CONFIDENCE_THRESHOLD = 0.9
        @staticmethod
        def print_sudoku_grid(grid, confs=None, threshold=0.0): pass


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
            # Ensure renderer is available
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

            # iterate shuffled cell indices
            idxs = list(range(self.total_cells))
            random.shuffle(idxs) # Use standard random, should be fine per worker

            for idx in idxs:
                # Check if we have enough of both types already
                if n_dig >= self.want_digits and n_emp >= self.want_empty:
                    break

                cell = cells[idx]
                # Determine label (0 becomes EMPTY_LABEL)
                flat_gt = gt_grid.flatten()
                label = EMPTY_LABEL if flat_gt[idx] == 0 else flat_gt[idx]

                # Preprocess the cell
                proc = self.preprocess_func(cell)
                if proc is None:  # should be rare now
                    continue

                # Balance bookkeeping and add if needed
                if label == EMPTY_LABEL:
                    if n_emp < self.want_empty:
                        n_emp += 1
                        xs.append(proc)
                        ys.append(label)
                else: # It's a digit
                    if n_dig < self.want_digits:
                        n_dig += 1
                        xs.append(proc)
                        ys.append(label)

        # At this point we should have a perfectly balanced batch
        # Convert to numpy arrays
        x_arr = np.asarray(xs, dtype="float32")[..., np.newaxis]
        y_arr = np.asarray(ys, dtype="int64")

        # Shuffle the final batch
        p = np.random.permutation(self.batch_size)
        x_batch = x_arr[p]
        y_batch = y_arr[p]

        # Optional histo print for debugging
        if self.debug_print and (index + 1) % 50 == 0:
             # Use try-except for getpid() in case it's not available/needed
             try:
                 pid_info = f" worker {os.getpid()}"
             except AttributeError:
                 pid_info = ""
             print(f"[{self.name}{pid_info}] Batch {index+1}/{self.steps} label histogram:",
                   np.bincount(y_batch, minlength=NUM_CLASSES))

        del xs, ys, x_arr, y_arr
        return x_batch, y_batch

    def on_epoch_end(self):
        """Called by Keras at the end of each epoch."""
        # Can be used to re-initialize things if needed between epochs
        pass


# ------------------------------------------------------------------ #
# 6.  layer helpers (modified for serialization)
# ------------------------------------------------------------------ #
def _norm():
    """Returns a GroupNormalization layer instance."""
    # Specify groups explicitly if needed, otherwise defaults usually work well.
    # Example: layers.GroupNormalization(groups=8) or layers.GroupNormalization(groups=-1)
    return layers.GroupNormalization()

# Use the string identifier directly in the model definition for serialization
_ACTIVATION = "gelu"


# ------------------------------------------------------------------ #
# 7.  classifier object
# ------------------------------------------------------------------ #
class DigitClassifier:
    """
    Handles loading, training and inference of the CNN digit classifier.
    Uses SudokuDataSequence for efficient, asynchronous data generation.
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
            print(f"Attempting to load model from: {self.model_path}")
            try:
                # Load model with custom objects if necessary, though not needed here
                # after switching to string activation.
                self.model = keras.models.load_model(self.model_path)
                # Basic check on input shape
                if self.model.input_shape[1:3] != self._model_input_size:
                    print(f"[Warning] Stored model input size {self.model.input_shape[1:3]} "
                          f"differs from expected {self._model_input_size}")
                print("Digit‑classifier model loaded from disk.")
            except Exception as e:
                print(f"[Error] failed to load model – will train from scratch.\nError details: {e}")
                self.model = None # Ensure model is None if loading failed
        else:
            if training_required:
                print("Training required flag is set.")
            if not self.model_path.exists():
                print(f"Model file not found at {self.model_path}.")
            print("Model will be trained from scratch.")


    # -------------------------------------------------------------- #
    # backbone (modified to use string activation)
    # -------------------------------------------------------------- #
    def _build_cnn_model(self) -> keras.Model:
        """Simple CNN using string activation for serialization."""
        cfg = [32, 32, 64, 64, 96, 96, 96, 128, 128, 128, 128, 192, 192]
        pool_at = {1, 3, 6, 10}

        x_in = keras.Input(shape=MODEL_INPUT_SHAPE)
        x = x_in
        for i, f in enumerate(cfg):
            x = layers.Conv2D(f, 3, padding="same", use_bias=False, name=f'conv_{i}')(x)
            # Use the string identifier for activation
            x = layers.Activation(_ACTIVATION, name=f'act_{i}')(x)
            x = _norm()(x) # GroupNormalization layer names itself
            if i in pool_at:
                x = layers.MaxPooling2D(2, name=f'pool_{i}')(x)

        # 1×1 bottleneck
        x = layers.Conv2D(256, 1, use_bias=False, name='conv_bottleneck')(x)
        # Use the string identifier for activation
        x = layers.Activation(_ACTIVATION, name='act_bottleneck')(x)
        x = _norm()(x)

        # classifier head
        x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
        # Use the string identifier for activation
        x = layers.Dense(128, activation=_ACTIVATION, name='dense_1')(x)
        x = layers.Dense(64, activation=_ACTIVATION, name='dense_2')(x)
        y_out = layers.Dense(NUM_CLASSES, activation="softmax", name='output_softmax')(x)

        model = models.Model(x_in, y_out, name="simplenet_digits_gn_gelu") # Updated name
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=3e-4), # Use learning_rate arg
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        # model.summary() # Summary will be printed when model is built during training
        return model

    # -------------------------------------------------------------- #
    # preprocessing (fixed – tolerant)
    # -------------------------------------------------------------- #
    def _preprocess_cell_for_model(self, cell: np.ndarray) -> Optional[np.ndarray]:
        """
        Convert raw cell → 28×28 float32 in [0,1].
        Never raises; returns None only if `cell` itself is invalid.
        """
        if cell is None or cell.size < 10: # Basic check for valid input
            return None

        # Ensure input is grayscale uint8
        if cell.ndim == 3 and cell.shape[2] == 3:
            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        elif cell.ndim == 2:
            gray = cell
        else: # Unexpected shape
             return None # Or handle other cases like RGBA if needed

        if gray.dtype != np.uint8:
             # Attempt to convert safely if possible, e.g., from float
             if np.issubdtype(gray.dtype, np.floating):
                 gray = np.clip(gray * 255, 0, 255).astype(np.uint8)
             else:
                 try:
                     gray = gray.astype(np.uint8)
                 except (ValueError, TypeError):
                     return None # Cannot convert to uint8

        # Adaptive thresholding
        try:
            # Ensure block size is odd and appropriate for cell size
            h, w = gray.shape[:2]
            blk = max(3, min(h, w) // 4)
            if blk % 2 == 0: blk += 1 # Ensure odd
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blk, 7
            )
        except cv2.error as e:
            # Fallback to Otsu if adaptive fails (e.g., very small image)
            # print(f"[Warning] Adaptive threshold failed ({e}), falling back to Otsu.")
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find non-zero points (the digit)
        pts = cv2.findNonZero(thresh)
        if pts is None:  # Looks empty – return black canvas
            return np.zeros(self._model_input_size, dtype="float32")

        # Get bounding box of the digit
        x, y, w, h = cv2.boundingRect(pts)
        if w <= 0 or h <= 0: # Check width and height strictly positive
            return np.zeros(self._model_input_size, dtype="float32")

        # Extract the region of interest (ROI)
        roi = thresh[y : y + h, x : x + w]

        # Calculate scaling factor to fit within TARGET_CELL_CONTENT_SIZE
        scale = min(
            TARGET_CELL_CONTENT_SIZE / max(1, w),
            TARGET_CELL_CONTENT_SIZE / max(1, h),
        )
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        # Resize ROI
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        try:
            resized = cv2.resize(roi, (new_w, new_h), interpolation=interp)
        except cv2.error as e:
             # Handle potential resize errors (e.g., zero dimensions somehow)
             # print(f"[Warning] Resizing failed ({e}), returning empty canvas.")
             return np.zeros(self._model_input_size, dtype="float32")


        # Create a black canvas of the target model input size
        canvas = np.zeros(self._model_input_size, np.uint8)

        # Calculate top-left corner to center the digit
        top = max(0, (self._model_input_size[0] - new_h) // 2)
        left = max(0, (self._model_input_size[1] - new_w) // 2)

        # Define slices ensuring they don't exceed canvas bounds
        h_slice = slice(top, min(top + new_h, self._model_input_size[0]))
        w_slice = slice(left, min(left + new_w, self._model_input_size[1]))

        # Calculate the actual height/width of the slices
        canvas_h = h_slice.stop - h_slice.start
        canvas_w = w_slice.stop - w_slice.start

        # Place the resized ROI onto the canvas, ensuring dimensions match
        try:
            canvas[h_slice, w_slice] = resized[:canvas_h, :canvas_w]
        except ValueError as e:
            # Mismatch can happen if calculations are off slightly
            # print(f"[Warning] Error placing ROI on canvas ({e}), returning empty canvas.")
            return np.zeros(self._model_input_size, dtype="float32")


        # Normalize to [0, 1] float32
        return canvas.astype("float32") / 255.0

    # -------------------------------------------------------------- #
    # training (updated for torch backend Sequence usage)
    # -------------------------------------------------------------- #
    def train(
        self,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        train_steps: int = TRAIN_STEPS_PER_EPOCH,
        val_steps: int = VALIDATION_STEPS,
    ) -> None:
        print(f"\nTraining: epochs={epochs} batch={batch_size} (Parallelism handled by Keras/Backend)")
        try:
            test_img_path, test_gt = generate_and_save_test_example()
            # Ensure test_img_path is valid before creating callback
            if not Path(test_img_path).exists():
                 raise FileNotFoundError(f"Test image not found at {test_img_path}")
            epoch_cb = EpochTestCallback(test_img_path, test_gt, self)
        except Exception as e:
            print(f"[Warning] Epoch test callback disabled ({e})")
            epoch_cb = None

        # Create Sequence instances for training and validation
        # Pass the instance method directly as the callable
        train_seq = SudokuDataSequence(
            renderer=SudokuRenderer(),
            batch_size=batch_size,
            steps_per_epoch=train_steps,
            preprocess_func=self._preprocess_cell_for_model,
            input_size=MODEL_INPUT_SHAPE,
            name="TrainGen"
        )
        val_seq = SudokuDataSequence(
            renderer=SudokuRenderer(),
            batch_size=batch_size,
            steps_per_epoch=val_steps,
            preprocess_func=self._preprocess_cell_for_model,
            input_size=MODEL_INPUT_SHAPE,
            name="ValGen"
        )

        if self.model is None:
            self.model = self._build_cnn_model()
            print("Built new model:")
            self.model.summary(line_length=100) # Print summary when building new model

        # Callbacks list
        cbs: list[callbacks.Callback] = [
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5, # Stop after 5 epochs with no improvement
                restore_best_weights=True,
                verbose=1,
            ),
            callbacks.ModelCheckpoint(
                filepath=self.model_path,
                monitor="val_loss",
                save_best_only=True, # Save only the best model
                verbose=1,
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.2, # Reduce LR by factor of 5
                patience=3, # Reduce after 3 epochs with no improvement
                min_lr=1e-6,
                verbose=1,
            ),
        ]
        if epoch_cb and epoch_cb.preprocessed is not None:
            cbs.append(epoch_cb)

        # Use the Sequence objects directly in fit
        # Keras+Torch handles parallelism internally for Sequence
        print("Starting model training...")
        history = self.model.fit(
            train_seq,
            epochs=epochs,
            validation_data=val_seq,
            callbacks=cbs,
            verbose=1,
            # DO NOT pass workers/use_multiprocessing for torch backend
        )

        print("\nFinal evaluation using best weights (restored by EarlyStopping):")
        # Recreate a sequence for final evaluation
        eval_seq = SudokuDataSequence(
            renderer=SudokuRenderer(),
            batch_size=batch_size,
            steps_per_epoch=val_steps, # Use validation steps count
            preprocess_func=self._preprocess_cell_for_model,
            input_size=MODEL_INPUT_SHAPE,
            name="EvalGen"
        )
        # Evaluate the model (should have best weights restored)
        loss, acc = self.model.evaluate(
            eval_seq,
            verbose=1,
            # DO NOT pass workers/use_multiprocessing for torch backend
        )
        print(f"Final Validation Loss: {loss:.4f}")
        print(f"Final Validation Accuracy: {acc:.4f}")

        # Explicitly save the final model (best weights should be loaded)
        try:
            self.model.save(self.model_path)
            print(f"Model saved successfully to {self.model_path}")
        except Exception as e:
            print(f"[Error] Failed to save the final model: {e}")


        del train_seq, val_seq, eval_seq # Explicitly delete sequences
        gc.collect() # Suggest garbage collection

    # -------------------------------------------------------------- #
    # inference
    # -------------------------------------------------------------- #
    @torch.no_grad() # Decorator for inference mode with PyTorch backend
    def recognise(
        self,
        cell: np.ndarray,
        confidence_threshold: float = 0.7,
    ) -> Tuple[int, float]:
        """Recognise a single digit cell image."""
        if self.model is None:
            print("[Error] Recognise called but model is not loaded.")
            return 0, 0.0 # Return empty, 0 confidence

        # Preprocess the input cell
        proc = self._preprocess_cell_for_model(cell)
        if proc is None:
            # Return 0 (empty) with 0 confidence if preprocessing fails
            return 0, 0.0

        # Add batch and channel dimensions -> (1, H, W, 1)
        x_np = proc[np.newaxis, ..., np.newaxis]

        # Use Keras predict method for backend abstraction
        # It handles the conversion to the appropriate tensor type (e.g., torch.Tensor)
        try:
            probs = self.model.predict(x_np, verbose=0)[0] # predict expects numpy, returns numpy
        except Exception as e:
             print(f"[Error] Model prediction failed: {e}")
             return 0, 0.0

        # Find the predicted class index and confidence
        idx = int(np.argmax(probs))
        conf = float(probs[idx])

        # Treat low confidence or explicit EMPTY_LABEL prediction as empty (0)
        if idx == EMPTY_LABEL or conf < confidence_threshold:
            # Return 0 (representing empty) but retain the actual confidence
            # of the top prediction (which might be EMPTY_LABEL or a low-conf digit)
            return 0, conf
        # Otherwise, return the recognised digit (1-9) and its confidence
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
        frequency: int = 1, # Run every epoch by default
    ) -> None:
        super().__init__()
        self.frequency = max(1, frequency)
        self.gt_grid = gt_grid
        self.classifier = classifier # Keep reference to the classifier instance
        self.test_img_path = str(test_img_path) # Ensure path is string
        self.preprocessed = None # Initialize as None

        print(f"[Callback] Initializing with test image: {self.test_img_path}")
        try:
            # Extract cells from the test image
            cells, _, _ = extract_cells_from_image(self.test_img_path, debug=False)
            if not cells or len(cells) != GRID_SIZE * GRID_SIZE:
                print(f"[Callback] Failed to extract {GRID_SIZE*GRID_SIZE} cells from test image.")
                return # Leave self.preprocessed as None

            # Preprocess each cell using the classifier's method
            buf = []
            for i, cell_img in enumerate(cells):
                proc = self.classifier._preprocess_cell_for_model(cell_img)
                if proc is None:
                    # Use zeros if preprocessing fails for a cell
                    # print(f"[Callback Warning] Preprocessing failed for cell {i}, using zeros.")
                    proc = np.zeros(self.classifier._model_input_size, dtype="float32")
                buf.append(proc)

            # Add channel dimension for the model input -> (81, H, W, 1)
            self.preprocessed = np.asarray(buf, dtype="float32")[..., np.newaxis]
            print("[Callback] Test example prepared successfully.")

        except FileNotFoundError:
             print(f"[Callback Error] Test image file not found at {self.test_img_path}. Callback disabled.")
        except Exception as e:
            print(f"[Callback Error] Failed preparing test example: {e}. Callback disabled.")
            self.preprocessed = None


    def on_epoch_end(self, epoch, logs=None):
        """Runs at the end of each epoch."""
        # Check if callback is enabled and frequency matches
        if self.model is None or self.preprocessed is None or (epoch + 1) % self.frequency != 0:
            return

        print(f"\n--- Running Epoch {epoch+1} Test Callback ---")
        logs = logs or {}
        val_loss = logs.get('val_loss', -1)
        val_acc = logs.get('val_accuracy', -1) # Keras uses 'val_accuracy'
        print(f"Epoch {epoch+1} - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")


        try:
            # Use the model attached to the callback by Keras
            probs = self.model.predict(self.preprocessed, verbose=0) # Shape (81, NUM_CLASSES)
            idxs = np.argmax(probs, axis=1) # Shape (81,)
            confs = np.max(probs, axis=1) # Shape (81,)

            # Apply final confidence threshold for Sudoku grid recognition logic
            final_preds = [
                i if (i != EMPTY_LABEL and c >= sudoku_recogniser.FINAL_CONFIDENCE_THRESHOLD) else 0
                for i, c in zip(idxs, confs)
            ]
            pred_grid = np.asarray(final_preds).reshape(GRID_SIZE, GRID_SIZE)
            conf_grid = confs.reshape(GRID_SIZE, GRID_SIZE) # Grid of confidences

            print("Ground Truth Grid:")
            sudoku_recogniser.print_sudoku_grid(self.gt_grid, threshold=1.1) # threshold > 1 shows all GT
            print("Predicted Grid (Thresholded):")
            sudoku_recogniser.print_sudoku_grid(pred_grid, conf_grid) # Uses default threshold

            # Calculate accuracy against ground truth
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
    FORCE_TRAIN = False # Set to True to always retrain, False to load if exists

    # Check if model exists and delete if FORCE_TRAIN is True
    model_file = Path(MODEL_FILENAME)
    if FORCE_TRAIN and model_file.exists():
        print(f"Force training enabled: deleting existing model '{model_file}'")
        try:
            model_file.unlink()
        except OSError as e:
            print(f"Error deleting model file: {e}")

    # Instantiate the classifier
    # training_required is True only if FORCE_TRAIN is True AND model was deleted/didn't exist
    clf = DigitClassifier(model_path=model_file, training_required=FORCE_TRAIN)

    # Train if the model wasn't loaded successfully
    if clf.model is None:
        print("Model not loaded or needs training.")
        try:
            clf.train(
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                train_steps=TRAIN_STEPS_PER_EPOCH,
                val_steps=VALIDATION_STEPS,
                # No workers/use_multiprocessing args needed for torch backend fit
            )
        except Exception as train_error:
             print(f"\n--- Training Failed ---")
             print(f"An error occurred during training: {train_error}")
             # Optionally re-raise or exit
             # raise train_error
             exit(1) # Exit if training fails
    else:
        print("Model loaded successfully.")

    # Perform sanity check if model is available (either loaded or just trained)
    if clf.model:
        print("\n--- Running Sanity Check ---")
        try:
            # Test 1: Vertical stroke (should ideally be '1')
            dummy_line = np.zeros((50, 50), np.uint8)
            cv2.line(dummy_line, (25, 10), (25, 40), 255, 3) # White line
            d1, c1 = clf.recognise(dummy_line, confidence_threshold=0.5) # Lower threshold for test
            print(f"Vertical stroke → Predicted: {d1} (Confidence: {c1:.3f})")

            # Test 2: Blank cell (should be '0')
            blank = np.zeros((50, 50), np.uint8)
            d0, c0 = clf.recognise(blank, confidence_threshold=0.1) # Very low threshold
            print(f"Blank cell      → Predicted: {d0} (Confidence: {c0:.3f})")

            # Test 3: Noisy blank cell (should ideally be '0')
            noisy_blank = np.random.randint(0, 30, size=(50, 50), dtype=np.uint8)
            dn, cn = clf.recognise(noisy_blank, confidence_threshold=0.1)
            print(f"Noisy blank     → Predicted: {dn} (Confidence: {cn:.3f})")

            # Test 4: Try loading a real digit image if available (e.g., MNIST sample)
            # This part requires having a sample image file.
            sample_digit_path = Path("sample_digit_7.png") # Example path
            if sample_digit_path.exists():
                 sample_img = cv2.imread(str(sample_digit_path), cv2.IMREAD_GRAYSCALE)
                 if sample_img is not None:
                     ds, cs = clf.recognise(sample_img, confidence_threshold=0.5)
                     print(f"Sample digit ({sample_digit_path.name}) → Predicted: {ds} (Confidence: {cs:.3f})")
                 else:
                     print(f"Could not read sample digit image: {sample_digit_path}")
            else:
                 print(f"Sample digit image not found ({sample_digit_path}), skipping test.")

        except Exception as sanity_error:
            print(f"\n--- Sanity Check Failed ---")
            print(f"An error occurred during sanity check: {sanity_error}")

    else:
        print("\nModel is not available. Cannot run sanity check.")

    print("\nScript finished.")