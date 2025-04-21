"""
SudokuBot – digit classifier
Improved version 2025‑04‑21

Major fixes
• data generator now really yields balanced batches
• preprocessing is tolerant – almost never rejects a cell
• **Improved CNN architecture (ResNet-style) for higher accuracy**
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
from pathlib import Path
from typing import Callable, Generator, Optional, Tuple

import cv2
import numpy as np
import torch
import keras
from keras import callbacks, layers, models, activations, regularizers

# ------------------------------------------------------------------ #
# 3.  project‑local imports
# ------------------------------------------------------------------ #
# Assume these exist in the same directory or are installable
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
    # Provide dummy implementations or raise error if essential
    GRID_SIZE = 9
    class SudokuRenderer:
        def render_sudoku(self, allow_empty=True): return None, None, None
    def generate_and_save_test_example(): return Path("dummy_test.png"), np.zeros((9,9), dtype=int)
    def extract_cells_from_image(path, debug=False): return [], None, None
    class sudoku_recogniser:
        FINAL_CONFIDENCE_THRESHOLD = 0.9
        @staticmethod
        def print_sudoku_grid(grid, confs=None, threshold=0.0): pass


# ------------------------------------------------------------------ #
# 4.  constants
# ------------------------------------------------------------------ #
MODEL_FILENAME = "sudoku_digit_classifier_resnet.keras" # Changed filename

MODEL_INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 11  # digits 0‑9 + “empty”
EMPTY_LABEL = 10

TARGET_CELL_CONTENT_SIZE = 26          # preprocessing (Increased slightly)
TARGET_DIGIT_RATIO = 1.5               # 60 % digits / 40 % empty

EPOCHS = 10 # Increased epochs slightly, EarlyStopping will handle it
STEPS_PER_EPOCH = 150 # Increased steps slightly
BATCH_SIZE = 256
VALIDATION_STEPS = 50

DataBatch = Tuple[np.ndarray, np.ndarray]

# ------------------------------------------------------------------ #
# 5.  balanced data generator (fixed) - unchanged
# ------------------------------------------------------------------ #
def sudoku_data_generator(
    renderer: SudokuRenderer,
    batch_size: int,
    preprocess_func: Callable[[np.ndarray], Optional[np.ndarray]],
    input_size: Tuple[int, int, int],
    target_digit_ratio: float = TARGET_DIGIT_RATIO,
) -> Generator[DataBatch, None, None]:
    """
    Yield *balanced* batches of (cell, label).

    The function never gives up early – it keeps sampling Sudokus until the
    required number of digit and empty cells has been collected.
    """
    total_cells = GRID_SIZE * GRID_SIZE
    # Calculate exact numbers needed for balance
    num_digits_float = batch_size * target_digit_ratio / (1 + target_digit_ratio)
    want_digits = int(round(num_digits_float))
    want_empty = batch_size - want_digits

    # Ensure the sum is exactly batch_size after rounding
    if want_digits + want_empty != batch_size:
         # Adjust the one closer to its float representation's rounding direction
         if num_digits_float - int(num_digits_float) >= 0.5: # Rounded up
             want_digits = batch_size - want_empty
         else: # Rounded down
             want_empty = batch_size - want_digits

    in_h, in_w = input_size[:2]

    batch_counter = 0
    while True:
        xs, ys = [], []
        n_dig = n_emp = 0

        needed_digits = want_digits
        needed_empty = want_empty

        while needed_digits > 0 or needed_empty > 0:
            img, gt_grid, corners = renderer.render_sudoku(allow_empty=True)
            if img is None or corners is None:
                continue
            rectified = rectify_grid(img, corners)
            if rectified is None:
                continue
            cells, _ = split_into_cells(rectified)
            if len(cells) != total_cells:
                continue

            # iterate shuffled cell indices
            idxs = list(range(total_cells))
            random.shuffle(idxs)

            for idx in idxs:
                if needed_digits <= 0 and needed_empty <= 0:
                    break # Batch full

                cell = cells[idx]
                # Handle potential None from gt_grid (though unlikely with allow_empty=True)
                gt_val = gt_grid.flat[idx] if gt_grid is not None else 0
                label = EMPTY_LABEL if gt_val == 0 else gt_val

                # Check if we still need this type of label
                is_empty = (label == EMPTY_LABEL)
                if is_empty and needed_empty <= 0:
                    continue
                if not is_empty and needed_digits <= 0:
                    continue

                proc = preprocess_func(cell)
                if proc is None:                       # should be rare now
                    continue

                # Add to batch and decrement needed count
                xs.append(proc)
                ys.append(label)
                if is_empty:
                    needed_empty -= 1
                else:
                    needed_digits -= 1


        # at this point we have a perfectly balanced batch
        x_arr = np.asarray(xs, dtype="float32")[..., np.newaxis]
        y_arr = np.asarray(ys, dtype="int64")

        # Sanity check batch size and balance (optional)
        # assert len(xs) == batch_size, f"Batch size mismatch: expected {batch_size}, got {len(xs)}"
        # counts = np.bincount(y_arr, minlength=NUM_CLASSES)
        # assert counts[EMPTY_LABEL] == want_empty, f"Empty count mismatch: expected {want_empty}, got {counts[EMPTY_LABEL]}"
        # assert np.sum(counts[:EMPTY_LABEL]) == want_digits, f"Digit count mismatch: expected {want_digits}, got {np.sum(counts[:EMPTY_LABEL])}"


        p = np.random.permutation(batch_size) # Shuffle within the batch
        batch_counter += 1

        # optional histo print for debugging
        if (
            os.environ.get("SUDOKU_DEBUG_GENERATOR", "0") == "1"
            and batch_counter % 500 == 0
        ):
            print(f"Batch {batch_counter} label histogram:", np.bincount(y_arr, minlength=NUM_CLASSES))

        yield x_arr[p], y_arr[p]
        del xs, ys, x_arr, y_arr
        gc.collect()


# ------------------------------------------------------------------ #
# 6.  layer helpers (Removed _norm, using BN directly)
# ------------------------------------------------------------------ #
# Using ReLU directly in layers or via layers.Activation('relu')

# ------------------------------------------------------------------ #
# 7.  classifier object
# ------------------------------------------------------------------ #
class DigitClassifier:
    """
    Handles loading, training and inference of the CNN digit classifier.
    Uses an improved ResNet-style architecture.
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
                # When loading custom objects like custom activation or layers (if any were used),
                # you might need custom_objects={'CustomLayer': CustomLayer}
                self.model = keras.models.load_model(self.model_path)
                if self.model.input_shape[1:3] != self._model_input_size:
                    print(f"[Warning] Stored model input size {self.model.input_shape[1:3]} "
                          f"differs from expected {self._model_input_size}")
                print(f"Digit-classifier model loaded from {self.model_path}")
            except Exception as e:
                print(f"[Error] Failed to load model from {self.model_path} – will train from scratch ({e})")
                self.model = None # Ensure model is None if loading failed

        # If training is required, or loading failed, ensure model is None
        if training_required and self.model is not None:
             print("Training required, ignoring loaded model.")
             self.model = None
        elif training_required and self.model is None:
             print("Training required, model will be built.")
        elif not training_required and self.model is None:
             print("Model not found or failed to load, and training not required. Classifier will not work.")


    # -------------------------------------------------------------- #
    # ResNet-style building block
    # -------------------------------------------------------------- #
    def _residual_block(self, x, filters, strides=1, activation="relu"):
        """Basic residual block."""
        shortcut = x
        # Downsample shortcut if needed (stride > 1 or different number of filters)
        if strides > 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(
                filters, 1, strides=strides, use_bias=False, kernel_initializer="he_normal"
            )(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        # First convolution
        y = layers.Conv2D(
            filters, 3, strides=strides, padding="same", use_bias=False, kernel_initializer="he_normal"
        )(x)
        y = layers.BatchNormalization()(y)
        y = layers.Activation(activation)(y)

        # Second convolution
        y = layers.Conv2D(
            filters, 3, padding="same", use_bias=False, kernel_initializer="he_normal"
        )(y)
        y = layers.BatchNormalization()(y)

        # Add shortcut
        y = layers.Add()([shortcut, y])
        y = layers.Activation(activation)(y)
        return y

    # -------------------------------------------------------------- #
    # backbone (Improved ResNet-style)
    # -------------------------------------------------------------- #
    def _build_cnn_model(self) -> keras.Model:
        """Builds a small ResNet-style CNN model."""
        activation_func = "relu" # Or keep 'gelu' if preferred

        x_in = keras.Input(shape=MODEL_INPUT_SHAPE)

        # Initial Convolution (Stem) - adjusted for small 28x28 input
        # No initial max pooling needed for 28x28
        x = layers.Conv2D(32, 3, padding="same", use_bias=False, kernel_initializer="he_normal")(x_in)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation_func)(x)
        # x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x) # Optional: if more aggressive downsampling needed early

        # Residual Blocks
        # Block 1 (32 filters)
        x = self._residual_block(x, 32, activation=activation_func)
        x = self._residual_block(x, 32, activation=activation_func)

        # Block 2 (64 filters, downsample)
        x = self._residual_block(x, 64, strides=2, activation=activation_func) # 28x28 -> 14x14
        x = self._residual_block(x, 64, activation=activation_func)

        # Block 3 (128 filters, downsample)
        x = self._residual_block(x, 128, strides=2, activation=activation_func) # 14x14 -> 7x7
        x = self._residual_block(x, 128, activation=activation_func)

        # Block 4 (256 filters, downsample) - Optional, maybe too much for 7x7
        # x = self._residual_block(x, 256, strides=2, activation=activation_func) # 7x7 -> 4x4
        # x = self._residual_block(x, 256, activation=activation_func)

        # Classifier Head
        x = layers.GlobalAveragePooling2D()(x) # Feature vector
        x = layers.Flatten()(x) # Ensure flat vector after GAP

        x = layers.Dense(128, kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x) # BN before activation in dense layers
        x = layers.Activation(activation_func)(x)
        x = layers.Dropout(0.5)(x) # Regularization

        # Removed the intermediate 64-unit dense layer, maybe not needed
        # x = layers.Dense(64, activation=activation_func, kernel_initializer="he_normal")(x)
        # x = layers.Dropout(0.5)(x)

        y_out = layers.Dense(NUM_CLASSES, activation="softmax")(x) # Output layer

        model = models.Model(x_in, y_out, name="resnet_digits")

        # Consider AdamW optimizer if available and needed
        # optimizer = keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4)
        optimizer = keras.optimizers.Adam(learning_rate=3e-4)

        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.summary()
        return model

    # -------------------------------------------------------------- #
    # preprocessing (fixed – tolerant, slightly adjusted target size)
    # -------------------------------------------------------------- #
    def _preprocess_cell_for_model(self, cell: np.ndarray) -> Optional[np.ndarray]:
        """
        Convert raw cell → 28×28 float32 in [0,1].
        Never raises; returns None only if `cell` itself is invalid.
        """
        if cell is None or cell.size < 10: # Basic check for validity
            return None

        # Ensure input is grayscale uint8
        if cell.ndim == 3 and cell.shape[2] == 3:
            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        elif cell.ndim == 2:
            gray = cell
        else: # Unexpected shape
             return None # Or try to handle other cases like RGBA

        # Ensure uint8 type for thresholding
        if gray.dtype != np.uint8:
             # Try to safely convert (e.g., scale if it's float)
             if np.issubdtype(gray.dtype, np.floating):
                 gray = (gray * 255).clip(0, 255).astype(np.uint8)
             else:
                 gray = gray.astype(np.uint8) # Hope for the best

        # --- Thresholding ---
        # Apply slight Gaussian blur before thresholding to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Adaptive thresholding (robust to varying lighting)
        try:
            # Block size needs to be odd and > 1. Choose based on image size.
            block_size = max(5, min(gray.shape[0], gray.shape[1]) // 4)
            if block_size % 2 == 0: block_size += 1 # Ensure odd
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, 5 # C=5 is a decent starting point
            )
        except cv2.error:
            # Fallback to Otsu's method if adaptive fails (e.g., very small image)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # --- Find digit bounding box ---
        pts = cv2.findNonZero(thresh)
        if pts is None:  # Cell is empty or thresholding failed
            return np.zeros(self._model_input_size, dtype="float32") # Return black canvas

        x, y, w, h = cv2.boundingRect(pts)
        if w <= 0 or h <= 0: # Should not happen if pts is not None, but check anyway
            return np.zeros(self._model_input_size, dtype="float32")

        roi = thresh[y : y + h, x : x + w]

        # --- Resize and Center ---
        # Calculate scaling factor to fit ROI into TARGET_CELL_CONTENT_SIZE box
        # Use TARGET_CELL_CONTENT_SIZE (e.g., 26)
        target_size = TARGET_CELL_CONTENT_SIZE
        scale = min(target_size / w, target_size / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        # Resize using INTER_AREA for shrinking
        resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Create target canvas (28x28)
        canvas = np.zeros(self._model_input_size, dtype=np.uint8)

        # Calculate padding to center the resized digit
        pad_top = (self._model_input_size[0] - new_h) // 2
        pad_left = (self._model_input_size[1] - new_w) // 2

        # Place the resized digit onto the canvas
        canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized

        # Normalize to [0, 1] float32
        return canvas.astype("float32") / 255.0

    # ------------------------------------------------------------------ #
    # 7.1  augmentation for training
    # ------------------------------------------------------------------ #
    def _augment_cell(self, proc: np.ndarray) -> np.ndarray:
        """Apply random small rotations, translations and brightness/contrast jitter."""
        h, w = proc.shape
        # 1) random rotation
        angle = random.uniform(-15, 15)
        M_rot = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        proc = cv2.warpAffine(proc, M_rot, (w, h),
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # 2) random translation
        tx, ty = random.uniform(-2, 2), random.uniform(-2, 2)
        M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
        proc = cv2.warpAffine(proc, M_trans, (w, h),
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # 3) photometric jitter (contrast & brightness)
        alpha = random.uniform(0.8, 1.2)
        beta  = random.uniform(-0.1, 0.1)
        proc = np.clip(proc * alpha + beta, 0.0, 1.0).astype("float32")
        return proc

    def _augment_cell_hard(self, proc: np.ndarray) -> np.ndarray:
        """Apply stronger distortions for challenging examples."""
        h, w = proc.shape
        # 1) stronger random rotation
        angle = random.uniform(-30, 30)
        M_rot = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        proc = cv2.warpAffine(proc, M_rot, (w, h),
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # 2) stronger random translation
        tx, ty = random.uniform(-5, 5), random.uniform(-5, 5)
        M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
        proc = cv2.warpAffine(proc, M_trans, (w, h),
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # 3) morphological distortions (erode/dilate)
        if random.random() < 0.5:
            k = random.randint(1, 3)
            kernel = np.ones((k, k), np.uint8)
            if random.random() < 0.5:
                proc = cv2.erode(proc, kernel)
            else:
                proc = cv2.dilate(proc, kernel)
        # 4) perspective jitter
        jitter = min(h, w) * 0.1
        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst_pts = src_pts + np.random.uniform(-jitter, jitter, src_pts.shape).astype(np.float32)
        M_pers = cv2.getPerspectiveTransform(src_pts, dst_pts)
        proc = cv2.warpPerspective(proc, M_pers, (w, h),
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # 5) photometric jitter
        alpha = random.uniform(0.8, 1.2)
        beta  = random.uniform(-0.2, 0.2)
        proc = np.clip(proc * alpha + beta, 0.0, 1.0).astype("float32")
        return proc

    # -------------------------------------------------------------- #
    # training
    # -------------------------------------------------------------- #
    def train(
        self,
        epochs: int = EPOCHS,
        steps_per_epoch: int = STEPS_PER_EPOCH,
        batch_size: int = BATCH_SIZE,
        validation_steps: int = VALIDATION_STEPS,
    ) -> None:
        print(f"\nTraining: epochs={epochs} steps={steps_per_epoch} batch={batch_size}")
        if self.model is None:
            self.model = self._build_cnn_model()
        elif not isinstance(self.model, keras.Model):
             print("[Error] self.model is not a valid Keras model. Cannot train.")
             return

        try:
            test_img_path, test_gt = generate_and_save_test_example()
            # Ensure classifier instance is passed correctly
            epoch_cb = EpochTestCallback(test_img_path, test_gt, self)
            if epoch_cb.preprocessed is None:
                 print("[Warning] EpochTestCallback disabled due to preprocessing issues.")
                 epoch_cb = None # Disable if setup failed
        except Exception as e:
            print(f"[Warning] Epoch-callback disabled during setup ({e})")
            epoch_cb = None

        # Wrap preprocessing with augmentation for training
        def train_preproc(cell):
            proc = self._preprocess_cell_for_model(cell)
            if proc is None:
                return None
            # Mix easy and hard augmentations to oversample challenging cases
            if random.random() < 0.5:
                return self._augment_cell(proc)
            else:
                return self._augment_cell_hard(proc)

        # Use a fresh renderer instance for each generator if state matters
        train_gen = sudoku_data_generator(
            SudokuRenderer(),
            batch_size,
            train_preproc,
            MODEL_INPUT_SHAPE,
        )
        val_gen = sudoku_data_generator(
            SudokuRenderer(),
            batch_size,
            self._preprocess_cell_for_model,
            MODEL_INPUT_SHAPE,
        )
        # ------------------------------------------------------------------
        # Dump a batch of augmented training samples *after* preprocessing/augmentation
        dump_dir = Path("dumped_training_samples")
        dump_dir.mkdir(exist_ok=True)
        try:
            vis_gen = sudoku_data_generator(
                SudokuRenderer(),
                batch_size,
                train_preproc,
                MODEL_INPUT_SHAPE,
            )
            x_vis, y_vis = next(vis_gen)
            n_dump = min(32, x_vis.shape[0])
            for i in range(n_dump):
                # convert float32 [0,1] → uint8 [0,255]
                img = (x_vis[i, ..., 0] * 255).astype(np.uint8)
                label = y_vis[i]
                cv2.imwrite(str(dump_dir / f"sample_{i}_label_{label}.png"), img)
            print(f"[Info] Dumped {n_dump} augmented training samples to {dump_dir}")
        except Exception as e:
            print(f"[Warning] Could not dump training samples: {e}")

        # Callbacks
        cbs: list[callbacks.Callback] = [
            callbacks.EarlyStopping(
                monitor="val_accuracy", # Monitor validation accuracy
                patience=8,          # Increase patience slightly
                restore_best_weights=True,
                verbose=1,
                mode='max' # We want to maximize accuracy
            ),
            callbacks.ModelCheckpoint(
                filepath=self.model_path,
                monitor="val_accuracy", # Save based on best validation accuracy
                save_best_only=True,
                verbose=1,
                mode='max'
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_accuracy", # Reduce LR based on validation accuracy
                factor=0.3,          # More aggressive reduction factor
                patience=4,          # Reduce LR sooner if plateauing
                min_lr=1e-7,
                verbose=1,
                mode='max'
            ),
            # TensorBoard callback (optional, for visualization)
            # callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)
        ]
        if epoch_cb: # Add epoch test callback only if it was initialized successfully
            cbs.append(epoch_cb)

        # Start Training
        history = self.model.fit(
            train_gen,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=cbs,
            verbose=1,
        )

        # Load best weights back if EarlyStopping restored them
        # (ModelCheckpoint already saved the best one, but loading ensures the instance has them)
        if self.model_path.exists():
             print(f"Loading best weights from {self.model_path}")
             self.model.load_weights(self.model_path) # Use load_weights if only weights were saved, or load_model if entire model

        print("\nFinal evaluation using best weights:")
        # Use a fresh generator for final evaluation
        final_eval_gen = sudoku_data_generator(
            SudokuRenderer(),
            batch_size,
            self._preprocess_cell_for_model,
            MODEL_INPUT_SHAPE,
        )
        loss, acc = self.model.evaluate(
            final_eval_gen,
            steps=validation_steps * 2, # Evaluate on more steps
            verbose=1,
        )
        print(f"Final val_loss={loss:.5f}  val_acc={acc:.5f}")

        # Explicitly save the final best model (ModelCheckpoint should have done this, but belt-and-suspenders)
        print(f"Saving final best model to {self.model_path}")
        self.model.save(self.model_path)

        del train_gen, val_gen, final_eval_gen
        gc.collect()

    # -------------------------------------------------------------- #
    # inference
    # -------------------------------------------------------------- #
    @torch.no_grad() # Keep torch decorator if using torch backend
    def recognise(
        self,
        cell: np.ndarray,
        confidence_threshold: float = 0.9, # Increased default threshold
    ) -> Tuple[int, float]:
        """Recognises a single digit cell."""
        if self.model is None:
            print("[Error] Recognise called but model is not loaded.")
            return 0, 0.0

        proc = self._preprocess_cell_for_model(cell)
        if proc is None:
            # This indicates an issue with the input cell itself
            # print("[Debug] Preprocessing returned None for a cell.")
            return 0, 0.0 # Treat as empty/unrecognizable

        # Add batch and channel dimensions: (H, W) -> (1, H, W, 1)
        x = proc[np.newaxis, ..., np.newaxis]

        # Predict using the Keras model
        # The torch.no_grad() context manager is primarily for PyTorch operations.
        # Keras with torch backend should handle inference mode correctly via training=False.
        # If using pure torch tensors were necessary: x_tensor = torch.from_numpy(x).float()
        probs = self.model(x, training=False) # Use training=False for inference

        # Ensure probs is a NumPy array
        if hasattr(probs, 'numpy'): # TF tensor
            probs = probs.cpu().detach().numpy()
        elif isinstance(probs, torch.Tensor): # PyTorch tensor
            probs = probs.cpu().numpy()
        # If it's already numpy, do nothing

        probs = probs[0] # Remove batch dimension

        idx = int(np.argmax(probs))
        conf = float(probs[idx])

        # Return 0 (empty) if classified as EMPTY_LABEL or confidence is too low
        if idx == EMPTY_LABEL or conf < confidence_threshold:
            # Optionally distinguish between low-conf digit and classified-empty
            # if idx == EMPTY_LABEL: print(f"Cell classified as empty (conf {conf:.3f})")
            # else: print(f"Cell classified as {idx} but low conf ({conf:.3f} < {confidence_threshold})")
            return 0, conf # Return 0 for empty/uncertain
        else:
            # Return the classified digit (1-9)
            return idx, conf


# ------------------------------------------------------------------ #
# 8.  epoch‑end callback (Unchanged conceptually, ensure imports/refs are correct)
# ------------------------------------------------------------------ #
class EpochTestCallback(callbacks.Callback):
    def __init__(
        self,
        test_img_path: Path | str,
        gt_grid: np.ndarray,
        classifier: "DigitClassifier", # Pass the classifier instance
        frequency: int = 1,
    ) -> None:
        super().__init__()
        self.frequency = max(1, frequency)
        self.gt_grid = gt_grid.flatten() # Flatten GT grid for easier comparison
        self.classifier = classifier # Store the classifier instance
        self.test_img_path = test_img_path
        self.preprocessed = None # Initialize as None

        # --- Preprocessing moved to on_train_begin ---
        # This ensures the classifier's model is built before preprocessing

    def on_train_begin(self, logs=None):
        # --- Preprocess test image cells here ---
        # Ensures the model (and its input size) exists if built dynamically
        print("[Callback] Preprocessing test example...")
        try:
            cells, _, _ = extract_cells_from_image(self.test_img_path, debug=False)
            if not cells or len(cells) != GRID_SIZE * GRID_SIZE:
                print(f"[Callback] Failed to extract correct number of cells ({len(cells)}) from {self.test_img_path}")
                self.preprocessed = None
                return

            buf = []
            model_input_size = self.classifier._model_input_size # Get from classifier
            for i, cell in enumerate(cells):
                proc = self.classifier._preprocess_cell_for_model(cell)
                if proc is None:
                    print(f"[Callback Warning] Preprocessing failed for cell {i}, using zeros.")
                    proc = np.zeros(model_input_size, dtype="float32")
                buf.append(proc)

            self.preprocessed = np.asarray(buf, dtype="float32")[..., np.newaxis]
            print(f"[Callback] Test example preprocessed successfully ({self.preprocessed.shape}).")

        except Exception as e:
            print(f"[Callback Error] Failed during test example preprocessing: {e}")
            self.preprocessed = None


    def on_epoch_end(self, epoch, logs=None):
        # Check if preprocessing was successful and if it's the right epoch
        if self.preprocessed is None or (epoch + 1) % self.frequency != 0:
            return

        if not hasattr(self.model, 'predict'):
             print("[Callback Error] Model object does not have predict method.")
             return

        print(f"\n--- Epoch {epoch+1} Test Example Evaluation ---")
        try:
            # Use the model attached to the callback (which is the one being trained)
            probs = self.model.predict(self.preprocessed, verbose=0)
            idxs = np.argmax(probs, axis=1)
            confs = np.max(probs, axis=1)

            # Apply the same logic as `recognise` for final prediction
            # Use a reasonable threshold for display purposes
            display_threshold = 0.7 # Lower than final recognition, just for display
            final = [
                i if (i != EMPTY_LABEL and c >= display_threshold) else 0
                for i, c in zip(idxs, confs)
            ]
            pred_grid_flat = np.asarray(final)
            pred_grid = pred_grid_flat.reshape(GRID_SIZE, GRID_SIZE)
            conf_grid = confs.reshape(GRID_SIZE, GRID_SIZE)

            print("Ground Truth:")
            sudoku_recogniser.print_sudoku_grid(self.gt_grid.reshape(GRID_SIZE, GRID_SIZE)) # Reshape GT back
            print("Prediction (Thresholded):")
            sudoku_recogniser.print_sudoku_grid(pred_grid, conf_grid, threshold=display_threshold)

            # Compare against the flattened ground truth
            correct_cells = (pred_grid_flat == self.gt_grid).sum()
            total_cells = GRID_SIZE * GRID_SIZE
            accuracy = correct_cells / total_cells
            print(f"Test Example Accuracy: {correct_cells}/{total_cells} = {accuracy:.4f}")
            print("--- End Epoch Test ---\n")

        except Exception as e:
            print(f"[Callback Error] Failed during prediction or display: {e}")


# ------------------------------------------------------------------ #
# 9.  CLI helper
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    # Set to True to force retraining even if a model file exists
    FORCE_TRAIN = False # Set to True to retrain

    model_file = Path(MODEL_FILENAME)
    train_needed = FORCE_TRAIN or not model_file.exists()

    if FORCE_TRAIN and model_file.exists():
        print(f"FORCE_TRAIN is True. Deleting existing model: {model_file}")
        try:
            model_file.unlink()
            train_needed = True # Ensure flag is set
        except OSError as e:
            print(f"Error deleting existing model: {e}")
            # Decide whether to proceed or exit
            # exit(1)

    # Instantiate the classifier. It will try to load if train_needed is False.
    clf = DigitClassifier(model_path=model_file, training_required=train_needed)

    # Train if needed (either forced or because loading failed/file missing)
    if train_needed:
        print("Starting training process...")
        clf.train()
        # After training, the best model should be saved and loaded by the train method.
        # Verify the model is loaded for the sanity check below.
        if clf.model is None:
             print("[Error] Training finished, but model is still None. Cannot proceed.")
             exit(1) # Exit if training failed to produce a model
    elif clf.model is None:
         print("[Error] Model loading failed, and training was not requested. Cannot proceed.")
         exit(1) # Exit if no model is available
    else:
         print("Model loaded successfully. Skipping training.")


    # Perform sanity check only if the model is available
    if clf.model:
        print("\nQuick sanity check:")

        # Test 1: Vertical stroke (should ideally be 1)
        dummy1 = np.zeros((50, 50), np.uint8)
        cv2.line(dummy1, (25, 10), (25, 40), 255, 4) # Thicker line
        d1, c1 = clf.recognise(dummy1, confidence_threshold=0.5) # Use lower threshold for test
        print(f"Vertical stroke → {d1}  (conf {c1:.3f})")

        # Test 2: Blank cell (should be 0)
        blank = np.zeros((50, 50), np.uint8)
        d0, c0 = clf.recognise(blank, confidence_threshold=0.5)
        print(f"Blank cell      → {d0}  (conf {c0:.3f})")

        # Test 3: A simple digit (e.g., '7')
        dummy7 = np.zeros((50, 50), np.uint8)
        cv2.putText(dummy7, '7', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 3)
        d7, c7 = clf.recognise(dummy7, confidence_threshold=0.5)
        print(f"Digit '7'       → {d7}  (conf {c7:.3f})")

    else:
        print("\nSanity check skipped: No model available.")