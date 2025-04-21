"""
SudokuBot – Digit Classifier (Full Grid Prediction)
Version 2025-04-22

Major Changes:
- Model predicts all 81 cells from the rectified grid image at once.
- Uses a Fully Convolutional Network (FCN) based on ResNet blocks.
- Data generator yields full grid images and corresponding label grids.
- Training data uses puzzles derived from valid, solvable Sudoku solutions.
- Preprocessing and augmentation apply to the full grid image.
"""

# ------------------------------------------------------------------ #
# 1.  choose backend BEFORE importing keras
# ------------------------------------------------------------------ #
import os
import time

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
try:
    # Use the updated renderer and extractor
    from sudoku_renderer import SudokuRenderer, generate_and_save_test_example
    from digit_extractor import (
        GRID_SIZE,
        DEFAULT_RECTIFIED_SIZE, # Use the size defined in extractor
        extract_cells_from_image,
        rectify_grid,
        # split_into_cells is not directly needed by the classifier anymore
    )
    import sudoku_recogniser # Needed for printing grids in callback
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print("Please ensure sudoku_renderer.py, digit_extractor.py, and sudoku_recogniser.py are available.")
    # Provide dummy implementations or raise error if essential
    GRID_SIZE = 9
    DEFAULT_RECTIFIED_SIZE = 252
    class SudokuRenderer:
        def render_sudoku(self, allow_empty=True, grid_spec=None): return None, None, None
    def generate_and_save_test_example(force=False): return Path("dummy_test.png"), np.zeros((9,9), dtype=int)
    def extract_cells_from_image(path, size=252, debug=False): return [], np.zeros((size,size), dtype=np.uint8), None
    class sudoku_recogniser:
        FINAL_CONFIDENCE_THRESHOLD = 0.8
        @staticmethod
        def print_sudoku_grid(grid, confs=None, threshold=0.0): pass


# ------------------------------------------------------------------ #
# 4.  constants
# ------------------------------------------------------------------ #
MODEL_FILENAME = "sudoku_full_grid_classifier_fcn.keras" # New filename for the new model type
# Input shape based on the rectified grid size from digit_extractor
MODEL_INPUT_SHAPE = (DEFAULT_RECTIFIED_SIZE, DEFAULT_RECTIFIED_SIZE, 1) # Grayscale input
NUM_CLASSES = 11  # digits 0-9 + "empty"
EMPTY_LABEL = 10  # Label used for empty cells in the target data (y)

# Training parameters
EPOCHS = 25 # Might need more epochs for a larger model/task
STEPS_PER_EPOCH = 100 # Adjust based on data generation speed and dataset size
BATCH_SIZE = 16 # Reduced batch size due to larger input images
VALIDATION_STEPS = 30

# Type alias for data batches
DataBatch = Tuple[np.ndarray, np.ndarray] # (batch_of_grids, batch_of_labels)

# ------------------------------------------------------------------ #
# 5.  data generator (simplified for full grids)
# ------------------------------------------------------------------ #
def sudoku_data_generator(
    renderer: SudokuRenderer,
    batch_size: int,
    preprocess_func: Callable[[np.ndarray], Optional[np.ndarray]],
    augment_func: Optional[Callable[[np.ndarray], np.ndarray]], # Optional augmentation
    input_shape: Tuple[int, int, int], # e.g., (252, 252, 1)
) -> Generator[DataBatch, None, None]:
    """
    Yields batches of (preprocessed_grid_image, target_label_grid).
    Generates valid Sudoku puzzles on the fly.
    """
    target_h, target_w, target_c = input_shape
    is_grayscale = (target_c == 1)

    batch_counter = 0
    while True:
        batch_x = np.zeros((batch_size, target_h, target_w, target_c), dtype="float32")
        batch_y = np.zeros((batch_size, GRID_SIZE, GRID_SIZE), dtype="int64")
        
        items_in_batch = 0
        while items_in_batch < batch_size:
            # Generate a rendered Sudoku image and its ground truth puzzle grid (0 for empty)
            img, gt_puzzle_grid, corners = renderer.render_sudoku(allow_empty=True) # Let renderer handle difficulty

            if img is None or gt_puzzle_grid is None or corners is None:
                # print("[Generator] Renderer failed, skipping.")
                time.sleep(0.1) # Avoid busy-looping if renderer fails consistently
                continue

            # Rectify the grid using the known corners
            # Use the target input size for rectification directly
            rectified = rectify_grid(img, corners, size=target_h) # Assuming target_h == target_w
            if rectified is None:
                # print("[Generator] Rectification failed, skipping.")
                continue

            # Preprocess the entire rectified grid image
            processed_grid = preprocess_func(rectified)
            if processed_grid is None:
                # print("[Generator] Preprocessing failed, skipping.")
                continue

            # Apply augmentation if provided (usually only for training)
            if augment_func:
                processed_grid = augment_func(processed_grid)

            # Ensure the processed grid has the correct shape (H, W, C)
            if processed_grid.shape != (target_h, target_w, target_c):
                 print(f"[Generator Warning] Processed grid shape mismatch: expected {(target_h, target_w, target_c)}, got {processed_grid.shape}. Skipping.")
                 continue


            # Convert the ground truth puzzle grid (0 for empty) to the target format
            # where empty cells are represented by EMPTY_LABEL (10)
            target_labels = gt_puzzle_grid.copy()
            target_labels[target_labels == 0] = EMPTY_LABEL

            # Add to batch
            batch_x[items_in_batch] = processed_grid
            batch_y[items_in_batch] = target_labels
            items_in_batch += 1

        # Yield the complete batch
        batch_counter += 1
        # Optional debug print
        # if batch_counter % 10 == 0:
        #     print(f"[Generator] Yielding batch {batch_counter}. Example label counts: {np.bincount(batch_y[0].flatten(), minlength=NUM_CLASSES)}")

        yield batch_x, batch_y
        # No need for explicit gc.collect() here usually, Python handles it.


# ------------------------------------------------------------------ #
# 6.  classifier object (handles full grid model)
# ------------------------------------------------------------------ #
class DigitClassifier:
    """
    Handles loading, training and inference of the FCN Sudoku grid classifier.
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
        self._model_input_shape = MODEL_INPUT_SHAPE # e.g., (252, 252, 1)

        if not training_required and self.model_path.exists():
            print(f"Attempting to load model from {self.model_path}...")
            try:
                self.model = keras.models.load_model(self.model_path)
                # Check if loaded model's input shape matches config
                # Keras models sometimes store input shape as (None, H, W, C)
                loaded_shape = self.model.input_shape[1:]
                if loaded_shape != self._model_input_shape:
                    print(f"[Warning] Loaded model input shape {loaded_shape} "
                          f"differs from expected {self._model_input_shape}. Mismatch may cause errors.")
                print(f"Full-grid classifier model loaded successfully.")
            except Exception as e:
                print(f"[Error] Failed to load model from {self.model_path}: {e}")
                print("Will build and train a new model if training is enabled.")
                self.model = None # Ensure model is None if loading failed

        # Handle training_required flag
        if training_required and self.model is not None:
             print("Training required: Ignoring previously loaded model and building a new one.")
             self.model = None
        elif training_required and self.model is None:
             print("Training required: Model will be built.")
        elif not training_required and self.model is None:
             print("Model not found or failed to load, and training not required. Classifier is inactive.")


    # -------------------------------------------------------------- #
    # ResNet-style building block (same as before)
    # -------------------------------------------------------------- #
    def _residual_block(self, x, filters, strides=1, activation="relu"):
        """Basic residual block."""
        shortcut = x
        # Downsample shortcut if needed
        if strides > 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(
                filters, 1, strides=strides, use_bias=False, kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-5) # Added slight L2 regularization
            )(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        # First convolution
        y = layers.Conv2D(
            filters, 3, strides=strides, padding="same", use_bias=False, kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(1e-5)
        )(x)
        y = layers.BatchNormalization()(y)
        y = layers.Activation(activation)(y)

        # Second convolution
        y = layers.Conv2D(
            filters, 3, padding="same", use_bias=False, kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(1e-5)
        )(y)
        y = layers.BatchNormalization()(y)

        # Add shortcut
        y = layers.Add()([shortcut, y])
        y = layers.Activation(activation)(y)
        return y

    # -------------------------------------------------------------- #
    # backbone (FCN for full grid prediction)
    # -------------------------------------------------------------- #
    def _build_cnn_model(self) -> keras.Model:
        """Builds a Fully Convolutional Network (FCN) for grid prediction."""
        activation_func = "relu" # Or 'gelu'

        x_in = keras.Input(shape=self._model_input_shape) # e.g., (252, 252, 1)

        # --- Encoder Path (Downsampling) ---
        # Initial Conv Layer (Stem)
        # Use stride 3 to quickly reduce dimensions: 252 -> 84
        filters = 32
        x = layers.Conv2D(filters, 7, strides=3, padding="same", use_bias=False, kernel_initializer="he_normal")(x_in)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation_func)(x)
        # x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x) # Optional extra pooling

        # Residual Blocks with increasing filters and downsampling
        # Block 1: 84x84 -> 84x84 (filters=32)
        x = self._residual_block(x, filters, strides=1, activation=activation_func)
        x = self._residual_block(x, filters, strides=1, activation=activation_func)

        # Block 2: 84x84 -> 28x28 (filters=64, stride=3)
        filters *= 2 # 64
        x = self._residual_block(x, filters, strides=3, activation=activation_func)
        x = self._residual_block(x, filters, strides=1, activation=activation_func)

        # Block 3: 28x28 -> 28x28 (filters=128) - Stride 1 here
        filters *= 2 # 128
        x = self._residual_block(x, filters, strides=1, activation=activation_func)
        x = self._residual_block(x, filters, strides=1, activation=activation_func)

        # Block 4: 28x28 -> 14x14 (filters=256, stride=2) - Aiming for spatial dim > 9
        filters *= 2 # 256
        x = self._residual_block(x, filters, strides=2, activation=activation_func)
        x = self._residual_block(x, filters, strides=1, activation=activation_func)

        # --- Prediction Head ---
        # At this point, spatial dimension is 14x14. We need 9x9 output.
        # Option 1: Use a Conv layer with stride to reduce size (might lose info)
        # Option 2: Use Adaptive Pooling (if available/easy in Keras backend)
        # Option 3: Use Conv + Upsampling (more complex)
        # Option 4: Use a final Conv layer that implicitly handles the size reduction (less common)

        # Let's try a Conv2D layer designed to output the correct spatial dimensions.
        # We need to get closer to 9x9. Add another block?
        # Block 5: 14x14 -> 7x7 (filters=512, stride=2)
        # filters *= 2 # 512
        # x = self._residual_block(x, filters, strides=2, activation=activation_func)
        # x = self._residual_block(x, filters, strides=1, activation=activation_func)
        # Now spatial dim is 7x7. This is too small.

        # Backtrack: Let's stop at 14x14 (Block 4 output).
        # How to get to (9, 9, NUM_CLASSES)?
        # Use a 1x1 Conv to reduce filters, then maybe resize/crop or use specific conv?

        # Try a final Conv layer with appropriate kernel/padding to target 9x9.
        # Input to this layer is (None, 14, 14, 256)
        # Output needed is (None, 9, 9, NUM_CLASSES)

        # Use a 1x1 convolution to adjust the number of channels first
        x = layers.Conv2D(128, 1, padding='same', activation=activation_func, kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        # Now (None, 14, 14, 128)

        # Use a final Conv2D layer to produce the output map.
        # A kernel size of 6 with valid padding on a 14x14 input gives (14-6+1) = 9x9 output.
        # Stride 1 is implicit.
        x = layers.Conv2D(
            filters=NUM_CLASSES,
            kernel_size=6, # Kernel size chosen to map 14x14 -> 9x9 with 'valid' padding
            padding="valid", # 'valid' means no padding
            activation=None, # Apply softmax separately across the class dimension
            kernel_initializer='he_normal',
            name="final_conv_logits"
        )(x)
        # Output shape should now be (None, 9, 9, NUM_CLASSES)

        # Apply Softmax activation across the last axis (classes)
        y_out = layers.Activation("softmax", name="output_softmax")(x)

        # Create the model
        model = models.Model(x_in, y_out, name="fcn_sudoku_grid")

        # Compile the model
        optimizer = keras.optimizers.Adam(learning_rate=5e-4) # Slightly higher LR?
        # Loss function suitable for integer targets and probability outputs
        loss = "sparse_categorical_crossentropy"
        # Metrics: Accuracy calculated per cell prediction
        metrics = ["accuracy"]

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
        )
        model.summary(line_length=120)
        return model

    # -------------------------------------------------------------- #
    # preprocessing (for full grid image)
    # -------------------------------------------------------------- #
    def _preprocess_grid_for_model(self, rectified_img: np.ndarray) -> Optional[np.ndarray]:
        """
        Prepares the rectified grid image for the FCN model.
        Resizes, converts to grayscale, normalizes to [0, 1], adds channel dim.
        """
        if rectified_img is None or rectified_img.size == 0:
            return None

        target_h, target_w, target_c = self._model_input_shape
        is_grayscale = (target_c == 1)

        # Resize to target input size
        # Use INTER_AREA for shrinking, INTER_LINEAR for enlarging
        current_h, current_w = rectified_img.shape[:2]
        if current_h * current_w > target_h * target_w:
            interp = cv2.INTER_AREA
        else:
            interp = cv2.INTER_LINEAR
        resized = cv2.resize(rectified_img, (target_w, target_h), interpolation=interp)

        # Ensure correct number of channels (grayscale)
        if is_grayscale:
            if resized.ndim == 3 and resized.shape[2] == 3:
                processed = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            elif resized.ndim == 3 and resized.shape[2] == 4:
                processed = cv2.cvtColor(resized, cv2.COLOR_BGRA2GRAY)
            elif resized.ndim == 2:
                processed = resized
            else:
                print(f"[Preprocess Error] Unexpected image shape: {resized.shape}")
                return None
            # Add channel dimension: (H, W) -> (H, W, 1)
            processed = processed[..., np.newaxis]
        else: # If model expected color input (target_c == 3)
            if resized.ndim == 2: # Convert grayscale to BGR
                processed = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            elif resized.ndim == 3 and resized.shape[2] == 4: # Convert BGRA to BGR
                 processed = cv2.cvtColor(resized, cv2.COLOR_BGRA2BGR)
            elif resized.ndim == 3 and resized.shape[2] == 3:
                 processed = resized # Already BGR
            else:
                print(f"[Preprocess Error] Unexpected image shape for color: {resized.shape}")
                return None

        # Normalize to [0, 1] float32
        processed = processed.astype("float32") / 255.0

        return processed


    # ------------------------------------------------------------------ #
    # 7.1  augmentation (for full grid image)
    # ------------------------------------------------------------------ #
    def _augment_grid(self, grid_img: np.ndarray) -> np.ndarray:
        """Apply augmentations to the full grid image."""
        original_shape = grid_img.shape # Remember the original shape (H, W, C)
        h, w = grid_img.shape[:2]
        augmented = grid_img.copy()

        # 1. Small Rotation
        if random.random() < 0.5:
            angle = random.uniform(-8, 8) # Reduced angle for full grid
            M_rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            augmented = cv2.warpAffine(augmented, M_rot, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0) # Black border
            # warpAffine might strip the last dimension if it's 1

        # 2. Small Translation
        if random.random() < 0.5:
            tx = random.uniform(-w * 0.03, w * 0.03) # Max 3% translation
            ty = random.uniform(-h * 0.03, h * 0.03)
            M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
            # warpAffine might strip the last dimension if it's 1
            augmented = cv2.warpAffine(augmented, M_trans, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
        # 3. Brightness/Contrast Jitter
        if random.random() < 0.6:
            alpha = random.uniform(0.85, 1.15) # Contrast
            beta = random.uniform(-0.1, 0.1)   # Brightness
            augmented = np.clip(augmented * alpha + beta, 0.0, 1.0)

        # 4. Gaussian Noise
        if random.random() < 0.3:
            noise = np.random.normal(0, random.uniform(0.01, 0.05), augmented.shape)
            augmented = np.clip(augmented + noise, 0.0, 1.0)

        # Ensure channel dimension exists if it was stripped by warpAffine
        # Check if original had 3 dims (H, W, 1) and current has 2 (H, W)
        if len(original_shape) == 3 and original_shape[2] == 1 and len(augmented.shape) == 2:
            augmented = augmented[..., np.newaxis]

        # Ensure output is float32
        return augmented.astype("float32")


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
        """Trains the full-grid FCN model."""
        print(f"\n--- Training Full Grid Classifier ---")
        print(f"Epochs: {epochs}, Steps/Epoch: {steps_per_epoch}, Batch Size: {batch_size}")
        print(f"Input Shape: {self._model_input_shape}, Output: (9, 9, {NUM_CLASSES})")

        if self.model is None:
            self.model = self._build_cnn_model()
        elif not isinstance(self.model, keras.Model):
             print("[Error] self.model is not a valid Keras model. Cannot train.")
             return

        # Prepare test example for callback
        try:
            test_img_path, test_gt_grid = generate_and_save_test_example()
            if test_img_path is None or test_gt_grid is None:
                 raise ValueError("Failed to generate/load test example.")
            # Pass the classifier instance and the ground truth grid (0 for empty)
            epoch_cb = EpochTestCallback(test_img_path, test_gt_grid, self)
        except Exception as e:
            print(f"[Warning] Failed to set up EpochTestCallback: {e}. Callback disabled.")
            epoch_cb = None

        # Create data generators
        # Training generator uses augmentation
        train_gen = sudoku_data_generator(
            SudokuRenderer(),
            batch_size,
            self._preprocess_grid_for_model,
            self._augment_grid, # Apply augmentation
            self._model_input_shape,
        )
        # Validation generator does not use augmentation
        val_gen = sudoku_data_generator(
            SudokuRenderer(),
            batch_size,
            self._preprocess_grid_for_model,
            None, # No augmentation for validation
            self._model_input_shape,
        )

        # Dump a batch of augmented training samples for visualization
        dump_dir = Path("dumped_training_grids")
        dump_dir.mkdir(exist_ok=True)
        try:
            x_vis, y_vis = next(train_gen) # Get one batch from training gen
            n_dump = min(4, x_vis.shape[0]) # Dump fewer, larger images
            for i in range(n_dump):
                # Convert float32 [0,1] -> uint8 [0,255]
                img = (x_vis[i] * 255).astype(np.uint8)
                # Maybe add label info to filename if needed, but grid is complex
                cv2.imwrite(str(dump_dir / f"sample_grid_{i}.png"), img)
            print(f"[Info] Dumped {n_dump} augmented training grid samples to {dump_dir}")
        except Exception as e:
            print(f"[Warning] Could not dump training samples: {e}")


        # Callbacks
        cbs: list[callbacks.Callback] = [
            callbacks.EarlyStopping(
                monitor="val_accuracy", # Monitor validation accuracy (per-cell)
                patience=10,          # Increased patience for larger model
                restore_best_weights=True,
                verbose=1,
                mode='max'
            ),
            callbacks.ModelCheckpoint(
                filepath=str(self.model_path), # Ensure path is string
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
                mode='max'
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss", # Reduce LR based on validation loss
                factor=0.2,
                patience=4,
                min_lr=1e-7,
                verbose=1,
                mode='min' # Reduce LR when loss plateaus
            ),
            # Optional: TensorBoard
            # callbacks.TensorBoard(log_dir='./logs_fcn', histogram_freq=1)
        ]
        if epoch_cb:
            cbs.append(epoch_cb)

        # Start Training
        print("\nStarting model training...")
        history = self.model.fit(
            train_gen,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=cbs,
            verbose=1, # Use 1 for progress bar, 2 for one line per epoch
        )

        # Ensure the best model is loaded (EarlyStopping might restore, Checkpoint saves)
        if self.model_path.exists():
             print(f"Reloading best weights from {self.model_path}...")
             try:
                 # Use load_model if the whole architecture might change,
                 # or load_weights if only weights are saved/needed.
                 # ModelCheckpoint saves the whole model by default.
                 self.model = keras.models.load_model(self.model_path)
                 print("Best model reloaded.")
             except Exception as e:
                 print(f"[Error] Failed to reload best model after training: {e}")
                 # The model instance might still hold the best weights if EarlyStopping restored them.

        print("\n--- Final Evaluation (using best weights) ---")
        # Use a fresh validation generator for final evaluation
        final_eval_gen = sudoku_data_generator(
            SudokuRenderer(),
            batch_size,
            self._preprocess_grid_for_model,
            None, # No augmentation
            self._model_input_shape,
        )
        loss, acc = self.model.evaluate(
            final_eval_gen,
            steps=validation_steps * 2, # Evaluate on more validation steps
            verbose=1,
        )
        print(f"Final Validation Loss: {loss:.5f}")
        print(f"Final Validation Accuracy (per cell): {acc:.5f}")

        # Explicitly save the final best model again (belt-and-suspenders)
        try:
            print(f"Saving final best model to {self.model_path}")
            self.model.save(self.model_path)
        except Exception as e:
            print(f"[Error] Failed to save final model: {e}")

        del train_gen, val_gen, final_eval_gen, history
        gc.collect()
        print("--- Training Finished ---")


    # -------------------------------------------------------------- #
    # inference (for full grid)
    # -------------------------------------------------------------- #
    @torch.no_grad() # Keep decorator if using torch backend
    def recognise_grid(
        self,
        rectified_img: np.ndarray,
        confidence_threshold: float = 0.80, # Default threshold for accepting a digit
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Recognises digits in a rectified Sudoku grid image using the FCN model.

        Args:
            rectified_img: The (e.g., 252x252) rectified grid image (uint8 or float).
            confidence_threshold: Minimum confidence to accept a digit prediction (1-9).

        Returns:
            A tuple (predicted_grid, confidence_grid):
            - predicted_grid: (9, 9) numpy array with recognised digits (1-9) or 0 for empty/uncertain.
            - confidence_grid: (9, 9) numpy array with confidence scores for the predicted class in each cell.
            Returns (None, None) if inference fails.
        """
        if self.model is None:
            print("[Error] Recognise_grid called but model is not loaded.")
            return None, None

        # 1. Preprocess the input grid image
        proc_grid = self._preprocess_grid_for_model(rectified_img)
        if proc_grid is None:
            print("[Error] Preprocessing failed during recognition.")
            return None, None # Indicate failure

        # 2. Add batch dimension: (H, W, C) -> (1, H, W, C)
        x = proc_grid[np.newaxis, ...]

        # 3. Predict using the Keras model
        try:
            # Use training=False for inference (important for layers like BatchNorm, Dropout)
            raw_output = self.model(x, training=False) # Shape: (1, 9, 9, NUM_CLASSES)
        except Exception as e:
            print(f"[Error] Model prediction failed: {e}")
            return None, None

        # 4. Convert to NumPy and remove batch dimension
        # Keras with torch backend might return torch tensors
        if hasattr(raw_output, 'cpu') and hasattr(raw_output, 'numpy'): # Check if it's a tensor with cpu/numpy methods
             probs = raw_output.cpu().numpy()
        elif isinstance(raw_output, np.ndarray):
             probs = raw_output
        else:
             print(f"[Error] Unexpected model output type: {type(raw_output)}")
             return None, None

        probs = probs[0] # Shape: (9, 9, NUM_CLASSES)

        # 5. Decode probabilities to predictions and confidences
        predicted_indices = np.argmax(probs, axis=-1) # Shape: (9, 9), contains indices 0-10
        confidences = np.max(probs, axis=-1)       # Shape: (9, 9), contains max probability

        # 6. Create the final output grid
        # Initialize with zeros (representing empty/uncertain)
        final_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

        # Iterate through each cell prediction
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                pred_idx = predicted_indices[r, c]
                conf = confidences[r, c]

                # Check if the prediction is a digit (1-9), not EMPTY_LABEL (10),
                # and if the confidence meets the threshold.
                if pred_idx != EMPTY_LABEL and conf >= confidence_threshold:
                    final_grid[r, c] = pred_idx # Assign the predicted digit (1-9)
                # Otherwise, leave it as 0 (empty/uncertain)

        return final_grid, confidences


# ------------------------------------------------------------------ #
# 8.  epoch‑end callback (adapted for full grid)
# ------------------------------------------------------------------ #
class EpochTestCallback(callbacks.Callback):
    def __init__(
        self,
        test_img_path: Path | str,
        gt_puzzle_grid: np.ndarray, # Ground truth puzzle (0 for empty)
        classifier: "DigitClassifier",
        frequency: int = 1,
    ) -> None:
        super().__init__()
        self.frequency = max(1, frequency)
        self.gt_puzzle_grid = gt_puzzle_grid # Shape (9, 9), 0 for empty
        self.classifier = classifier
        self.test_img_path = test_img_path
        self.preprocessed_test_grid = None # To store the single preprocessed grid

    def on_train_begin(self, logs=None):
        """Preprocess the test image once before training starts."""
        print("\n[Callback] Preprocessing test example for epoch-end evaluation...")
        try:
            # Extract the rectified grid from the test image file
            # We don't need the individual cells here, just the rectified image
            _, rectified_test_img, _ = extract_cells_from_image(
                self.test_img_path,
                size=self.classifier._model_input_shape[0], # Use model's input size
                debug=False
            )

            if rectified_test_img is None:
                print("[Callback Error] Failed to extract rectified grid from test image.")
                self.preprocessed_test_grid = None
                return

            # Preprocess the rectified grid using the classifier's method
            self.preprocessed_test_grid = self.classifier._preprocess_grid_for_model(rectified_test_img)

            if self.preprocessed_test_grid is None:
                print("[Callback Error] Preprocessing of the test grid failed.")
            else:
                print(f"[Callback] Test grid preprocessed successfully ({self.preprocessed_test_grid.shape}).")

        except Exception as e:
            print(f"[Callback Error] Failed during test example setup: {e}")
            self.preprocessed_test_grid = None

    def on_epoch_end(self, epoch, logs=None):
        # Check if preprocessing was successful and if it's the right epoch
        if self.preprocessed_test_grid is None or (epoch + 1) % self.frequency != 0:
            return

        if not hasattr(self.model, 'predict'):
             print("[Callback Error] Model object in callback does not have predict method.")
             return

        print(f"\n--- Epoch {epoch+1} Test Example Evaluation ---")
        try:
            # Add batch dimension for prediction
            x_test = self.preprocessed_test_grid[np.newaxis, ...]

            # Predict using the model being trained
            raw_output = self.model.predict(x_test, verbose=0) # Shape: (1, 9, 9, NUM_CLASSES)

            # Decode the output (similar to recognise_grid)
            probs = raw_output[0] # Remove batch dim -> (9, 9, NUM_CLASSES)
            pred_indices = np.argmax(probs, axis=-1) # (9, 9) indices 0-10
            confs = np.max(probs, axis=-1)       # (9, 9) confidences

            # Apply thresholding for display (use a slightly lower threshold maybe)
            display_threshold = 0.7 # Threshold for visualization purposes
            display_grid = np.zeros_like(pred_indices, dtype=int)
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    idx = pred_indices[r, c]
                    cf = confs[r, c]
                    if idx != EMPTY_LABEL and cf >= display_threshold:
                        display_grid[r, c] = idx

            print("Ground Truth Puzzle:")
            # print_sudoku_grid expects 0 for empty
            sudoku_recogniser.print_sudoku_grid(self.gt_puzzle_grid)

            print(f"Prediction (Thresholded @ {display_threshold:.2f}):")
            sudoku_recogniser.print_sudoku_grid(display_grid, confs, threshold=display_threshold)

            # --- Calculate Accuracy ---
            # Compare prediction indices directly with the GT grid converted to use EMPTY_LABEL
            gt_labels = self.gt_puzzle_grid.copy()
            gt_labels[gt_labels == 0] = EMPTY_LABEL # Convert GT to use 10 for empty

            correct_cells = (pred_indices == gt_labels).sum()
            total_cells = GRID_SIZE * GRID_SIZE
            accuracy = correct_cells / total_cells
            print(f"Test Example Accuracy (Raw Prediction vs GT Labels): {correct_cells}/{total_cells} = {accuracy:.4f}")
            print("--- End Epoch Test ---\n")

        except Exception as e:
            print(f"[Callback Error] Failed during prediction or display: {e}")
            import traceback
            traceback.print_exc()


# ------------------------------------------------------------------ #
# 9.  CLI helper
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    # Set to True to force retraining even if a model file exists
    FORCE_TRAIN = False

    model_file = Path(MODEL_FILENAME)
    train_needed = FORCE_TRAIN or not model_file.exists()

    if FORCE_TRAIN and model_file.exists():
        print(f"FORCE_TRAIN is True. Deleting existing model: {model_file}")
        try:
            model_file.unlink()
            train_needed = True
        except OSError as e:
            print(f"Error deleting existing model: {e}. Proceeding might use old model if loading works.")
            # Decide whether to exit or continue
            # exit(1)

    # Instantiate the classifier. It will try to load if train_needed is False.
    clf = DigitClassifier(model_path=model_file, training_required=train_needed)

    # Train if needed
    if train_needed:
        if clf.model is not None:
             print("[Warning] Model was loaded despite train_needed=True? Retraining anyway.")
             clf.model = None # Ensure model is rebuilt
        print("Starting training process...")
        clf.train(epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, batch_size=BATCH_SIZE)
        # After training, the best model should be saved and reloaded by the train method.
        if clf.model is None:
             print("[Error] Training finished, but model is still None. Cannot proceed.")
             exit(1)
    elif clf.model is None:
         print("[Error] Model loading failed, and training was not requested. Cannot proceed.")
         exit(1)
    else:
         print("Model loaded successfully. Skipping training.")


    # --- Perform Sanity Check using the Test Example ---
    if clf.model:
        print("\n--- Sanity Check: Recognising the Test Example ---")
        test_img_path_str, test_gt_grid = generate_and_save_test_example() # Get path and GT

        if test_img_path_str is None or test_gt_grid is None:
            print("Failed to load test example for sanity check.")
        else:
            test_img_path = Path(test_img_path_str)
            print(f"Loading test image: {test_img_path}")
            # Extract rectified grid from the test image
            _, rectified_test, _ = extract_cells_from_image(
                test_img_path,
                size=clf._model_input_shape[0], # Use model's input size
                debug=False
            )

            if rectified_test is None:
                print("Failed to extract rectified grid from test image for sanity check.")
            else:
                print("Running recognise_grid on the test image...")
                # Use a reasonable confidence threshold for the check
                pred_grid, conf_grid = clf.recognise_grid(rectified_test, confidence_threshold=0.75)

                if pred_grid is None:
                    print("Recognition failed during sanity check.")
                else:
                    print("\nGround Truth Puzzle:")
                    sudoku_recogniser.print_sudoku_grid(test_gt_grid)

                    print("Recognised Grid (Thresholded @ 0.75):")
                    sudoku_recogniser.print_sudoku_grid(pred_grid, conf_grid, threshold=0.75)

                    # Calculate accuracy for the sanity check
                    correct_cells = (pred_grid == test_gt_grid).sum()
                    # Account for empty cells being 0 in both GT and prediction (correctly)
                    # Non-empty cells must match exactly
                    correct_non_empty = ((pred_grid == test_gt_grid) & (test_gt_grid != 0)).sum()
                    correct_empty = ((pred_grid == 0) & (test_gt_grid == 0)).sum()
                    total_correct = correct_non_empty + correct_empty

                    total_cells = GRID_SIZE * GRID_SIZE
                    accuracy = total_correct / total_cells
                    print(f"\nSanity Check Accuracy: {total_correct}/{total_cells} = {accuracy:.4f}")
    else:
        print("\nSanity check skipped: No model available.")

    print("\nScript finished.")
