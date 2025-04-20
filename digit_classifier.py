# digit_classifier.py
# (Imports and constants remain the same, except maybe adjust TARGET_CELL_CONTENT_SIZE if needed)
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

# Assuming these imports exist and are correct
try:
    from sudoku_renderer import SudokuRenderer, generate_and_save_test_example
    from digit_extractor import GRID_SIZE, extract_cells_from_image, rectify_grid, split_into_cells
    from sudoku_recogniser import FINAL_CONFIDENCE_THRESHOLD, print_sudoku_grid
except ImportError as e:
    print(f"Error importing helper modules: {e}")
    print("Please ensure sudoku_renderer.py, digit_extractor.py, and sudoku_recogniser.py are available.")
    exit(1)


MODEL_FILENAME = "sudoku_digit_classifier_cnn.keras"
MODEL_INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 11        # digits 0–9 plus one “empty” class
EMPTY_LABEL = 10
# TARGET_CELL_CONTENT_SIZE = 20 # Reduced slightly to give more padding room in 28x28
TARGET_CELL_CONTENT_SIZE = 22 # Let's try slightly larger than 20
TARGET_DIGIT_RATIO = 1.5 # Ratio of digit cells to empty cells in a batch

EPOCHS = 15 # Increased slightly as convergence might be slower but more stable
STEPS_PER_EPOCH = 150 # Increased slightly
BATCH_SIZE = 256
VALIDATION_STEPS = 50

DataBatch = Tuple[np.ndarray, np.ndarray]


def sudoku_data_generator(
    renderer: SudokuRenderer, # Accept renderer instance
    batch_size: int,
    preprocess_func: Callable[[np.ndarray], Optional[np.ndarray]],
    input_size: Tuple[int, int, int],
    target_digit_ratio: float = TARGET_DIGIT_RATIO,
) -> Generator[DataBatch, None, None]:
    """
    Yield balanced batches of (cell_image, label) for training.
    Uses a shared SudokuRenderer instance.
    """
    total_cells = GRID_SIZE * GRID_SIZE
    # Calculate target number of digits and empty cells per batch
    target_digits = int(batch_size * target_digit_ratio / (1 + target_digit_ratio))
    target_empty = batch_size - target_digits
    input_h, input_w = input_size[:2]

    while True:
        x_list, y_list = [], []
        n_digits = n_empty = 0
        attempts = 0
        # Increased max_attempts slightly, robust preprocessing might reject more cells
        max_attempts = batch_size * 5

        while len(x_list) < batch_size and attempts < max_attempts:
            attempts += 1
            # Generate a Sudoku image using the provided renderer
            # Allow empty cells frequently, but ensure some digits exist
            allow_empty = (random.random() < 0.85)
            img, gt_grid, corners = renderer.render_sudoku(allow_empty=allow_empty)

            # Basic checks for valid generation
            if img is None or corners is None or gt_grid is None:
                # print("[Generator] Render failed, skipping.")
                continue
            if np.sum(gt_grid > 0) == 0 and not allow_empty: # Ensure some digits if allow_empty=False
                 # print("[Generator] No digits rendered when expected, skipping.")
                 continue

            # Rectify and split into cells
            rectified = rectify_grid(img, corners)
            if rectified is None:
                # print("[Generator] Rectify failed, skipping.")
                continue

            cells, _ = split_into_cells(rectified)
            if len(cells) != total_cells:
                # print(f"[Generator] Split failed ({len(cells)} cells), skipping.")
                continue

            gt_flat = gt_grid.flatten()
            indices = list(range(total_cells))
            random.shuffle(indices) # Process cells in random order

            # Iterate through cells to fill the batch
            for idx in indices:
                cell = cells[idx]
                label = EMPTY_LABEL if gt_flat[idx] == 0 else gt_flat[idx]
                is_empty = (label == EMPTY_LABEL)

                # Check if we need more of this type (digit or empty)
                if is_empty and n_empty >= target_empty:
                    continue
                if not is_empty and n_digits >= target_digits:
                    continue

                # Preprocess the cell
                processed = preprocess_func(cell)
                # Ensure preprocessing was successful and output has correct shape
                if processed is None or processed.shape != (input_h, input_w):
                    # print(f"[Generator] Preprocessing failed or wrong shape for cell {idx}, skipping.")
                    continue

                # Add processed cell and label to the batch lists
                x_list.append(processed)
                y_list.append(label)
                if is_empty:
                    n_empty += 1
                else:
                    n_digits += 1

                # Stop if the batch is full
                if len(x_list) >= batch_size:
                    break

        # If the batch couldn't be filled sufficiently, generate a new Sudoku
        if len(x_list) < batch_size // 2: # Require at least half a batch
            # print(f"[Generator] Batch too small ({len(x_list)}/{batch_size}), retrying.")
            del x_list, y_list # Clean up before next attempt
            gc.collect()
            continue

        # Convert lists to numpy arrays, shuffle, and yield
        x_arr = np.array(x_list, dtype="float32")[..., np.newaxis]
        y_arr = np.array(y_list, dtype="int64")
        perm = np.random.permutation(len(y_arr))
        yield x_arr[perm], y_arr[perm]

        # Clean up memory
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
        self.test_img_path = test_img_path # Store path for potential re-extraction
        self.preprocessed = self._prepare_test_data()

    def _prepare_test_data(self) -> Optional[np.ndarray]:
        """Extracts and preprocesses cells from the test image."""
        try:
            # Use the main extraction function
            cells, _, _ = extract_cells_from_image(self.test_img_path, debug=False)
            if not cells or len(cells) != GRID_SIZE * GRID_SIZE:
                print("[Callback] Test image cell extraction failed.")
                return None

            processed = []
            for i, cell in enumerate(cells):
                # Use the classifier's preprocessing method
                proc = self.classifier._preprocess_cell_for_model(cell)
                if proc is None:
                    # If preprocessing fails for a cell, use zeros - model should predict 'empty'
                    print(f"[Callback] Preprocessing failed for test cell {i}, using zeros.")
                    proc = np.zeros(self.classifier._model_input_size, dtype="float32")
                processed.append(proc)

            # Add channel dimension for the model
            return np.array(processed, dtype="float32")[..., np.newaxis]

        except Exception as e:
            print(f"[Callback] Error preparing test data: {e}")
            return None

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        if self.preprocessed is None:
            # Attempt to prepare data again if it failed initially
            print("[Callback] Retrying test data preparation...")
            self.preprocessed = self._prepare_test_data()
            if self.preprocessed is None:
                print("[Callback] Test data preparation failed again, callback inactive.")
                return # Still failed, do nothing

        # Check frequency
        if (epoch + 1) % self.frequency != 0:
            return

        print(f"\n--- Epoch {epoch+1} Test Example ({self.test_img_path.name}) ---")
        try:
            # Get predictions
            preds = self.model.predict(self.preprocessed, verbose=0)
            # Find best class index and confidence for each cell
            idxs = np.argmax(preds, axis=1)
            confs = np.max(preds, axis=1)

            final_pred = []
            for idx, conf in zip(idxs, confs):
                # Assign digit only if not empty class and confidence is high
                digit = idx if (idx != EMPTY_LABEL and conf >= FINAL_CONFIDENCE_THRESHOLD) else 0
                final_pred.append(digit)

            # Reshape predictions and confidences into grids
            pred_grid = np.array(final_pred).reshape(GRID_SIZE, GRID_SIZE)
            conf_grid = confs.reshape(GRID_SIZE, GRID_SIZE)

            # Print ground truth and prediction
            print("Ground truth:")
            print_sudoku_grid(self.gt_grid, threshold=1.1) # Show all GT digits
            print("Prediction:")
            print_sudoku_grid(pred_grid, conf_grid, threshold=FINAL_CONFIDENCE_THRESHOLD)

            # Calculate and print accuracy
            correct = np.sum(pred_grid == self.gt_grid)
            total = GRID_SIZE * GRID_SIZE
            accuracy = correct / total
            print(f"Accuracy: {correct}/{total} ({accuracy:.4f})")

        except Exception as e:
            print(f"[Callback] Error during prediction/evaluation: {e}")
        finally:
            print("--- End Test Example ---\n")


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
        self._model_input_size = MODEL_INPUT_SHAPE[:2] # (28, 28)

        if not training_required and self.model_path.exists():
            try:
                # Load the model using Keras API
                self.model = keras.models.load_model(self.model_path)
                # Verify input shape compatibility
                loaded_shape = self.model.input_shape[1:3] # Exclude batch and channel dims
                if loaded_shape != self._model_input_size:
                    print(
                        f"[Warning] Loaded model input shape {loaded_shape} "
                        f"differs from expected {self._model_input_size}. Mismatches may occur."
                    )
                print(f"Model loaded successfully from {self.model_path}")
            except Exception as exc:
                print(f"[Error] Could not load model from {self.model_path}: {exc}")
                print("Will proceed assuming training is needed or a new model will be built.")
                self.model = None # Ensure model is None if loading failed

    def _build_cnn_model(self) -> keras.Model:
        """Builds the SimpleNet-like CNN model."""
        # Configuration for convolutional layers (filters) and pooling layers
        cfg_filters   = [32, 32,     # block‑1
                         64, 64,     # block‑2
                         96, 96, 96, # block‑3
                         128,128,128,128,   # block‑4
                         192,192]    # block‑5
        # Indices *after* which max pooling is applied
        pool_after_idx = {1, 3, 6, 10} # Corresponds to end of block 1, 2, 3, 4

        inputs = keras.Input(shape=MODEL_INPUT_SHAPE)
        x = inputs

        # Build convolutional blocks
        for i, f in enumerate(cfg_filters):
            x = layers.Conv2D(f, kernel_size=3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            # Apply max pooling after specified layers
            if i in pool_after_idx:
                x = layers.MaxPooling2D(pool_size=2)(x)

        # Optional 1x1 bottleneck layer before global pooling
        x = layers.Conv2D(256, kernel_size=1, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        # Global pooling and dense layers for classification
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.3)(x) # Added dropout for regularization
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.3)(x) # Added dropout for regularization
        outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

        model = keras.Model(inputs, outputs, name="simplenet_digits_v2")

        # Compile the model with optimizer, loss function, and metrics
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=5e-4), # Slightly lower LR
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.summary() # Print model architecture and parameter count
        return model

    def _preprocess_cell_for_model(self, cell: np.ndarray) -> Optional[np.ndarray]:
        """
        Robustly preprocess a single cell image for the CNN model.
        Binarize, clean noise, find largest contour (digit), crop, resize, center, normalize.
        Returns a 2D float32 array of shape `_model_input_size`, or None on failure.
        """
        if cell is None or cell.size < 10: # Basic check for empty/tiny input
            return None

        # Ensure input is grayscale
        gray = (
            cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            if cell.ndim == 3 and cell.shape[2] == 3
            else cell.copy()
        )
        if gray.ndim != 2: # Handle cases like RGBA or unexpected shapes
             return None

        # Apply Gaussian blur to reduce noise before thresholding
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive thresholding (more robust to lighting variations)
        # Use a fixed block size and adjust constant C
        block_size = 15 # Must be odd
        C = 5         # Constant subtracted from the mean
        try:
            thresh = cv2.adaptiveThreshold(
                blurred, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, # Invert: digit is white, background black
                block_size, C
            )
        except cv2.error as e:
            # print(f"[Preprocess] AdaptiveThreshold failed: {e}")
            return None # Thresholding failed

        # Morphological Opening: Erode then Dilate to remove small noise spots
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find contours on the cleaned thresholded image
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # No contours found, likely an empty cell or preprocessing failed
            return None # Treat as empty/unidentifiable

        # Find the contour with the largest area (presumably the digit)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        # Filter out tiny contours that are likely noise remnants
        min_area_threshold = 5 # Adjust based on expected digit size relative to cell
        if area < min_area_threshold:
            return None # Contour too small

        # Get the bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Extract the Region of Interest (ROI) using the bounding box
        # Add a small padding around the bounding box to avoid cutting off edges
        padding = 2
        y1, y2 = max(0, y - padding), min(cleaned.shape[0], y + h + padding)
        x1, x2 = max(0, x - padding), min(cleaned.shape[1], x + w + padding)
        roi = cleaned[y1:y2, x1:x2]

        # Check if ROI is valid after padding/cropping
        if roi.size == 0 or w <= 0 or h <= 0:
             return None

        # Resize the ROI to fit within the target content size, maintaining aspect ratio
        current_h, current_w = roi.shape
        scale = min(TARGET_CELL_CONTENT_SIZE / current_w, TARGET_CELL_CONTENT_SIZE / current_h)
        new_w = max(1, int(current_w * scale))
        new_h = max(1, int(current_h * scale))

        try:
            resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except cv2.error as e:
            # print(f"[Preprocess] Resize failed: {e}")
            return None

        # Create a black canvas of the final model input size
        canvas_h, canvas_w = self._model_input_size
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

        # Calculate top-left corner position to center the resized digit on the canvas
        top = max(0, (canvas_h - new_h) // 2)
        left = max(0, (canvas_w - new_w) // 2)

        # Place the resized digit onto the canvas
        # Ensure slicing does not go out of bounds
        h_slice = slice(top, min(top + new_h, canvas_h))
        w_slice = slice(left, min(left + new_w, canvas_w))
        # Ensure the shape matches for assignment
        canvas_roi_h = h_slice.stop - h_slice.start
        canvas_roi_w = w_slice.stop - w_slice.start
        canvas[h_slice, w_slice] = resized[:canvas_roi_h, :canvas_roi_w]

        # Normalize the canvas to [0, 1] float32 for the model
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
            f"\nStarting training with parameters: "
            f"epochs={epochs}, steps_per_epoch={steps_per_epoch}, "
            f"batch_size={batch_size}, validation_steps={validation_steps}"
        )

        # --- Create ONE shared renderer instance ---
        shared_renderer = SudokuRenderer()
        print("Shared SudokuRenderer instance created.")

        # --- Setup Epoch Test Callback ---
        epoch_test_cb = None # Initialize
        try:
            test_img_path_str, test_gt = generate_and_save_test_example()
            test_img_path = Path(test_img_path_str)
            if test_img_path.exists():
                 # Pass self (classifier instance) to the callback
                epoch_test_cb = EpochTestCallback(test_img_path, test_gt, self, frequency=1)
                print("EpochTestCallback setup with fixed test example.")
            else:
                print("[Warning] Test example image not found, EpochTestCallback disabled.")
        except Exception as exc:
            print(f"[Warning] Could not set up EpochTestCallback: {exc}")

        # --- Create Data Generators using the shared renderer ---
        print("Creating data generators...")
        train_gen = sudoku_data_generator(
            shared_renderer, # Pass shared instance
            batch_size,
            self._preprocess_cell_for_model,
            MODEL_INPUT_SHAPE,
        )
        val_gen = sudoku_data_generator(
            shared_renderer, # Pass shared instance
            batch_size,
            self._preprocess_cell_for_model,
            MODEL_INPUT_SHAPE,
        )
        print("Data generators created.")

        # --- Build or load model ---
        if self.model is None:
            print("Building new CNN model...")
            self.model = self._build_cnn_model()
        else:
            print("Continuing training with existing model.")

        # --- Define Callbacks ---
        cbs = [
            callbacks.EarlyStopping(
                monitor="val_loss", patience=6, # Increased patience slightly
                restore_best_weights=True, verbose=1,
                start_from_epoch=3 # Start monitoring after initial epochs
            ),
            callbacks.ModelCheckpoint(
                filepath=self.model_path,
                monitor="val_loss", save_best_only=True, verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.3, # Less aggressive reduction
                patience=3, min_lr=1e-6, verbose=1
            ),
            # Add TensorBoard callback for visualization if desired
            # callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)
        ]
        # Add the epoch test callback only if it was successfully created
        if epoch_test_cb and epoch_test_cb.preprocessed is not None:
            cbs.append(epoch_test_cb)
        elif epoch_test_cb:
             print("[Warning] EpochTestCallback created but data preparation failed, callback not added.")


        # --- Start Training ---
        print("Starting model training...")
        history = self.model.fit(
            train_gen,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=cbs,
            verbose=1, # Use 1 for progress bar, 2 for one line per epoch
        )

        # --- Final Evaluation ---
        # Reload best weights if EarlyStopping restored them
        if any(isinstance(cb, callbacks.EarlyStopping) and cb.restore_best_weights for cb in cbs):
             print("Reloading best weights found during training for final evaluation...")
             # Keras with TF backend automatically restores, check if needed for Torch backend explicitly
             # For Keras 3 with Torch backend, load_model might be needed if restore_best_weights isn't automatic
             try:
                 best_model = keras.models.load_model(self.model_path)
                 self.model = best_model
                 print("Best weights reloaded.")
             except Exception as e:
                 print(f"[Warning] Could not reload best weights from {self.model_path}: {e}")


        print("\nPerforming final evaluation on validation data...")
        # Create a fresh validation generator for final evaluation
        final_val_gen = sudoku_data_generator(
            shared_renderer, # Use the same renderer
            batch_size,
            self._preprocess_cell_for_model,
            MODEL_INPUT_SHAPE,
        )
        # Evaluate the model
        loss, acc = self.model.evaluate(
            final_val_gen,
            steps=validation_steps * 2, # Evaluate on more steps for stability
            verbose=1,
        )
        print(f"Final validation loss = {loss:.4f}, accuracy = {acc:.4f}")

        # --- Save Final Model ---
        # The ModelCheckpoint callback already saved the best model during training.
        # We can optionally save the *final* state (after potential further epochs or if early stopping wasn't triggered)
        final_model_path = self.model_path.with_name(f"{self.model_path.stem}_final{self.model_path.suffix}")
        print(f"Saving final model state to {final_model_path}")
        self.model.save(final_model_path)
        print(f"Best model during training saved at {self.model_path}")


        # --- Cleanup ---
        del train_gen, val_gen, final_val_gen, shared_renderer
        gc.collect()
        print("Training finished.")


    @torch.no_grad() # Essential for inference with PyTorch backend
    def recognise(
        self,
        cell: np.ndarray,
        confidence_threshold: float = 0.7, # Default threshold for accepting a digit
    ) -> Tuple[int, float]:
        """
        Predict the digit in a single cell image using the trained model.
        Returns (predicted_digit, confidence_score).
        predicted_digit = 0 indicates empty cell or low confidence.
        """
        if self.model is None:
            print("[Error] Recognise called but model is not loaded.")
            return 0, 0.0

        # Preprocess the input cell image
        proc_cell = self._preprocess_cell_for_model(cell)
        if proc_cell is None:
            # Preprocessing failed or determined cell is empty/invalid
            return 0, 0.0 # Return 0 (empty) with 0 confidence

        # Prepare the input tensor for the model
        # Add batch and channel dimensions: (1, H, W, 1)
        x = proc_cell[np.newaxis, ..., np.newaxis]
        # Convert to PyTorch tensor if using Torch backend
        # Keras 3 should handle this conversion automatically, but explicit is safer
        try:
            # Keras 3 with torch backend expects torch tensor
            x_tensor = torch.from_numpy(x).float()
            # If using GPU, move tensor to the correct device (optional, assumes model is on CPU if not specified)
            # if torch.cuda.is_available():
            #     x_tensor = x_tensor.cuda()
        except AttributeError:
             # Fallback or if backend isn't torch, Keras might handle numpy directly
             x_tensor = x # Use numpy array directly


        # Perform inference
        # Use `training=False` for layers like BatchNormalization and Dropout
        predictions = self.model(x_tensor, training=False)

        # Process the output probabilities
        # Ensure predictions are on CPU and converted to NumPy array
        if isinstance(predictions, torch.Tensor):
            probs = predictions.cpu().numpy()[0] # Get probabilities for the single image
        elif isinstance(predictions, np.ndarray):
             probs = predictions[0]
        else:
             # Handle unexpected output type from model prediction
             print(f"[Error] Unexpected prediction type: {type(predictions)}")
             return 0, 0.0


        # Find the class index with the highest probability
        predicted_index = int(np.argmax(probs))
        # Get the confidence score for the predicted class
        confidence = float(probs[predicted_index])

        # Determine the final digit
        # If the predicted class is the 'empty' label or confidence is below threshold, return 0
        if predicted_index == EMPTY_LABEL or confidence < confidence_threshold:
            # Return 0 for empty/uncertain, but still return the actual confidence
            # of the most likely class (which might be EMPTY_LABEL or a low-conf digit)
            return 0, confidence
        else:
            # Return the recognised digit (1-9) and its confidence
            return predicted_index, confidence


if __name__ == "__main__":
    # --- Configuration ---
    FORCE_TRAIN = False # Set to True to force retraining even if model file exists
    MODEL_PATH = Path(MODEL_FILENAME)

    # --- Force Training Option ---
    if FORCE_TRAIN and MODEL_PATH.exists():
        print(f"FORCE_TRAIN is True. Deleting existing model file: {MODEL_PATH}")
        try:
            MODEL_PATH.unlink()
        except OSError as e:
            print(f"Error deleting model file: {e}")

    # --- Initialize Classifier ---
    # Pass training_required=FORCE_TRAIN to potentially skip loading if forcing train
    print("Initializing DigitClassifier...")
    classifier = DigitClassifier(model_path=MODEL_PATH, training_required=FORCE_TRAIN)

    # --- Train if necessary ---
    if classifier.model is None:
        print("Model not loaded or not found. Starting training process...")
        try:
            classifier.train()
            # After training, the best model should be loaded or available
            if classifier.model is None:
                 print("[Error] Training finished but model is still None. Exiting.")
                 exit(1)
            print("Training complete. Model is ready.")
        except Exception as e:
            print(f"\n[Error] An exception occurred during training: {e}")
            import traceback
            traceback.print_exc()
            exit(1)
    else:
        print("Model loaded successfully. Skipping training.")

    # --- Quick Test ---
    if classifier.model:
        print("\n--- Running Quick Dummy Test ---")
        try:
            # Test 1: Image with a vertical line (should ideally be empty or low conf)
            dummy_line = np.zeros((50, 50), dtype=np.uint8)
            cv2.line(dummy_line, (25, 10), (25, 40), 255, 3) # Draw a white line
            digit1, conf1 = classifier.recognise(dummy_line, confidence_threshold=0.5)
            print(f"Prediction (vertical line): Digit={digit1}, Confidence={conf1:.3f}")
            # Expected: Digit=0 (empty or low confidence)

            # Test 2: Blank image (should be empty)
            dummy_empty = np.zeros((50, 50), dtype=np.uint8)
            digit2, conf2 = classifier.recognise(dummy_empty, confidence_threshold=0.1) # Low threshold to see raw prediction
            print(f"Prediction (blank image):   Digit={digit2}, Confidence={conf2:.3f}")
            # Expected: Digit=0 (empty), Confidence might be high for EMPTY_LABEL

            # Test 3: Simple digit '1' drawn (should be recognised)
            dummy_one = np.zeros((50, 50), dtype=np.uint8)
            cv2.putText(dummy_one, '1', (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 3)
            digit3, conf3 = classifier.recognise(dummy_one, confidence_threshold=0.5)
            print(f"Prediction (drawn '1'):     Digit={digit3}, Confidence={conf3:.3f}")
            # Expected: Digit=1, Confidence > 0.5

        except Exception as e:
            print(f"[Error] Exception during dummy test: {e}")
        print("--- End Quick Dummy Test ---")
    else:
        print("Skipping dummy test as model is not available.")