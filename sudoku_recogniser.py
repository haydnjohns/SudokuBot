# sudoku_recogniser.py
import cv2
import numpy as np
import sys
from pathlib import Path
import time

from digit_extractor import extract_cells_from_image, GRID_SIZE
# --- Use v6 Model Filename ---
from digit_classifier import DigitClassifier, MODEL_FILENAME

# --- Constants ---
# Confidence threshold for accepting a digit prediction during final recognition
FINAL_CONFIDENCE_THRESHOLD = 0.80 # Keep or adjust as needed

def recognise_sudoku(image_path, classifier, debug_cells=False):
    """
    Takes an image path, extracts cells, and uses the classifier to predict digits.

    Args:
        image_path (str | Path): Path to the Sudoku image.
        classifier (DigitClassifier): An initialized DigitClassifier instance.
        debug_cells (bool): If True, save raw and preprocessed cells.

    Returns:
        tuple: (predicted_grid, confidence_values, rectified_image)
            - predicted_grid (np.ndarray | None): 9x9 NumPy array of recognised digits (0 for empty/unknown).
            - confidence_values (np.ndarray | None): 9x9 NumPy array of confidence scores (0.0-1.0).
            - rectified_image (np.ndarray | None): The perspective-corrected grid image.
    """
    print(f"\nProcessing image: {image_path}")
    start_time = time.time()

    cells, rectified_grid, grid_contour = extract_cells_from_image(image_path, debug=False)

    if cells is None:
        print("Failed to extract Sudoku grid or cells.")
        return None, None, None

    if len(cells) != GRID_SIZE * GRID_SIZE:
        print(f"Error: Expected {GRID_SIZE*GRID_SIZE} cells, got {len(cells)}.")
        return None, None, None

    print(f"Grid extraction successful ({time.time() - start_time:.2f}s)")
    extract_time = time.time()

    cell_debug_dir = None
    if debug_cells:
        cell_debug_dir = Path(f"debug_recogniser_cells_{Path(image_path).stem}")
        cell_debug_dir.mkdir(exist_ok=True)
        print(f"Debugging cells to: {cell_debug_dir}")
        if rectified_grid is not None:
             cv2.imwrite(str(cell_debug_dir / "_rectified_grid.png"), rectified_grid)

    predicted_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    confidence_values = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)

    if classifier.model is None:
         print("[ERROR] Classifier model is not loaded in recognise_sudoku. Cannot proceed.")
         return None, None, rectified_grid

    for i, cell_img in enumerate(cells):
        row, col = divmod(i, GRID_SIZE)

        if cell_debug_dir and cell_img is not None:
             cv2.imwrite(str(cell_debug_dir / f"cell_{row}_{col}_0_raw.png"), cell_img)

        if cell_img is None or cell_img.size < 10:
            predicted_grid[row, col] = 0
            confidence_values[row, col] = 1.0 # Confident it's empty based on input
            if cell_debug_dir:
                 empty_dbg = np.zeros(classifier._model_input_size, dtype=np.uint8)
                 cv2.imwrite(str(cell_debug_dir / f"cell_{row}_{col}_1_preprocessed_EMPTYINPUT.png"), empty_dbg)
            continue

        # Use low internal threshold to get prediction/confidence regardless
        digit, confidence = classifier.recognise(cell_img, confidence_threshold=0.1)
        confidence_values[row, col] = confidence

        # Apply final threshold for the output grid
        if digit != 0 and confidence >= FINAL_CONFIDENCE_THRESHOLD:
             predicted_grid[row, col] = digit
        else:
             predicted_grid[row, col] = 0 # Mark as unknown/empty

        if cell_debug_dir:
            processed_for_debug = classifier._preprocess_cell_for_model(cell_img)
            if processed_for_debug is not None:
                 processed_img_uint8 = (processed_for_debug * 255).astype(np.uint8)
                 pred_suffix = f"_pred{digit}_conf{confidence:.2f}"
                 if predicted_grid[row, col] == 0 and digit != 0: pred_suffix += "_REJECTED"
                 elif digit == 0: pred_suffix += "_EMPTY"
                 cv2.imwrite(str(cell_debug_dir / f"cell_{row}_{col}_1_preprocessed{pred_suffix}.png"), processed_img_uint8)
            else:
                 failed_dbg = np.full(classifier._model_input_size, 128, dtype=np.uint8)
                 cv2.imwrite(str(cell_debug_dir / f"cell_{row}_{col}_1_preprocessed_FAILED.png"), failed_dbg)

    print(f"Digit recognition complete ({time.time() - extract_time:.2f}s)")
    print(f"Total processing time: {time.time() - start_time:.2f}s")

    return predicted_grid, confidence_values, rectified_grid


def print_sudoku_grid(grid, confidence_values=None, threshold=FINAL_CONFIDENCE_THRESHOLD):
    """Prints the Sudoku grid. Shows digits above threshold, '.' otherwise."""
    if grid is None:
        print("No grid to print.")
        return
    grid = np.array(grid) # Ensure numpy array
    if grid.shape != (GRID_SIZE, GRID_SIZE):
         print(f"Invalid grid shape for printing: {grid.shape}")
         return

    # Check if confidence values are usable for marking uncertainty (optional)
    show_uncertainty = isinstance(confidence_values, np.ndarray) and confidence_values.shape == (GRID_SIZE, GRID_SIZE)

    # Use threshold=1.1 when printing GT to avoid any '?' marks
    print_threshold = threshold if threshold <= 1.0 else FINAL_CONFIDENCE_THRESHOLD
    print(f"\nDetected Sudoku Grid (Threshold: {print_threshold:.2f}):")
    print("-" * 25)
    for r in range(GRID_SIZE):
        if r % 3 == 0 and r != 0: print("|-------+-------+-------|")
        row_str = "| "
        for c in range(GRID_SIZE):
            digit = grid[r, c]
            display_char = str(digit) if digit != 0 else "."
            uncertain_marker = ""

            # Optional: Mark low-confidence *empty* cells if desired
            # if show_uncertainty and digit == 0 and confidence_values[r, c] < 0.5:
            #      uncertain_marker = "?"

            row_str += f"{display_char}{uncertain_marker} "
            if (c + 1) % 3 == 0: row_str += "| "
        print(row_str)
    print("-" * 25)


def main():
    if len(sys.argv) < 2:
        print("Usage: python sudoku_recogniser.py <path_to_sudoku_image> [--debug-cells] [--force-train]")
        # Try various default images
        default_paths = ["epoch_test_sudoku.png", "rendered_sudoku_specific.png", "rendered_sudoku_random.png"]
        test_image = None
        for p in default_paths:
             if Path(p).exists():
                 test_image = Path(p)
                 break

        if test_image:
             print(f"\nNo image provided. Running on default test image: {test_image}")
             image_path = test_image
        else:
             print("No image provided and default test images not found. Please provide an image path.")
             sys.exit(1)
    else:
        image_path = Path(sys.argv[1])

    debug_cells_flag = "--debug-cells" in sys.argv
    force_train_flag = "--force-train" in sys.argv

    if not image_path.exists():
        print(f"Error: Image file not found at '{image_path}'")
        sys.exit(1)

    print(f"Initializing Digit Classifier (Model: {MODEL_FILENAME})...")
    model_p = Path(MODEL_FILENAME) # Use v6 filename

    classifier = DigitClassifier(training_required=force_train_flag)

    if classifier.model is None:
        print(f"\nClassifier model ('{model_p.name}') needs training.")
        print("Training will start (includes epoch-end example evaluation)...")
        try:
            classifier.train() # This now runs the callback internally
            if classifier.model is None:
                 print("\n[Error] Classifier training failed. Exiting.")
                 sys.exit(1)
            print("\nTraining complete. Proceeding with recognition.")
        except Exception as e:
            print(f"\n[Error] An exception occurred during training: {e}")
            import traceback; traceback.print_exc(); sys.exit(1)
    else:
        print("Classifier model loaded successfully.")

    predicted_grid, confidence_values, rectified_image = recognise_sudoku(
        image_path, classifier, debug_cells=debug_cells_flag
    )

    print_sudoku_grid(predicted_grid, confidence_values, FINAL_CONFIDENCE_THRESHOLD)

    if rectified_image is not None:
        try:
            display_img = rectified_image.copy()
            if display_img.ndim == 2: display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
            cell_h, cell_w = display_img.shape[0] // GRID_SIZE, display_img.shape[1] // GRID_SIZE
            if predicted_grid is not None:
                for r in range(GRID_SIZE):
                    for c in range(GRID_SIZE):
                        digit = predicted_grid[r, c]
                        if digit != 0:
                            text = str(digit)
                            center_x, center_y = c * cell_w + cell_w // 2, r * cell_h + cell_h // 2
                            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                            text_x, text_y = center_x - text_size[0] // 2, center_y + text_size[1] // 2
                            # Draw slightly offset shadow first for better visibility
                            cv2.putText(display_img, text, (text_x + 1, text_y + 1), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2) # Black shadow
                            cv2.putText(display_img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2) # Green text

            cv2.imshow("Rectified Sudoku Grid with Predictions", display_img)
            print("\nDisplaying rectified grid with predictions. Press any key to close.")
            cv2.waitKey(0); cv2.destroyAllWindows()
        except Exception as e:
            print(f"\nCould not display the image (maybe no GUI available?): {e}")

if __name__ == "__main__":
    main()