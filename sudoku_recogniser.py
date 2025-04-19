# sudoku_recogniser.py
import cv2
import numpy as np
import sys
from pathlib import Path
import time

# Import necessary components from other modules
from digit_extractor import extract_cells_from_image
from digit_classifier import DigitClassifier, MODEL_FILENAME # Use new MODEL_FILENAME

# --- Constants ---
GRID_SIZE = 9
# Confidence threshold for accepting a digit prediction during final recognition
# Can be different from the internal threshold in classifier.recognise if needed
FINAL_CONFIDENCE_THRESHOLD = 0.75

def recognise_sudoku(image_path, classifier, debug_cells=False):
    """
    Takes an image path, extracts cells, and uses the classifier to predict digits.

    Args:
        image_path (str | Path): Path to the Sudoku image.
        classifier (DigitClassifier): An initialized DigitClassifier instance.
        debug_cells (bool): If True, save preprocessed cells fed to the classifier.

    Returns:
        tuple: (predicted_grid, confidence_values, rectified_image)
            - predicted_grid (np.ndarray | None): 9x9 NumPy array of recognised digits (0 for empty/unknown).
            - confidence_values (np.ndarray | None): 9x9 NumPy array of confidence scores (0.0-1.0).
            - rectified_image (np.ndarray | None): The perspective-corrected grid image.
    """
    print(f"\nProcessing image: {image_path}")
    start_time = time.time()

    # 1. Extract Cells (pass debug flag if extractor needs it)
    cells, rectified_grid, _ = extract_cells_from_image(image_path, debug=False) # Set debug=True for extractor debug images

    if cells is None:
        print("Failed to extract Sudoku grid or cells.")
        return None, None, None

    if len(cells) != GRID_SIZE * GRID_SIZE:
        print(f"Error: Expected {GRID_SIZE*GRID_SIZE} cells, got {len(cells)}.")
        return None, None, None

    print(f"Grid extraction successful ({time.time() - start_time:.2f}s)")
    extract_time = time.time()

    # --- Debug Cell Saving ---
    cell_debug_dir = None
    if debug_cells:
        cell_debug_dir = Path(f"debug_recogniser_cells_{Path(image_path).stem}")
        cell_debug_dir.mkdir(exist_ok=True)
        print(f"Debugging cells to: {cell_debug_dir}")
    # ---

    # 2. Recognise Digits in Cells
    predicted_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    confidence_values = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)

    for i, cell_img in enumerate(cells):
        row, col = divmod(i, GRID_SIZE)

        if cell_img is None or cell_img.size < 10:
            predicted_grid[row, col] = 0
            confidence_values[row, col] = 1.0 # Confident it's empty
            continue

        # --- Save raw cell before preprocessing for debug ---
        if cell_debug_dir:
             cv2.imwrite(str(cell_debug_dir / f"cell_{row}_{col}_raw.png"), cell_img)
        # ---

        # Use the classifier's recognise method - returns (digit, confidence)
        # Pass a slightly lower internal threshold to get confidence even for rejects
        digit, confidence = classifier.recognise(cell_img, confidence_threshold=0.5)

        confidence_values[row, col] = confidence

        # Apply the *final* threshold for the grid output
        if confidence >= FINAL_CONFIDENCE_THRESHOLD:
             predicted_grid[row, col] = digit # Keep digit if confidence is high enough
        else:
             predicted_grid[row, col] = 0 # Otherwise, mark as unknown (0)

        # --- Save preprocessed cell (optional, classifier can also do this) ---
        # if cell_debug_dir:
        #     processed_for_debug = classifier._preprocess_cell_for_model(cell_img)
        #     cv2.imwrite(str(cell_debug_dir / f"cell_{row}_{col}_preprocessed.png"), (processed_for_debug * 255).astype(np.uint8))
        # ---


    print(f"Digit recognition complete ({time.time() - extract_time:.2f}s)")
    print(f"Total processing time: {time.time() - start_time:.2f}s")

    return predicted_grid, confidence_values, rectified_grid


def print_sudoku_grid(grid, confidence_values=None, threshold=FINAL_CONFIDENCE_THRESHOLD):
    """Prints the Sudoku grid, marking uncertain digits."""
    if grid is None:
        print("No grid to print.")
        return

    print(f"\nDetected Sudoku Grid (0 = empty/unknown, ? = low confidence < {threshold:.2f}):")
    print("-" * 25)
    for r in range(GRID_SIZE):
        if r % 3 == 0 and r != 0: print("|-------+-------+-------|")
        row_str = "| "
        for c in range(GRID_SIZE):
            digit = grid[r, c]
            uncertain = ""
            # Mark if confidence exists and is below threshold (even for '0' if desired, but usually for digits)
            if confidence_values is not None and confidence_values[r, c] < threshold:
                 # Optionally show the low-confidence digit instead of '?'
                 # low_conf_digit = classifier.recognise(cell, threshold=0.01)[0] # Re-recognise? No, use grid value
                 # if digit != 0: # Only mark non-zero digits that fell below threshold
                 uncertain = "?"

            # Handle the case where grid[r,c] is 0 because confidence was low
            display_digit = str(digit) if digit != 0 else "." # Use '.' for empty/unknown

            row_str += f"{display_digit}{uncertain} "
            if (c + 1) % 3 == 0: row_str += "| "
        print(row_str)
    print("-" * 25)


def main():
    if len(sys.argv) < 2:
        print("Usage: python sudoku_recogniser.py <path_to_sudoku_image> [--debug-cells]")
        test_image = Path("rendered_sudoku_specific.png") # Or a real photo path
        if test_image.exists():
             print(f"\nNo image provided. Running on default test image: {test_image}")
             image_path = test_image
        else: sys.exit(1)
    else:
        image_path = Path(sys.argv[1])

    debug_cells_flag = "--debug-cells" in sys.argv

    if not image_path.exists():
        print(f"Error: Image file not found at '{image_path}'")
        sys.exit(1)

    print("Initializing Digit Classifier (v2)...")
    model_p = Path(MODEL_FILENAME) # Use new filename
    needs_training = not model_p.exists()
    if needs_training:
         print(f"Classifier model ('{model_p.name}') not found.")
         print("Training will start using synthetic data (may take minutes)...")
         # Add confirmation here if desired

    classifier = DigitClassifier() # Loads or prepares for training

    if classifier.model is None:
        print("\nStarting classifier training...")
        try:
            # Use reasonably robust parameters for default training
            classifier.train(epochs=10, steps_per_epoch=10, batch_size=128)
            if classifier.model is None:
                 print("\n[Error] Classifier training failed. Exiting.")
                 sys.exit(1)
            print("\nTraining complete. Proceeding with recognition.")
        except Exception as e:
            print(f"\n[Error] An exception occurred during training: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # --- Recognise the Sudoku ---
    predicted_grid, confidence_values, rectified_image = recognise_sudoku(
        image_path, classifier, debug_cells=debug_cells_flag
    )

    # --- Print Results ---
    print_sudoku_grid(predicted_grid, confidence_values) # Pass confidence values

    # --- Optional: Display Rectified Image ---
    if rectified_image is not None:
        try:
            # Draw confidence values on the rectified image? (Optional, can be cluttered)
            # ...
            cv2.imshow("Rectified Sudoku Grid", rectified_image)
            print("\nDisplaying rectified grid. Press any key to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"\nCould not display the image (maybe no GUI available?): {e}")

if __name__ == "__main__":
    main()