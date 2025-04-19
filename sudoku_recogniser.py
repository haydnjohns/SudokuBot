# sudoku_recogniser.py
import cv2
import numpy as np
import sys
from pathlib import Path
import time

# Import necessary components from other modules
from digit_extractor import extract_cells_from_image
from digit_classifier import DigitClassifier, MODEL_FILENAME

# --- Constants ---
GRID_SIZE = 9

def recognise_sudoku(image_path, classifier):
    """
    Takes an image path, extracts cells, and uses the classifier to predict digits.

    Args:
        image_path (str | Path): Path to the Sudoku image.
        classifier (DigitClassifier): An initialized DigitClassifier instance.

    Returns:
        tuple: (predicted_grid, confidence_mask, rectified_image)
            - predicted_grid (np.ndarray | None): 9x9 NumPy array of recognised digits (0 for empty/unknown).
            - confidence_mask (np.ndarray | None): 9x9 NumPy array (boolean) indicating high confidence cells.
            - rectified_image (np.ndarray | None): The perspective-corrected grid image.
            Returns (None, None, None) if extraction fails.
    """
    print(f"\nProcessing image: {image_path}")
    start_time = time.time()

    # 1. Extract Cells
    cells, rectified_grid, _ = extract_cells_from_image(image_path)

    if cells is None:
        print("Failed to extract Sudoku grid or cells.")
        return None, None, None

    if len(cells) != GRID_SIZE * GRID_SIZE:
        print(f"Error: Expected {GRID_SIZE*GRID_SIZE} cells, got {len(cells)}.")
        return None, None, None

    print(f"Grid extraction successful ({time.time() - start_time:.2f}s)")
    extract_time = time.time()

    # 2. Recognise Digits in Cells
    predicted_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    confidence_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)

    for i, cell_img in enumerate(cells):
        row, col = divmod(i, GRID_SIZE)

        if cell_img is None or cell_img.size < 10: # Skip tiny/empty cells from extraction margin issues
            predicted_grid[row, col] = 0
            confidence_mask[row, col] = True # Confident it's empty
            continue

        # Use the classifier's recognise method
        # Use a slightly lower threshold here, let the mask show uncertainty
        digit = classifier.recognise(cell_img, confidence_threshold=0.6)
        predicted_grid[row, col] = digit

        # Set confidence mask based on whether a non-zero digit was predicted confidently
        # (We are confident about '0' if the model predicted EMPTY_LABEL or low conf)
        is_confident = (digit != 0) # Assume confident if recognise returned non-zero

        # Refine confidence: check if prediction was EMPTY or low conf
        # This requires modifying recognise or adding another method to return confidence
        # For now, let's assume recognise returning non-zero means reasonably confident
        # A better approach would be for recognise to return (digit, confidence)
        confidence_mask[row, col] = True # Placeholder - refine if recognise returns confidence

    print(f"Digit recognition complete ({time.time() - extract_time:.2f}s)")
    print(f"Total processing time: {time.time() - start_time:.2f}s")

    return predicted_grid, confidence_mask, rectified_grid


def print_sudoku_grid(grid, confidence_mask=None):
    """Prints the Sudoku grid nicely formatted."""
    if grid is None:
        print("No grid to print.")
        return

    print("\nDetected Sudoku Grid (0 = empty/unknown):")
    print("-" * 25)
    for r in range(GRID_SIZE):
        if r % 3 == 0 and r != 0:
            print("|-------+-------+-------|")
        row_str = "| "
        for c in range(GRID_SIZE):
            digit = grid[r, c]
            uncertain = ""
            # Uncomment below if confidence_mask is implemented reliably
            # if confidence_mask is not None and not confidence_mask[r, c] and digit != 0:
            #     uncertain = "?" # Mark uncertain non-zero digits

            row_str += f"{digit}{uncertain} "
            if (c + 1) % 3 == 0:
                row_str += "| "
        print(row_str)
    print("-" * 25)


def main():
    if len(sys.argv) < 2:
        print("Usage: python sudoku_recogniser.py <path_to_sudoku_image>")
        # Try running on a default test image if available
        test_image = Path("rendered_sudoku_specific.png") # Or a real photo path
        if test_image.exists():
             print(f"\nNo image provided. Running on default test image: {test_image}")
             image_path = test_image
        else:
             sys.exit(1)
    else:
        image_path = Path(sys.argv[1])

    if not image_path.exists():
        print(f"Error: Image file not found at '{image_path}'")
        sys.exit(1)

    # --- Initialize the Classifier ---
    # This will load the model or trigger training if needed
    print("Initializing Digit Classifier...")
    # Check if model exists, if not, inform user training will start
    model_p = Path(MODEL_FILENAME)
    needs_training = not model_p.exists()
    if needs_training:
         print(f"Classifier model ('{model_p.name}') not found.")
         print("The classifier needs to be trained using synthetic data.")
         print("This may take several minutes depending on your system...")
         # Ask for confirmation?
         # confirm = input("Proceed with training? (y/n): ")
         # if confirm.lower() != 'y':
         #     print("Training aborted.")
         #     sys.exit(1)

    classifier = DigitClassifier() # Loads or prepares for training

    # --- Train if necessary ---
    if classifier.model is None:
        print("\nStarting classifier training...")
        try:
            # Use default training parameters for the first run
            classifier.train()
            if classifier.model is None: # Check if training failed
                 print("\n[Error] Classifier training failed. Exiting.")
                 sys.exit(1)
            print("\nTraining complete. Proceeding with recognition.")
        except Exception as e:
            print(f"\n[Error] An exception occurred during training: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # --- Recognise the Sudoku ---
    predicted_grid, confidence_mask, rectified_image = recognise_sudoku(image_path, classifier)

    # --- Print Results ---
    print_sudoku_grid(predicted_grid, confidence_mask)

    # --- Optional: Display Rectified Image ---
    if rectified_image is not None:
        try:
            cv2.imshow("Rectified Sudoku Grid", rectified_image)
            print("\nDisplaying rectified grid. Press any key to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"\nCould not display the image (maybe no GUI available?): {e}")

if __name__ == "__main__":
    main()