# sudoku_recogniser.py
import cv2
import numpy as np
import sys
from pathlib import Path
import time

# Import necessary components from other modules
from digit_extractor import extract_cells_from_image
# --- Use v5 Model Filename ---
from digit_classifier import DigitClassifier, MODEL_FILENAME

# --- Constants ---
GRID_SIZE = 9
# Confidence threshold for accepting a digit prediction during final recognition
FINAL_CONFIDENCE_THRESHOLD = 0.80 # Maybe increase slightly if model is better

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

    # 1. Extract Cells
    # Set debug=True here if you want to see the contour finding steps from digit_extractor
    cells, rectified_grid, grid_contour = extract_cells_from_image(image_path, debug=False)

    if cells is None:
        print("Failed to extract Sudoku grid or cells.")
        return None, None, None

    if len(cells) != GRID_SIZE * GRID_SIZE:
        print(f"Error: Expected {GRID_SIZE*GRID_SIZE} cells, got {len(cells)}.")
        return None, None, None

    print(f"Grid extraction successful ({time.time() - start_time:.2f}s)")
    extract_time = time.time()

    # --- Debug Cell Saving Setup ---
    cell_debug_dir = None
    if debug_cells:
        cell_debug_dir = Path(f"debug_recogniser_cells_{Path(image_path).stem}")
        cell_debug_dir.mkdir(exist_ok=True)
        print(f"Debugging cells to: {cell_debug_dir}")
        # Save the rectified grid for context
        cv2.imwrite(str(cell_debug_dir / "_rectified_grid.png"), rectified_grid)
    # ---

    # 2. Recognise Digits in Cells
    predicted_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    confidence_values = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)

    if classifier.model is None:
         print("[ERROR] Classifier model is not loaded in recognise_sudoku. Cannot proceed.")
         return None, None, rectified_grid # Return grid even if recognition fails

    for i, cell_img in enumerate(cells):
        row, col = divmod(i, GRID_SIZE)

        # --- Save raw cell before preprocessing for debug ---
        if cell_debug_dir and cell_img is not None:
             cv2.imwrite(str(cell_debug_dir / f"cell_{row}_{col}_0_raw.png"), cell_img)
        # ---

        if cell_img is None or cell_img.size < 10: # Check validity before recognition
            predicted_grid[row, col] = 0
            confidence_values[row, col] = 1.0 # Confident it's empty based on input
            # Save an empty image for debugging consistency
            if cell_debug_dir:
                 empty_dbg = np.zeros(classifier._model_input_size, dtype=np.uint8)
                 cv2.imwrite(str(cell_debug_dir / f"cell_{row}_{col}_1_preprocessed_EMPTY.png"), empty_dbg)
            continue

        # Use the classifier's recognise method
        # Pass a lower internal threshold to always get a prediction and confidence
        # The FINAL_CONFIDENCE_THRESHOLD is applied *after* this call.
        digit, confidence = classifier.recognise(cell_img, confidence_threshold=0.1) # Low internal threshold

        confidence_values[row, col] = confidence

        # Apply the *final* threshold for the grid output
        if digit != 0 and confidence >= FINAL_CONFIDENCE_THRESHOLD:
             predicted_grid[row, col] = digit # Keep digit if confidence is high enough
        else:
             # Mark as unknown (0) if it was predicted as empty OR
             # if it was a digit but confidence was too low.
             predicted_grid[row, col] = 0

        # --- Save preprocessed cell AFTER recognition attempt ---
        if cell_debug_dir:
            # Re-preprocess to get the image that was fed to the model
            # (recognise already does this, but we do it again for saving)
            processed_for_debug = classifier._preprocess_cell_for_model(cell_img)
            if processed_for_debug is not None:
                 # Convert from float (0-1) back to uint8 (0-255) for saving
                 processed_img_uint8 = (processed_for_debug * 255).astype(np.uint8)
                 # Add suffix indicating the prediction for easier debugging
                 pred_suffix = f"_pred{digit}_conf{confidence:.2f}"
                 if predicted_grid[row, col] == 0 and digit != 0: # Mark if rejected by final threshold
                      pred_suffix += "_REJECTED"
                 elif digit == 0:
                      pred_suffix += "_EMPTY"

                 cv2.imwrite(str(cell_debug_dir / f"cell_{row}_{col}_1_preprocessed{pred_suffix}.png"), processed_img_uint8)
            else:
                 # Handle case where preprocessing failed during debug save
                 failed_dbg = np.full(classifier._model_input_size, 128, dtype=np.uint8) # Gray image
                 cv2.imwrite(str(cell_debug_dir / f"cell_{row}_{col}_1_preprocessed_FAILED.png"), failed_dbg)
        # ---


    print(f"Digit recognition complete ({time.time() - extract_time:.2f}s)")
    print(f"Total processing time: {time.time() - start_time:.2f}s")

    return predicted_grid, confidence_values, rectified_grid


# --- Revised Grid Printing ---
def print_sudoku_grid(grid, confidence_values=None, threshold=FINAL_CONFIDENCE_THRESHOLD):
    """Prints the Sudoku grid. Shows digits above threshold, '.' otherwise."""
    if grid is None:
        print("No grid to print.")
        return

    print(f"\nDetected Sudoku Grid (Threshold: {threshold:.2f}):")
    print("-" * 25)
    for r in range(GRID_SIZE):
        if r % 3 == 0 and r != 0: print("|-------+-------+-------|")
        row_str = "| "
        for c in range(GRID_SIZE):
            digit = grid[r, c] # This grid already incorporates the threshold check

            # Display the digit if it's non-zero (meaning it passed the threshold)
            # Otherwise, display '.' for empty or low-confidence cells.
            display_char = str(digit) if digit != 0 else "."

            # Optional: Add '?' only if confidence is available and *very* low (e.g. < 0.3)
            # low_conf_marker = ""
            # if confidence_values is not None and digit != 0 and confidence_values[r, c] < 0.3:
            #      low_conf_marker = "?"

            row_str += f"{display_char} " # Removed the '?' logic for simplicity
            if (c + 1) % 3 == 0: row_str += "| "
        print(row_str)
    print("-" * 25)


def main():
    if len(sys.argv) < 2:
        print("Usage: python sudoku_recogniser.py <path_to_sudoku_image> [--debug-cells] [--force-train]")
        # Try using a default rendered image if available
        test_image = Path("rendered_sudoku_specific.png") # Or a real photo path
        if not test_image.exists(): test_image = Path("rendered_sudoku_random.png")

        if test_image.exists():
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
    model_p = Path(MODEL_FILENAME) # Use v5 filename

    # Pass force_train flag to classifier init
    classifier = DigitClassifier(training_required=force_train_flag)

    # Train if model is None (means loading failed or force_train was True and file existed/was removed)
    if classifier.model is None:
        print(f"\nClassifier model ('{model_p.name}') needs training.")
        print("Training will start using synthetic data (this can take several minutes)...")
        try:
            # Use the new defaults defined in the classifier's train method
            classifier.train()
            if classifier.model is None: # Check if training actually produced a model
                 print("\n[Error] Classifier training failed to produce a model. Exiting.")
                 sys.exit(1)
            print("\nTraining complete. Proceeding with recognition.")
        except Exception as e:
            print(f"\n[Error] An exception occurred during training: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("Classifier model loaded successfully.")


    # --- Recognise the Sudoku ---
    predicted_grid, confidence_values, rectified_image = recognise_sudoku(
        image_path, classifier, debug_cells=debug_cells_flag
    )

    # --- Print Results ---
    # Pass confidence values and threshold for potential future use, even if print func doesn't use them now
    print_sudoku_grid(predicted_grid, confidence_values, FINAL_CONFIDENCE_THRESHOLD)

    # --- Optional: Display Rectified Image ---
    if rectified_image is not None:
        try:
            # Optionally draw the recognised grid onto the rectified image
            display_img = rectified_image.copy()
            if display_img.ndim == 2: # Convert to BGR if grayscale
                 display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)

            cell_h = display_img.shape[0] // GRID_SIZE
            cell_w = display_img.shape[1] // GRID_SIZE

            if predicted_grid is not None:
                for r in range(GRID_SIZE):
                    for c in range(GRID_SIZE):
                        digit = predicted_grid[r, c]
                        if digit != 0: # Draw only recognised digits
                            text = str(digit)
                            # Calculate position (center of cell)
                            center_x = c * cell_w + cell_w // 2
                            center_y = r * cell_h + cell_h // 2
                            # Get text size to center it better
                            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                            text_x = center_x - text_size[0] // 2
                            text_y = center_y + text_size[1] // 2
                            cv2.putText(display_img, text, (text_x, text_y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2) # Red color

            cv2.imshow("Rectified Sudoku Grid with Predictions", display_img)
            print("\nDisplaying rectified grid with predictions. Press any key to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"\nCould not display the image (maybe no GUI available?): {e}")

if __name__ == "__main__":
    main()