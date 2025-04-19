# sudoku_recogniser.py
import cv2
import numpy as np
import sys
from pathlib import Path
import time

# Local imports
from digit_extractor import extract_cells_from_image, GRID_SIZE
from digit_classifier import DigitClassifier, MODEL_FILENAME, EMPTY_LABEL

# --- Constants ---
# Confidence threshold for accepting a digit prediction during final recognition
FINAL_CONFIDENCE_THRESHOLD = 0.80

# --- Core Recognition Logic ---
def recognise_sudoku(image_path, classifier, debug_cells=False):
    """
    Takes an image path, extracts Sudoku cells, and uses the classifier to predict digits.

    Args:
        image_path (str | Path): Path to the Sudoku image.
        classifier (DigitClassifier): An initialized DigitClassifier instance.
        debug_cells (bool): If True, save raw and preprocessed cell images during recognition.

    Returns:
        tuple: (predicted_grid, confidence_values, rectified_image)
            - predicted_grid (np.ndarray | None): 9x9 NumPy array of recognised digits (0 for empty/unknown).
            - confidence_values (np.ndarray | None): 9x9 NumPy array of confidence scores (0.0-1.0) for each cell's prediction.
            - rectified_image (np.ndarray | None): The perspective-corrected grid image.
    """
    print(f"\nProcessing image: {image_path}")
    start_time = time.time()

    # 1. Extract cells from the image
    cells, rectified_grid, grid_contour = extract_cells_from_image(image_path, debug=False) # Debug for extractor is separate

    if cells is None:
        print("Failed to extract Sudoku grid or cells.")
        return None, None, None

    if len(cells) != GRID_SIZE * GRID_SIZE:
        print(f"Error: Expected {GRID_SIZE*GRID_SIZE} cells, got {len(cells)}.")
        return None, None, rectified_grid # Return rectified grid if available

    print(f"Grid extraction successful ({time.time() - start_time:.2f}s)")
    extract_time = time.time()

    # Prepare directory for saving debug cell images if requested
    cell_debug_dir = None
    if debug_cells:
        cell_debug_dir = Path(f"debug_recogniser_cells_{Path(image_path).stem}")
        cell_debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"Debugging cells to: {cell_debug_dir}")
        if rectified_grid is not None:
             # Save the rectified grid in the debug folder for context
             cv2.imwrite(str(cell_debug_dir / "_rectified_grid.png"), rectified_grid)

    # Initialize result arrays
    predicted_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    confidence_values = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)

    # Check if classifier model is loaded
    if classifier.model is None:
         print("[ERROR] Classifier model is not loaded. Cannot perform recognition.")
         # Return results based on extraction only
         return np.zeros((GRID_SIZE, GRID_SIZE), dtype=int), np.zeros((GRID_SIZE, GRID_SIZE), dtype=float), rectified_grid

    # 2. Classify each cell
    for i, cell_img in enumerate(cells):
        row, col = divmod(i, GRID_SIZE) # Get row and column index

        if cell_debug_dir and cell_img is not None:
             # Save the raw extracted cell image
             cv2.imwrite(str(cell_debug_dir / f"cell_{row}_{col}_0_raw.png"), cell_img)

        # Handle potentially empty or invalid cell images from extraction
        if cell_img is None or cell_img.size < 10: # Basic check
            predicted_grid[row, col] = 0
            confidence_values[row, col] = 1.0 # High confidence it's empty/invalid based on input
            if cell_debug_dir:
                 # Save a placeholder for preprocessed image
                 empty_dbg = np.zeros(classifier._model_input_size, dtype=np.uint8)
                 cv2.imwrite(str(cell_debug_dir / f"cell_{row}_{col}_1_preprocessed_EMPTYINPUT.png"), empty_dbg)
            continue

        # Perform recognition using the classifier
        # Use a low internal threshold to get the raw prediction and confidence
        digit, confidence = classifier.recognise(cell_img, confidence_threshold=0.1)
        confidence_values[row, col] = confidence

        # Apply the final threshold to decide the digit for the output grid
        if digit != 0 and confidence >= FINAL_CONFIDENCE_THRESHOLD:
             predicted_grid[row, col] = digit
        else:
             # Mark as empty/unknown if prediction is empty class, or confidence is too low
             predicted_grid[row, col] = 0

        # Save preprocessed cell image and prediction details if debugging
        if cell_debug_dir:
            processed_for_debug = classifier._preprocess_cell_for_model(cell_img)
            if processed_for_debug is not None:
                 # Convert normalized float image back to uint8 for saving
                 processed_img_uint8 = (processed_for_debug * 255).astype(np.uint8)
                 # Create filename suffix with prediction info
                 pred_suffix = f"_pred{digit}_conf{confidence:.2f}"
                 if predicted_grid[row, col] == 0 and digit != 0:
                     pred_suffix += "_REJECTED" # Mark if rejected due to threshold
                 elif digit == 0:
                     pred_suffix += "_EMPTY" # Mark if predicted as empty class
                 cv2.imwrite(str(cell_debug_dir / f"cell_{row}_{col}_1_preprocessed{pred_suffix}.png"), processed_img_uint8)
            else:
                 # Save a placeholder if preprocessing failed
                 failed_dbg = np.full(classifier._model_input_size, 128, dtype=np.uint8) # Gray image
                 cv2.imwrite(str(cell_debug_dir / f"cell_{row}_{col}_1_preprocessed_FAILED.png"), failed_dbg)

    print(f"Digit recognition complete ({time.time() - extract_time:.2f}s)")
    print(f"Total processing time: {time.time() - start_time:.2f}s")

    return predicted_grid, confidence_values, rectified_grid


# --- Utility Functions ---
def print_sudoku_grid(grid, confidence_values=None, threshold=FINAL_CONFIDENCE_THRESHOLD):
    """
    Prints the Sudoku grid to the console in a formatted way.
    Digits below the threshold (or predicted as empty) are shown as '.'.
    Optionally marks uncertain predictions with '?'.

    Args:
        grid (np.ndarray | list[list]): The 9x9 Sudoku grid (0 for empty/unknown).
        confidence_values (np.ndarray | None): Optional 9x9 array of confidence scores.
                                              If provided, digits predicted with confidence
                                              below `threshold` but not empty are marked '?'.
        threshold (float): Confidence threshold used for display.
    """
    if grid is None:
        print("No grid data to print.")
        return

    try:
        grid = np.array(grid) # Ensure it's a NumPy array
        if grid.shape != (GRID_SIZE, GRID_SIZE):
             print(f"Invalid grid shape for printing: {grid.shape}. Expected ({GRID_SIZE}, {GRID_SIZE}).")
             return
    except Exception as e:
        print(f"Error converting grid to NumPy array: {e}")
        return

    # Check if confidence values are provided and valid for marking uncertainty
    show_uncertainty = isinstance(confidence_values, np.ndarray) and \
                       confidence_values.shape == (GRID_SIZE, GRID_SIZE)

    print(f"\nDetected Sudoku Grid (Display Threshold: {threshold:.2f}):")
    print("-" * 25) # Top border
    for r in range(GRID_SIZE):
        # Print horizontal separator between 3x3 blocks
        if r % 3 == 0 and r != 0:
            print("|-------+-------+-------|")
        row_str = "| " # Start of row
        for c in range(GRID_SIZE):
            digit = grid[r, c]
            display_char = str(digit) if digit != 0 else "."
            uncertain_marker = ""

            # Mark with '?' if confidence is provided and below threshold, but wasn't empty
            if show_uncertainty and digit == 0: # Check original prediction if available
                 # This requires knowing the raw prediction before thresholding,
                 # which isn't directly passed. We infer based on confidence.
                 # If confidence is high for '0', it's likely truly empty.
                 # If confidence is low for '0', it might be a failed digit recognition.
                 # Let's mark low-confidence zeros.
                 # A better approach might involve passing raw predictions.
                 pass # Simplified: just show '.' for zeros based on final grid.

            row_str += f"{display_char}{uncertain_marker} "
            # Print vertical separator between 3x3 blocks
            if (c + 1) % 3 == 0:
                row_str += "| "
        print(row_str) # Print the completed row
    print("-" * 25) # Bottom border


def display_results_on_image(rectified_image, predicted_grid):
    """
    Overlays the predicted digits onto the rectified Sudoku grid image for display.

    Args:
        rectified_image (np.ndarray): The perspective-corrected grid image.
        predicted_grid (np.ndarray): The 9x9 grid of predicted digits.

    Returns:
        np.ndarray: The rectified image with predicted digits drawn on it.
    """
    if rectified_image is None or predicted_grid is None:
        return None

    display_img = rectified_image.copy()
    # Convert to BGR if grayscale for colored text drawing
    if display_img.ndim == 2:
        display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)

    h, w = display_img.shape[:2]
    cell_h, cell_w = h // GRID_SIZE, w // GRID_SIZE

    # Font properties
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_thickness = 2
    text_color = (0, 255, 0) # Green
    shadow_color = (0, 0, 0) # Black

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            digit = predicted_grid[r, c]
            if digit != 0: # Only draw non-empty predictions
                text = str(digit)
                # Calculate text size to center it
                (text_w, text_h), baseline = cv2.getTextSize(text, font_face, font_scale, font_thickness)
                # Calculate center of the cell
                center_x = c * cell_w + cell_w // 2
                center_y = r * cell_h + cell_h // 2
                # Calculate bottom-left corner of the text for cv2.putText
                text_x = center_x - text_w // 2
                text_y = center_y + text_h // 2

                # Draw shadow first (slightly offset) for better visibility
                cv2.putText(display_img, text, (text_x + 1, text_y + 1), font_face, font_scale, shadow_color, font_thickness, cv2.LINE_AA)
                # Draw the main text
                cv2.putText(display_img, text, (text_x, text_y), font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

    return display_img


# --- Main Execution Block ---
def main():
    """
    Main function to handle command-line arguments, run the recognition,
    and display results.
    """
    # --- Argument Parsing ---
    args = sys.argv[1:]
    image_path_str = None
    debug_cells_flag = False
    force_train_flag = False

    # Basic argument checking
    if "--debug-cells" in args:
        debug_cells_flag = True
        args.remove("--debug-cells")
    if "--force-train" in args:
        force_train_flag = True
        args.remove("--force-train")

    if len(args) >= 1:
        image_path_str = args[0]
    else:
        # No image path provided, try to find a default test image
        print("Usage: python sudoku_recogniser.py <path_to_sudoku_image> [--debug-cells] [--force-train]")
        default_paths = ["epoch_test_sudoku.png", "rendered_sudoku_specific.png", "rendered_sudoku_random.png"]
        found_default = False
        for p in default_paths:
             if Path(p).exists():
                 image_path_str = p
                 found_default = True
                 break
        if found_default:
             print(f"\nNo image path provided. Using default test image: {image_path_str}")
        else:
             print("\nError: No image path provided and default test images not found.")
             print("Please provide a path to a Sudoku image.")
             sys.exit(1)

    image_path = Path(image_path_str)
    if not image_path.exists():
        print(f"Error: Image file not found at '{image_path}'")
        sys.exit(1)

    # --- Classifier Initialization and Training ---
    print(f"\nInitializing Digit Classifier (Model: {MODEL_FILENAME})...")
    classifier = DigitClassifier(model_path=MODEL_FILENAME, training_required=force_train_flag)

    # Train the classifier if needed
    if classifier.model is None:
        print(f"\nClassifier model ('{MODEL_FILENAME}') needs training or failed to load.")
        print("Starting training process...")
        try:
            classifier.train() # Train the model
            # Check if training was successful
            if classifier.model is None:
                 print("\n[Error] Classifier training failed or did not produce a model. Exiting.")
                 sys.exit(1)
            print("\nTraining complete. Proceeding with recognition.")
        except Exception as e:
            print(f"\n[Error] An exception occurred during classifier training: {e}")
            import traceback; traceback.print_exc()
            sys.exit(1)
    else:
        print("Classifier model loaded successfully.")

    # --- Sudoku Recognition ---
    predicted_grid, confidence_values, rectified_image = recognise_sudoku(
        image_path, classifier, debug_cells=debug_cells_flag
    )

    # --- Display Results ---
    print_sudoku_grid(predicted_grid, confidence_values, FINAL_CONFIDENCE_THRESHOLD)

    if rectified_image is not None:
        try:
            # Overlay predictions on the rectified image
            result_image = display_results_on_image(rectified_image, predicted_grid)
            if result_image is not None:
                cv2.imshow("Rectified Sudoku Grid with Predictions", result_image)
                print("\nDisplaying rectified grid with predictions.")
                print("Press any key in the image window to close.")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("\nCould not generate result image for display.")
        except Exception as e:
            # Handle cases where GUI is not available (e.g., running on server)
            print(f"\nCould not display the result image (GUI might be unavailable): {e}")
            # Optionally save the result image instead
            save_path = f"result_{image_path.stem}.png"
            if result_image is not None:
                cv2.imwrite(save_path, result_image)
                print(f"Saved result image to '{save_path}'")


if __name__ == "__main__":
    main()

