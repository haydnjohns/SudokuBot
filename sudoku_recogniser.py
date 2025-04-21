"""
Command-line utility to detect a Sudoku grid in an image and recognise its digits
using the full-grid classifier.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np

# Import the updated classifier and extractor
import digit_classifier
from digit_extractor import GRID_SIZE, DEFAULT_RECTIFIED_SIZE, extract_cells_from_image

# Confidence threshold for final acceptance of a recognised digit (1-9)
# This can be tuned based on model performance.
FINAL_CONFIDENCE_THRESHOLD = 0.75 # Adjusted slightly


def print_sudoku_grid(
    grid: np.ndarray,
    conf: Optional[np.ndarray] = None,
    threshold: float = FINAL_CONFIDENCE_THRESHOLD
) -> None:
    """
    Nicely print a 9×9 Sudoku grid.
    Marks digits below the threshold with '?' if confidences are provided.
    Expects grid values 0-9, where 0 is empty.
    """
    grid = np.asarray(grid)
    if grid.shape != (GRID_SIZE, GRID_SIZE):
        print(f"[print_sudoku_grid] Invalid grid shape: {grid.shape}")
        return

    print("-" * 25) # Top border
    for r in range(GRID_SIZE):
        if r > 0 and r % 3 == 0:
            print("|-------+-------+-------|") # Separator line

        line = ["|"] # Start of row
        for c in range(GRID_SIZE):
            digit = grid[r, c]
            token = ""
            if digit == 0:
                token = "." # Represent empty cell
            else:
                token = str(digit)
                # If confidence is provided and below threshold, mark as uncertain
                if conf is not None and conf.shape == (GRID_SIZE, GRID_SIZE) and conf[r, c] < threshold:
                    token = "?"

            line.append(f" {token} ") # Add token with spacing

            if (c + 1) % 3 == 0:
                line.append("|") # Add column separator

        print("".join(line)) # Print the row
    print("-" * 25) # Bottom border
    print() # Add a blank line after the grid


def display_results_on_image(
    rectified: Optional[np.ndarray],
    grid: np.ndarray,
    conf: Optional[np.ndarray] = None,
    threshold: float = FINAL_CONFIDENCE_THRESHOLD
) -> Optional[np.ndarray]:
    """
    Draw recognised digits (above threshold) onto the rectified grid image.
    """
    if rectified is None:
        print("[Display] No rectified image provided.")
        return None

    # Ensure image is BGR for drawing colors
    img_display = (
        cv2.cvtColor(rectified, cv2.COLOR_GRAY2BGR)
        if rectified.ndim == 2 else rectified.copy()
    )
    h, w = img_display.shape[:2]
    if h == 0 or w == 0: return None # Invalid image dimensions

    cell_h, cell_w = h // GRID_SIZE, w // GRID_SIZE
    if cell_h == 0 or cell_w == 0: return img_display # Cannot draw if cells are too small

    # Choose font properties
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale_base = min(cell_h, cell_w) / 35.0 # Scale font based on cell size
    font_thickness = max(1, int(font_scale_base * 1.5))

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            digit = grid[r, c]
            confidence = conf[r, c] if conf is not None else 1.0 # Assume high conf if not provided

            # Only draw digits that are not empty and meet the confidence threshold
            if digit != 0 and confidence >= threshold:
                text = str(digit)

                # Calculate text size to center it
                (text_w, text_h), baseline = cv2.getTextSize(text, font_face, font_scale_base, font_thickness)

                # Calculate center position of the cell
                center_x = c * cell_w + cell_w // 2
                center_y = r * cell_h + cell_h // 2

                # Calculate bottom-left corner of the text for centering
                text_x = center_x - text_w // 2
                text_y = center_y + text_h // 2

                # Draw the text
                cv2.putText(
                    img_display,
                    text,
                    (text_x, text_y),
                    font_face,
                    font_scale_base,
                    (0, 200, 0), # Green color for recognised digits
                    font_thickness,
                    cv2.LINE_AA # Anti-aliased text
                )
            # Optional: Draw uncertain digits differently (e.g., red '?')
            # elif digit != 0 and conf is not None and confidence < threshold:
            #     text = "?"
            #     (text_w, text_h), baseline = cv2.getTextSize(text, font_face, font_scale_base, font_thickness)
            #     center_x = c * cell_w + cell_w // 2
            #     center_y = r * cell_h + cell_h // 2
            #     text_x = center_x - text_w // 2
            #     text_y = center_y + text_h // 2
            #     cv2.putText(img_display, text, (text_x, text_y), font_face, font_scale_base, (0, 0, 255), font_thickness, cv2.LINE_AA)


    return img_display


def recognise_sudoku(
    img_path: Path,
    classifier: digit_classifier.DigitClassifier
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract rectified grid and run the full-grid classifier.

    Returns:
        Tuple of (predicted_grid, confidence_grid, rectified_image).
        predicted_grid: (9, 9) int array, 0 for empty/uncertain.
        confidence_grid: (9, 9) float array of confidences.
        rectified_image: The (e.g., 252x252) rectified image.
        Returns (None, None, None) on failure.
    """
    print(f"Processing {img_path.name}...")
    start_time = time.time()

    # 1. Extract the rectified grid image
    # We don't need individual cells from the extractor anymore
    # Use the input size expected by the classifier model
    rectified_size = classifier._model_input_shape[0]
    _, rectified_img, _ = extract_cells_from_image(
        img_path,
        size=rectified_size,
        debug=False # Set to True for extraction debugging images
    )

    if rectified_img is None:
        print("❌ Failed to extract or rectify the Sudoku grid.")
        return None, None, None
    print(f"✅ Grid extracted and rectified ({time.time() - start_time:.2f}s)")
    extract_time = time.time()

    # 2. Recognise digits using the full-grid classifier
    print("🧠 Running grid recognition model...")
    predicted_grid, confidence_grid = classifier.recognise_grid(
        rectified_img,
        confidence_threshold=FINAL_CONFIDENCE_THRESHOLD
    )

    if predicted_grid is None or confidence_grid is None:
        print("❌ Model recognition failed.")
        return None, None, rectified_img # Return rectified image even if recognition fails

    elapsed_time = time.time() - start_time
    print(f"✅ Recognition complete ({time.time() - extract_time:.2f}s)")
    print(f"⏱️ Total time: {elapsed_time:.2f}s")

    return predicted_grid, confidence_grid, rectified_img


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python sudoku_recogniser.py <image_path> [--no-train]")
        print("  <image_path>: Path to the Sudoku image file.")
        print("  --no-train: Optional flag to prevent training if the model is missing.")
        sys.exit(0)

    img_path = Path(sys.argv[1])
    if not img_path.is_file():
        print(f"❌ Error: Image file not found at '{img_path}'")
        sys.exit(1)

    # Check for --no-train flag
    force_no_train = "--no-train" in sys.argv

    # Instantiate the classifier
    # It will try to load the model. Training is required if the model file
    # doesn't exist, unless --no-train is specified.
    model_exists = Path(digit_classifier.MODEL_FILENAME).exists()
    training_required = not model_exists

    if training_required and force_no_train:
        print("Model file not found, but --no-train specified. Exiting.")
        sys.exit(1)
    elif training_required:
        print(f"⚠️ Model file '{digit_classifier.MODEL_FILENAME}' not found.")
        print("Initiating training process...")
        classifier = digit_classifier.DigitClassifier(training_required=True)
        classifier.train() # Use default training parameters from classifier script
        # Check if model was successfully created after training
        if classifier.model is None:
             print("❌ Training failed to produce a model. Exiting.")
             sys.exit(1)
        print("✅ Training complete. Proceeding with recognition...")
    else:
        # Model exists or --no-train was used with existing model
        classifier = digit_classifier.DigitClassifier(training_required=False)
        if classifier.model is None:
             print(f"❌ Failed to load existing model '{digit_classifier.MODEL_FILENAME}'. Exiting.")
             sys.exit(1)
        print(f"✅ Model '{digit_classifier.MODEL_FILENAME}' loaded.")


    # Recognise the Sudoku in the input image
    try:
        grid, conf, rectified = recognise_sudoku(img_path, classifier)

        if grid is None or conf is None:
            print("\n❌ Sudoku recognition process failed.")
            # Rectified might still exist, show it if possible
            if rectified is not None:
                 cv2.imshow("Failed Extraction/Rectification", rectified)
                 cv2.waitKey(0)
            sys.exit(1)

        # Print the recognised grid to the console
        print("\n--- Recognised Sudoku Grid ---")
        print_sudoku_grid(grid, conf, threshold=FINAL_CONFIDENCE_THRESHOLD)

        # Display the rectified image with recognised digits overlaid
        result_img = display_results_on_image(rectified, grid, conf, threshold=FINAL_CONFIDENCE_THRESHOLD)
        if result_img is not None:
            print("ℹ️ Displaying recognised grid on image. Press any key to close.")
            cv2.imshow("Sudoku Recognition Result", result_img)
            cv2.waitKey(0)
        else:
            print("ℹ️ Could not generate result image.")

    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        cv2.destroyAllWindows() # Ensure any OpenCV windows are closed


if __name__ == "__main__":
    main()
