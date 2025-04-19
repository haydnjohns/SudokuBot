# sudoku_recogniser.py
import os

import cv2
import numpy as np
from pathlib import Path
import time

# Import necessary components from sibling modules
try:
    from digit_extractor import extract_digits
    from digit_classifier import DigitClassifier
except ImportError:
    print("[ERROR] Cannot import digit_extractor or DigitClassifier.")
    print("        Ensure digit_extractor.py and digit_classifier.py are in the same directory.")
    # Define dummy classes/functions if imports fail
    def extract_digits(*args, **kwargs): return [], None, None
    class DigitClassifier:
        def predict(self, *args, **kwargs): return 0 # Predict empty always
        def __init__(self, *args, **kwargs): pass


class SudokuRecogniser:
    """
    Recognises digits in a Sudoku puzzle image using DigitExtractor and DigitClassifier.
    """
    def __init__(self, classifier_model_path=None, extractor_options=None, classifier_options=None):
        """
        Initializes the Sudoku Recogniser.

        Args:
            classifier_model_path (str, optional): Path to the pre-trained classifier model.
                                                   Defaults to the DigitClassifier default.
            extractor_options (dict, optional): Options for digit_extractor.extract_digits.
            classifier_options (dict, optional): Options for DigitClassifier initialization
                                                 (e.g., input_shape if non-default).
        """
        classifier_args = classifier_options if classifier_options else {}
        if classifier_model_path:
            classifier_args['model_filename'] = classifier_model_path

        print("[INFO] Initializing Digit Classifier for Recogniser...")
        self.classifier = DigitClassifier(**classifier_args)
        if self.classifier.model is None:
             print("[WARN] Classifier model could not be loaded during Recogniser init.")
             # Depending on use case, you might raise an error here
             # raise ValueError("Classifier model failed to load.")

        self.extractor_options = extractor_options if extractor_options else {}
        # Example: Set default extractor options if needed
        # self.extractor_options.setdefault('target_grid_size', 450)
        self.extractor_options.setdefault('cell_border_frac', 0.08) # Match potential training

    def recognise(self, image_path_or_array, confidence_threshold=0.7):
        """
        Takes a Sudoku image and returns the recognised grid.

        Args:
            image_path_or_array (str | Path | np.ndarray): Path to the image or the image array.
            confidence_threshold (float): Minimum confidence for the classifier to accept a digit.

        Returns:
            tuple: (predicted_grid, rectified_image, status_message)
                - predicted_grid (np.ndarray | None): 9x9 numpy array of recognised digits
                                                     (0 for empty/uncertain). None on major failure.
                - rectified_image (np.ndarray | None): The perspective-corrected grid image. None if extraction fails.
                - status_message (str): A message indicating success or failure reason.
        """
        start_time = time.time()
        print(f"\n[INFO] Starting Sudoku recognition for: {image_path_or_array if isinstance(image_path_or_array, (str, Path)) else 'image array'}")

        if self.classifier.model is None:
            # Attempt to load again if it failed during init
            self.classifier._load_model()
            if self.classifier.model is None:
                return None, None, "Error: Classifier model is not loaded."

        # 1. Extract Cells
        print("[INFO] Extracting grid and cells...")
        cells, rectified_grid, contour = extract_digits(
            image_path_or_array,
            **self.extractor_options
        )

        if not cells or len(cells) != 81:
            msg = "Error: Failed to extract 81 cells."
            if contour is None:
                msg += " (Grid contour not found)"
            elif rectified_grid is None:
                 msg += " (Grid rectification failed)"
            else:
                 msg += f" (Found {len(cells)} cells)"
            print(f"[ERROR] {msg}")
            # Return the rectified grid if available, even if cell extraction failed
            return None, rectified_grid, msg

        print(f"[INFO] Cell extraction successful ({time.time() - start_time:.2f}s)")

        # 2. Classify each cell
        print("[INFO] Classifying digits in cells...")
        predicted_grid = np.zeros((9, 9), dtype=int)
        classification_start_time = time.time()

        for i, cell_img in enumerate(cells):
            row, col = divmod(i, 9)
            predicted_digit = self.classifier.predict(cell_img, confidence_threshold)
            predicted_grid[row, col] = predicted_digit

        end_time = time.time()
        print(f"[INFO] Classification complete ({end_time - classification_start_time:.2f}s)")
        print(f"[INFO] Total recognition time: {end_time - start_time:.2f}s")

        return predicted_grid, rectified_grid, "Success: Recognition complete."


# --- Example Usage ---
if __name__ == "__main__":
    print("[INFO] Initializing Sudoku Recogniser...")
    # Specify model path if it's not the default used by DigitClassifier
    # recogniser = SudokuRecogniser(classifier_model_path="path/to/your/model.keras")
    recogniser = SudokuRecogniser()

    # --- Test on a sample image ---
    # Generate a test image or use an existing one
    test_image_path = "temp_recogniser_test.png"
    try:
        from sudoku_renderer import SudokuRenderer
        print("[INFO] Generating a test image using SudokuRenderer...")
        # Use settings that might be challenging
        renderer = SudokuRenderer(warp_intensity=0.25, noise_level_range=(10, 25), blur_kernel_range=(1,2))
        test_img, test_truth, _ = renderer.render_sudoku(difficulty=0.5)
        cv2.imwrite(test_image_path, test_img)
        print(f"[INFO] Test image saved to {test_image_path}")
    except ImportError:
        print("[WARN] SudokuRenderer not found. Using fallback image path.")
        test_image_path = "sample_images/digit_5_img_0.png" # Fallback path
    except Exception as e:
         print(f"[WARN] Failed to generate test image with renderer: {e}. Using fallback path.")
         test_image_path = "sample_images/digit_5_img_0.png" # Fallback path


    if not Path(test_image_path).exists():
         print(f"[ERROR] Test image '{test_image_path}' not found. Cannot run recognition.")
         exit()

    # --- Run Recognition ---
    grid, rectified, message = recogniser.recognise(test_image_path, confidence_threshold=0.65)

    print(f"\n[INFO] Recognition Status: {message}")

    if grid is not None:
        print("\n[INFO] Recognised Sudoku Grid (0 = empty/uncertain):")
        print(grid)

        # Compare with ground truth if we generated the image
        if 'test_truth' in locals():
             print("\n[INFO] Ground Truth Grid:")
             print(test_truth)
             correct = np.sum((grid == test_truth) & (test_truth != 0))
             total = np.sum(test_truth != 0)
             accuracy = (correct / total) * 100 if total > 0 else float('nan')
             print(f"\nAccuracy on non-empty cells: {accuracy:.2f}% ({correct}/{total})")

        if rectified is not None:
            cv2.imshow("Rectified Grid from Recogniser", rectified)
            print("\n[INFO] Displaying rectified grid. Press any key to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("[WARN] Rectified grid was not available for display.")

    else:
        print("[ERROR] Sudoku recognition failed.")
        if rectified is not None:
             # Show rectified grid even on failure if available
             cv2.imshow("Rectified Grid (Recognition Failed)", rectified)
             print("\n[INFO] Displaying rectified grid (recognition failed). Press any key.")
             cv2.waitKey(0)
             cv2.destroyAllWindows()


    # Clean up temp file
    if Path("temp_recogniser_test.png").exists():
        try:
            os.remove("temp_recogniser_test.png")
        except Exception as e:
            print(f"Warning: Could not remove temp file {test_image_path}: {e}")

    print("\n[INFO] Recogniser testing complete.")