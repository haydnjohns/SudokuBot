# digit_extractor.py
import cv2
import numpy as np
from pathlib import Path
import random

# --- Constants ---
GRID_SIZE = 9
DEFAULT_RECTIFIED_SIZE = 450 # Target pixel size for the rectified grid image

# --- Helper Functions ---
def _order_points(pts):
    """
    Orders 4 points found for a contour: top-left, top-right, bottom-right, bottom-left.

    Args:
        pts (np.ndarray): Array of 4 points, shape (4, 2) or similar.

    Returns:
        np.ndarray: Array of 4 points ordered correctly, shape (4, 2), dtype float32.

    Raises:
        ValueError: If input cannot be reshaped to (4, 2).
    """
    if pts.shape != (4, 2):
         try:
             pts = pts.reshape(4, 2)
         except ValueError:
             raise ValueError(f"Input 'pts' must be reshapeable to (4, 2). Got shape {pts.shape}")

    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")

    # Top-left point has the smallest sum (x+y)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    # Bottom-right point has the largest sum (x+y)
    rect[2] = pts[np.argmax(s)]

    # Top-right point has the smallest difference (y-x)
    # Bottom-left point has the largest difference (y-x)
    diff_yx = pts[:, 1] - pts[:, 0] # More robust than np.diff for this ordering
    rect[1] = pts[np.argmin(diff_yx)]
    rect[3] = pts[np.argmax(diff_yx)]

    return rect

# --- Core Grid Finding and Extraction Logic ---

def find_sudoku_grid_contour(image, debug_dir=None):
    """
    Pre-processes the image and attempts to find the 4-point contour
    representing the outer boundary of the Sudoku grid.

    Args:
        image (np.ndarray): Input image (BGR or Grayscale).
        debug_dir (Path | None): If provided, saves intermediate processing steps to this directory.

    Returns:
        np.ndarray: A (4, 2) NumPy array containing the coordinates of the grid corners.

    Raises:
        ValueError: If input image is invalid or no suitable grid contour is found.
    """
    if image is None:
        raise ValueError("Input image is None.")

    # Convert to grayscale if necessary
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy() # Work on a copy if already grayscale

    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True) # Ensure debug directory exists

    # --- Preprocessing Steps ---
    # 1. Gaussian Blur to reduce noise
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    if debug_dir: cv2.imwrite(str(debug_dir / "01_blur.png"), blur)

    # 2. Adaptive Thresholding to binarize the image, highlighting lines
    thresh = cv2.adaptiveThreshold(blur, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, # Invert: grid lines should be white
                                   15, # Block size (must be odd)
                                   4)  # Constant subtracted from the mean
    if debug_dir: cv2.imwrite(str(debug_dir / "02_thresh.png"), thresh)

    # 3. Morphological Closing to connect potentially broken grid lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    if debug_dir: cv2.imwrite(str(debug_dir / "03_closed.png"), closed)
    # --- End Preprocessing ---

    # Find contours in the processed image
    contours, hierarchy = cv2.findContours(closed,
                                           cv2.RETR_EXTERNAL, # Retrieve only outer contours
                                           cv2.CHAIN_APPROX_SIMPLE) # Compress contour points

    if not contours:
        raise ValueError("No contours found after preprocessing.")

    # Sort contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    sudoku_contour = None
    min_area_ratio = 0.1 # Minimum area relative to the image size
    min_area = gray.shape[0] * gray.shape[1] * min_area_ratio

    if debug_dir:
        img_contours_debug = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) # For drawing contours

    # Iterate through sorted contours to find the grid
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        # Approximate the contour shape to simpler polygon
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True) # Epsilon factor might need tuning

        if debug_dir and i < 10: # Draw first few largest contours for debugging
             color = (random.randint(50,200), random.randint(50,200), random.randint(50,200))
             cv2.drawContours(img_contours_debug, [approx], -1, color, 2)
             # Put text label near the first point of the contour
             text_pos = tuple(approx[0][0])
             cv2.putText(img_contours_debug, f"{i}:{len(approx)}pts", text_pos,
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        # Filter contours: check area, number of vertices, convexity, aspect ratio
        if area < min_area:
            break # Stop searching if contours become too small

        if len(approx) == 4 and cv2.isContourConvex(approx):
            # Check aspect ratio of the bounding box as a quick filter
            x, y, w, h = cv2.boundingRect(approx)
            if h == 0: continue # Avoid division by zero
            aspect_ratio = w / float(h)
            # Allow some tolerance for perspective distortion
            if 0.8 < aspect_ratio < 1.2:
                sudoku_contour = approx.reshape(4, 2).astype("float32")
                if debug_dir:
                     # Highlight the chosen contour in green
                     cv2.drawContours(img_contours_debug, [approx], -1, (0, 255, 0), 3)
                break # Found a likely candidate

    if debug_dir:
        cv2.imwrite(str(debug_dir / "04_contours.png"), img_contours_debug)

    if sudoku_contour is None:
        raise ValueError("Could not find a suitable 4-point Sudoku grid contour.")

    return sudoku_contour


def rectify_grid(image, grid_contour, target_size=DEFAULT_RECTIFIED_SIZE):
    """
    Applies a perspective transformation to the image based on the detected
    grid contour to obtain a top-down, squared view of the Sudoku grid.

    Args:
        image (np.ndarray): The original image (BGR or Grayscale).
        grid_contour (np.ndarray): The (4, 2) array of corner points of the grid.
        target_size (int): The desired width and height of the output rectified image.

    Returns:
        np.ndarray: The perspective-corrected, square image of the Sudoku grid.
                    Returns None if transformation fails.
    """
    try:
        # Order the corner points: TL, TR, BR, BL
        ordered_corners = _order_points(grid_contour)

        # Define the target destination points for the perspective transform
        dst_pts = np.array([
            [0, 0],                         # Top-left
            [target_size - 1, 0],           # Top-right
            [target_size - 1, target_size - 1], # Bottom-right
            [0, target_size - 1]            # Bottom-left
        ], dtype="float32")

        # Calculate the perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(ordered_corners, dst_pts)

        # Apply the perspective warp
        rectified = cv2.warpPerspective(image, matrix, (target_size, target_size))
        return rectified
    except Exception as e:
        print(f"[Error] Perspective transformation failed: {e}")
        return None


def split_into_cells(rectified_grid_image):
    """
    Splits the rectified (square) grid image into 81 individual cell images.

    Args:
        rectified_grid_image (np.ndarray): The top-down view of the Sudoku grid.

    Returns:
        tuple: (list[np.ndarray], np.ndarray)
               - A list containing 81 cell images (np.ndarray).
               - The input rectified_grid_image (potentially resized if not square).

    Raises:
        ValueError: If input image is None.
        RuntimeError: If the wrong number of cells is extracted.
    """
    if rectified_grid_image is None:
        raise ValueError("Input rectified_grid_image is None.")

    h, w = rectified_grid_image.shape[:2]

    # Ensure the input is square (it should be after rectification)
    if h != w:
        print(f"[Warning] Rectified grid image is not square ({w}x{h}). Resizing to square.")
        size = max(h, w)
        rectified_grid_image = cv2.resize(rectified_grid_image, (size, size), interpolation=cv2.INTER_AREA)
        h, w = size, size

    current_cell_size = h // GRID_SIZE
    cells = []
    # Define a small margin to exclude grid lines from cell images
    margin_ratio = 0.04 # Percentage of cell size
    margin = int(current_cell_size * margin_ratio)

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            # Calculate cell boundaries with margin
            start_row = r * current_cell_size + margin
            start_col = c * current_cell_size + margin
            end_row = (r + 1) * current_cell_size - margin
            end_col = (c + 1) * current_cell_size - margin

            # Ensure coordinates are within image bounds
            start_row, start_col = max(0, start_row), max(0, start_col)
            end_row, end_col = min(h, end_row), min(w, end_col)

            # Extract the cell image
            if start_row >= end_row or start_col >= end_col:
                # Handle cases where margin makes the cell invalid (e.g., very small images)
                # Create an empty placeholder of expected type
                channels = rectified_grid_image.shape[2] if rectified_grid_image.ndim == 3 else 1
                cell_shape = (current_cell_size, current_cell_size, channels) if channels > 1 else (current_cell_size, current_cell_size)
                cell_img = np.zeros(cell_shape, dtype=rectified_grid_image.dtype)
            else:
                cell_img = rectified_grid_image[start_row:end_row, start_col:end_col]

            cells.append(cell_img)

    if len(cells) != GRID_SIZE * GRID_SIZE:
         # This should ideally not happen if logic is correct
         raise RuntimeError(f"Expected {GRID_SIZE*GRID_SIZE} cells, but extracted {len(cells)}")

    return cells, rectified_grid_image


def extract_cells_from_image(image_path_or_array, target_size=DEFAULT_RECTIFIED_SIZE, debug=False):
    """
    Main function to load an image, find the Sudoku grid, rectify it,
    and extract the individual cells.

    Args:
        image_path_or_array (str | Path | np.ndarray): Path to the image file or a NumPy array containing the image.
        target_size (int): The target size for the rectified grid.
        debug (bool): If True, saves intermediate images from `find_sudoku_grid_contour`.

    Returns:
        tuple: (cells, rectified_grid, grid_contour)
               - cells (list[np.ndarray] | None): List of 81 extracted cell images, or None on failure.
               - rectified_grid (np.ndarray | None): The rectified grid image, or None on failure.
               - grid_contour (np.ndarray | None): The detected 4-point grid contour, or None on failure.
    """
    debug_path = None
    image = None
    try:
        # Load image from path or use array directly
        if isinstance(image_path_or_array, (str, Path)):
            image_path = Path(image_path_or_array)
            if not image_path.is_file():
                 raise FileNotFoundError(f"Image file not found: {image_path}")
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not read image file: {image_path}")
            if debug:
                debug_path = Path(f"debug_extract_{image_path.stem}")
        elif isinstance(image_path_or_array, np.ndarray):
            image = image_path_or_array.copy() # Work on a copy
            if debug:
                debug_path = Path("debug_extract_array")
        else:
            raise TypeError("Input must be a file path (str/Path) or a NumPy array.")

        # 1. Find Grid Contour
        grid_contour = find_sudoku_grid_contour(image, debug_dir=debug_path)

        # 2. Rectify Perspective
        rectified_grid = rectify_grid(image, grid_contour, target_size)
        if rectified_grid is None:
            # Rectification failed, cannot proceed
            raise ValueError("Failed to rectify the Sudoku grid.")

        # 3. Split into Cells
        cells, rectified_grid_maybe_resized = split_into_cells(rectified_grid)

        return cells, rectified_grid_maybe_resized, grid_contour

    except (FileNotFoundError, ValueError, TypeError, RuntimeError) as e:
        print(f"[Error in digit_extractor]: {e}")
        return None, None, None # Return None for all outputs on failure
    except Exception as e:
        # Catch unexpected errors
        print(f"[Unexpected Error in digit_extractor]: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


# --- Example Usage (__main__) ---
if __name__ == "__main__":
    print("Testing DigitExtractor...")
    # Use a default test image if available, otherwise notify user
    test_image_path = "rendered_sudoku_random.png"
    if not Path(test_image_path).exists():
         print(f"Test image '{test_image_path}' not found.")
         print("Please run sudoku_renderer.py to generate it, or provide a path to a real Sudoku image.")
    else:
        print(f"Processing image: {test_image_path}")
        # Run extraction with debug output enabled
        cells, rectified, contour = extract_cells_from_image(test_image_path, debug=True)

        if cells:
            print(f"Successfully extracted {len(cells)} cells.")

            # Save the rectified grid image
            rectified_save_path = "extracted_rectified_grid.png"
            cv2.imwrite(rectified_save_path, rectified)
            print(f"Saved rectified grid to '{rectified_save_path}'")

            # Save a sample of the extracted cells (e.g., first 9)
            save_dir = Path("extracted_cells")
            save_dir.mkdir(exist_ok=True)
            num_saved = 0
            for i, cell_img in enumerate(cells):
                 if i >= 9: break # Limit saved samples
                 if cell_img is not None and cell_img.size > 0:
                     cell_filename = str(save_dir / f"cell_{i//GRID_SIZE}_{i%GRID_SIZE}.png")
                     cv2.imwrite(cell_filename, cell_img)
                     num_saved += 1
            print(f"Saved {num_saved} sample cells to '{save_dir}' directory.")

            # Save the original image with the detected contour overlaid
            original_image = cv2.imread(test_image_path)
            if original_image is not None and contour is not None:
                 overlay_save_path = "extracted_contour_overlay.png"
                 # Ensure contour points are integers for drawing
                 contour_int = contour.astype(int)
                 cv2.drawContours(original_image, [contour_int], -1, (0, 255, 0), 3) # Draw green contour
                 cv2.imwrite(overlay_save_path, original_image)
                 print(f"Saved contour overlay to '{overlay_save_path}'")
        else:
            print("Digit extraction failed for the test image.")

    print("\nExtractor test complete.")

