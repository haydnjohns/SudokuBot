# digit_extractor.py
import cv2
import numpy as np
from pathlib import Path

# --- Constants ---
GRID_SIZE = 9
DEFAULT_RECTIFIED_SIZE = 450 # Target size for the top-down grid view
CELL_SIZE = DEFAULT_RECTIFIED_SIZE // GRID_SIZE

# --- Helper Functions ---
def _order_points(pts):
    """Orders 4 points: top-left, top-right, bottom-right, bottom-left."""
    # Ensure input is shape (4, 2)
    if pts.shape != (4, 2):
         try:
              pts = pts.reshape(4, 2)
         except ValueError:
              raise ValueError(f"Input 'pts' must be reshapeable to (4, 2). Got shape {pts.shape}")

    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1) # Correct axis for difference calculation (y-x or x-y depending on order)
    # Let's use the original logic which worked: diff = p[1] - p[0] -> use np.array([p[1] - p[0] for p in pts])
    # Or simply use the argmin/argmax of the difference along axis 1
    diff_np = np.diff(pts, axis=1) # Shape (4, 1)
    rect[1] = pts[np.argmin(diff_np)] # Top-right has smallest y-x (or largest x-y)
    rect[3] = pts[np.argmax(diff_np)] # Bottom-left has largest y-x (or smallest x-y)

    # Re-evaluate the diff logic based on the original code's apparent success
    # Original code used: diff = np.array([p[1] - p[0] for p in pts])
    # This calculates y - x for each point.
    # Top-right should have the smallest y - x (large x, small y)
    # Bottom-left should have the largest y - x (small x, large y)
    diff_yx = pts[:, 1] - pts[:, 0]
    rect[1] = pts[np.argmin(diff_yx)]
    rect[3] = pts[np.argmax(diff_yx)]

    return rect

def find_sudoku_grid_contour(image):
    """
    Pre-processes the image and returns the 4-point contour of the outer Sudoku grid.
    Raises ValueError if no suitable contour is found.
    """
    if image is None:
        raise ValueError("Input image is None.")
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Preprocessing steps (similar to original ImageProcessing.py)
    # Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (7, 7), 0) # Kernel size might need tuning

    # Adaptive thresholding to handle varying lighting
    thresh = cv2.adaptiveThreshold(blur, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, # Invert: grid lines become white
                                   11, 2) # Block size and C value might need tuning

    # Optional: Morphological operations to close gaps / remove noise
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1) # Close gaps
    # thresh = cv2.dilate(thresh, kernel, iterations=1) # Slightly thicken lines

    # Find contours
    contours, hierarchy = cv2.findContours(thresh,
                                           cv2.RETR_EXTERNAL, # Get only outer contours
                                           cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No contours found in image after preprocessing.")

    # Sort contours by area (descending) and find the largest quadrilateral
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    sudoku_contour = None
    min_area = gray.shape[0] * gray.shape[1] * 0.1 # Require contour to be at least 10% of image area

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            # print(f"Skipping contour with area {area} (too small)")
            continue # Skip small contours early

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True) # Adjust epsilon (0.01-0.05)

        # print(f"Contour area: {area}, Approx vertices: {len(approx)}") # Debugging

        if len(approx) == 4:
            # Check if it's reasonably convex and square-like
            if cv2.isContourConvex(approx):
                sudoku_contour = approx.reshape(4, 2).astype("float32")
                # print(f"Found potential 4-point contour with area {area}")
                break # Found the largest quadrilateral
            # else:
                # print("Skipping non-convex 4-point contour")

    if sudoku_contour is None:
        # Fallback: Try RETR_LIST if RETR_EXTERNAL failed? Sometimes grid isn't outermost.
        # contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # for cnt in contours: ... (repeat check)
        # If still not found:
        raise ValueError("Could not find a suitable 4-point Sudoku grid contour.")

    return sudoku_contour


def rectify_grid(image, grid_contour, target_size=DEFAULT_RECTIFIED_SIZE):
    """
    Applies perspective transform to get a top-down view of the Sudoku grid.
    """
    ordered_corners = _order_points(grid_contour)

    # Destination points for the rectified square grid
    dst_pts = np.array([
        [0, 0],
        [target_size - 1, 0],
        [target_size - 1, target_size - 1],
        [0, target_size - 1]
    ], dtype="float32")

    # Compute the perspective transform matrix and apply it
    matrix = cv2.getPerspectiveTransform(ordered_corners, dst_pts)
    rectified = cv2.warpPerspective(image, matrix, (target_size, target_size))

    return rectified

def split_into_cells(rectified_grid_image):
    """
    Splits the rectified grid image into 81 individual cell images.
    """
    if rectified_grid_image is None:
        raise ValueError("Input rectified_grid_image is None.")

    h, w = rectified_grid_image.shape[:2]
    if h != w:
        # This shouldn't happen if rectify_grid worked correctly
        print(f"[Warning] Rectified grid image is not square ({h}x{w}). Resizing to square.")
        size = max(h, w)
        rectified_grid_image = cv2.resize(rectified_grid_image, (size, size))
        h, w = size, size

    current_cell_size = h // GRID_SIZE # Integer division
    cells = []

    # Add a small margin removal to avoid capturing grid lines within cells
    margin = int(current_cell_size * 0.05) # Remove 5% from each side

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            start_row = r * current_cell_size + margin
            start_col = c * current_cell_size + margin
            end_row = (r + 1) * current_cell_size - margin
            end_col = (c + 1) * current_cell_size - margin

            # Ensure indices are valid
            start_row, start_col = max(0, start_row), max(0, start_col)
            end_row, end_col = min(h, end_row), min(w, end_col)

            if start_row >= end_row or start_col >= end_col:
                # If margin is too large, add an empty placeholder (or handle error)
                print(f"[Warning] Cell ({r},{c}) has invalid dimensions after margin removal. Adding empty cell.")
                cell_img = np.zeros((CELL_SIZE, CELL_SIZE, rectified_grid_image.shape[2] if rectified_grid_image.ndim == 3 else 1), dtype=np.uint8)
            else:
                cell_img = rectified_grid_image[start_row:end_row, start_col:end_col]

            cells.append(cell_img)

    if len(cells) != GRID_SIZE * GRID_SIZE:
         raise RuntimeError(f"Expected {GRID_SIZE*GRID_SIZE} cells, but extracted {len(cells)}")

    return cells, rectified_grid_image # Return rectified grid as well

def extract_cells_from_image(image_path_or_array, target_size=DEFAULT_RECTIFIED_SIZE):
    """
    Main function to load an image, find the grid, rectify it, and extract cells.

    Args:
        image_path_or_array: Path to the image file or a NumPy array (BGR).
        target_size: The desired size of the rectified grid image.

    Returns:
        tuple: (list_of_cells, rectified_grid, grid_contour)
               - list_of_cells: List of 81 NumPy arrays, each representing a cell.
               - rectified_grid: NumPy array of the top-down view of the grid.
               - grid_contour: The 4-point contour found in the original image.
            Returns (None, None, None) if any step fails.
    """
    try:
        if isinstance(image_path_or_array, (str, Path)):
            image = cv2.imread(str(image_path_or_array))
            if image is None:
                raise FileNotFoundError(f"Could not read image: {image_path_or_array}")
        elif isinstance(image_path_or_array, np.ndarray):
            image = image_path_or_array.copy() # Work on a copy
        else:
            raise TypeError("Input must be a file path or a NumPy array.")

        # 1. Find Grid Contour
        grid_contour = find_sudoku_grid_contour(image)
        # print("Found grid contour.") # Debug

        # 2. Rectify Perspective
        rectified_grid = rectify_grid(image, grid_contour, target_size)
        # print("Rectified grid.") # Debug

        # 3. Split into Cells
        cells, _ = split_into_cells(rectified_grid) # We already have rectified_grid
        # print(f"Extracted {len(cells)} cells.") # Debug

        return cells, rectified_grid, grid_contour

    except (FileNotFoundError, ValueError, TypeError, RuntimeError) as e:
        print(f"[Error in digit_extractor]: {e}")
        return None, None, None
    except Exception as e: # Catch unexpected errors
        print(f"[Unexpected Error in digit_extractor]: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


# --- Example Usage (for testing) ---
if __name__ == "__main__":
    print("Testing DigitExtractor...")
    # Assumes you have run sudoku_renderer.py and have the output images
    test_image_path = "rendered_sudoku_specific.png"
    if not Path(test_image_path).exists():
         print(f"Test image '{test_image_path}' not found. Please run sudoku_renderer.py first.")
    else:
        print(f"Processing image: {test_image_path}")
        cells, rectified, contour = extract_cells_from_image(test_image_path)

        if cells:
            print(f"Successfully extracted {len(cells)} cells.")

            # Save the rectified grid
            cv2.imwrite("extracted_rectified_grid.png", rectified)
            print("Saved rectified grid to extracted_rectified_grid.png")

            # Save a few extracted cells for inspection
            save_dir = Path("extracted_cells")
            save_dir.mkdir(exist_ok=True)
            for i, cell_img in enumerate(cells[:9]): # Save first row
                 if cell_img is not None and cell_img.size > 0:
                     cv2.imwrite(str(save_dir / f"cell_{i//GRID_SIZE}_{i%GRID_SIZE}.png"), cell_img)
                 else:
                     print(f"Cell {i} is empty or invalid.")
            print(f"Saved sample cells to '{save_dir}' directory.")

            # Optional: Draw contour on original image
            original_image = cv2.imread(test_image_path)
            cv2.drawContours(original_image, [contour.astype(int)], -1, (0, 255, 0), 3)
            cv2.imwrite("extracted_contour_overlay.png", original_image)
            print("Saved contour overlay to extracted_contour_overlay.png")

        else:
            print("Digit extraction failed.")

    print("\nExtractor test complete.")