# digit_extractor.py
import cv2
import numpy as np
from pathlib import Path

# --- Constants ---
GRID_SIZE = 9
DEFAULT_RECTIFIED_SIZE = 450 # Target size for the top-down grid view
# CELL_SIZE = DEFAULT_RECTIFIED_SIZE // GRID_SIZE # Not strictly needed here

# --- Helper Functions ---
# (Keep _order_points as before)
def _order_points(pts):
    """Orders 4 points: top-left, top-right, bottom-right, bottom-left."""
    if pts.shape != (4, 2):
         try: pts = pts.reshape(4, 2)
         except ValueError: raise ValueError(f"Input 'pts' must be reshapeable to (4, 2). Got shape {pts.shape}")
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff_yx = pts[:, 1] - pts[:, 0]
    rect[1] = pts[np.argmin(diff_yx)]
    rect[3] = pts[np.argmax(diff_yx)]
    return rect

# --- Core Grid Finding and Extraction Logic ---

def find_sudoku_grid_contour(image, debug_dir=None):
    """
    Pre-processes the image and returns the 4-point contour of the outer Sudoku grid.
    Raises ValueError if no suitable contour is found.
    """
    if image is None: raise ValueError("Input image is None.")
    if image.ndim == 3: gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else: gray = image.copy() # Work on a copy if already grayscale

    if debug_dir: debug_dir.mkdir(exist_ok=True)

    # --- Preprocessing Improvements ---
    # 1. Blur
    blur = cv2.GaussianBlur(gray, (7, 7), 0) # Kernel size 7x7 seems reasonable
    if debug_dir: cv2.imwrite(str(debug_dir / "01_blur.png"), blur)

    # 2. Adaptive Threshold
    # Increased block size slightly, adjust C if needed
    thresh = cv2.adaptiveThreshold(blur, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, # Invert: grid lines white
                                   15, 4) # Block size 15, C=4
    if debug_dir: cv2.imwrite(str(debug_dir / "02_thresh.png"), thresh)

    # 3. Morphological Closing (to connect broken lines)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2) # More iterations?
    if debug_dir: cv2.imwrite(str(debug_dir / "03_closed.png"), closed)
    # --- End Preprocessing Improvements ---

    # Find contours on the *closed* image
    contours, hierarchy = cv2.findContours(closed,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

    if not contours: raise ValueError("No contours found after preprocessing.")

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    sudoku_contour = None
    min_area = gray.shape[0] * gray.shape[1] * 0.1 # At least 10% of image area

    if debug_dir:
        img_contours_debug = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True) # Epsilon might need tuning

        if debug_dir and i < 10: # Draw first 10 largest contours
             cv2.drawContours(img_contours_debug, [approx], -1, (random.randint(0,255), random.randint(0,255), random.randint(0,255)), 2)
             cv2.putText(img_contours_debug, f"{i}:{len(approx)}", tuple(approx[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)


        if area < min_area: break # Stop early if contours get too small

        if len(approx) == 4 and cv2.isContourConvex(approx):
            # Basic check: aspect ratio of bounding box should be near 1
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 0.8 < aspect_ratio < 1.2: # Allow some tolerance for perspective
                sudoku_contour = approx.reshape(4, 2).astype("float32")
                if debug_dir:
                     cv2.drawContours(img_contours_debug, [approx], -1, (0, 255, 0), 3) # Highlight chosen one in green
                break
            # else:
                # print(f"Skipping 4-point contour with aspect ratio {aspect_ratio:.2f}")


    if debug_dir: cv2.imwrite(str(debug_dir / "04_contours.png"), img_contours_debug)

    if sudoku_contour is None:
        raise ValueError("Could not find a suitable 4-point Sudoku grid contour after filtering.")

    return sudoku_contour


def rectify_grid(image, grid_contour, target_size=DEFAULT_RECTIFIED_SIZE):
    """Applies perspective transform to get a top-down view."""
    ordered_corners = _order_points(grid_contour)
    dst_pts = np.array([
        [0, 0], [target_size - 1, 0],
        [target_size - 1, target_size - 1], [0, target_size - 1]
    ], dtype="float32")
    matrix = cv2.getPerspectiveTransform(ordered_corners, dst_pts)
    rectified = cv2.warpPerspective(image, matrix, (target_size, target_size))
    return rectified

def split_into_cells(rectified_grid_image):
    """Splits the rectified grid image into 81 individual cell images."""
    if rectified_grid_image is None: raise ValueError("Input rectified_grid_image is None.")
    h, w = rectified_grid_image.shape[:2]
    if h != w: # Should be square after rectify_grid
        size = max(h, w)
        rectified_grid_image = cv2.resize(rectified_grid_image, (size, size))
        h, w = size, size

    current_cell_size = h // GRID_SIZE
    cells = []
    # Reduced margin slightly, maybe 3%?
    margin = int(current_cell_size * 0.04)

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            start_row = r * current_cell_size + margin
            start_col = c * current_cell_size + margin
            end_row = (r + 1) * current_cell_size - margin
            end_col = (c + 1) * current_cell_size - margin
            start_row, start_col = max(0, start_row), max(0, start_col)
            end_row, end_col = min(h, end_row), min(w, end_col)

            if start_row >= end_row or start_col >= end_col:
                cell_img = np.zeros((current_cell_size, current_cell_size, rectified_grid_image.shape[2] if rectified_grid_image.ndim == 3 else 1), dtype=np.uint8)
                if rectified_grid_image.ndim == 2: # Ensure correct shape if grayscale
                     cell_img = cell_img.squeeze()
            else:
                cell_img = rectified_grid_image[start_row:end_row, start_col:end_col]

            cells.append(cell_img)

    if len(cells) != GRID_SIZE * GRID_SIZE:
         raise RuntimeError(f"Expected {GRID_SIZE*GRID_SIZE} cells, but extracted {len(cells)}")

    # Return the grid itself as well, might be useful
    return cells, rectified_grid_image

def extract_cells_from_image(image_path_or_array, target_size=DEFAULT_RECTIFIED_SIZE, debug=False):
    """
    Main function: loads image, finds grid, rectifies, extracts cells.
    Args:
        debug (bool): If True, saves intermediate images for debugging contour finding.
    """
    debug_path = None
    try:
        if isinstance(image_path_or_array, (str, Path)):
            image_path = Path(image_path_or_array)
            image = cv2.imread(str(image_path))
            if image is None: raise FileNotFoundError(f"Could not read image: {image_path}")
            if debug: debug_path = Path(f"debug_extract_{image_path.stem}")
        elif isinstance(image_path_or_array, np.ndarray):
            image = image_path_or_array.copy()
            if debug: debug_path = Path("debug_extract_array")
        else:
            raise TypeError("Input must be a file path or a NumPy array.")

        # 1. Find Grid Contour (pass debug path if enabled)
        grid_contour = find_sudoku_grid_contour(image, debug_dir=debug_path)

        # 2. Rectify Perspective
        rectified_grid = rectify_grid(image, grid_contour, target_size)

        # 3. Split into Cells
        cells, _ = split_into_cells(rectified_grid)

        return cells, rectified_grid, grid_contour

    except (FileNotFoundError, ValueError, TypeError, RuntimeError) as e:
        print(f"[Error in digit_extractor]: {e}")
        return None, None, None
    except Exception as e:
        print(f"[Unexpected Error in digit_extractor]: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

# (Keep __main__ test block as before, maybe add debug=True to the call)
if __name__ == "__main__":
    print("Testing DigitExtractor...")
    test_image_path = "rendered_sudoku_specific.png" # Or use a real image
    if not Path(test_image_path).exists():
         print(f"Test image '{test_image_path}' not found. Run sudoku_renderer.py or provide a real image path.")
    else:
        print(f"Processing image: {test_image_path}")
        # --- Enable debug output for extractor ---
        cells, rectified, contour = extract_cells_from_image(test_image_path, debug=True)
        # ---

        if cells:
            print(f"Successfully extracted {len(cells)} cells.")
            # ... (rest of the saving logic from previous version) ...
            cv2.imwrite("extracted_rectified_grid.png", rectified)
            print("Saved rectified grid to extracted_rectified_grid.png")
            save_dir = Path("extracted_cells")
            save_dir.mkdir(exist_ok=True)
            for i, cell_img in enumerate(cells[:9]):
                 if cell_img is not None and cell_img.size > 0:
                     cv2.imwrite(str(save_dir / f"cell_{i//GRID_SIZE}_{i%GRID_SIZE}.png"), cell_img)
            print(f"Saved sample cells to '{save_dir}' directory.")
            original_image = cv2.imread(test_image_path)
            if original_image is not None and contour is not None:
                 cv2.drawContours(original_image, [contour.astype(int)], -1, (0, 255, 0), 3)
                 cv2.imwrite("extracted_contour_overlay.png", original_image)
                 print("Saved contour overlay to extracted_contour_overlay.png")
        else:
            print("Digit extraction failed.")
    print("\nExtractor test complete.")