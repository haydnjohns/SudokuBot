"""
Find and extract Sudoku grid cells from an image.
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

GRID_SIZE = 9
# Changed default size to be divisible by common CNN strides (e.g., 2^3=8 or 3^3=27)
# 252 = 9 * 28 -> divisible by 2, 3, 4, 6, 7, 9, 12, 14, 18, 21, 28...
DEFAULT_RECTIFIED_SIZE = 252


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype("float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Top-left has smallest sum
    rect[2] = pts[np.argmax(s)] # Bottom-right has largest sum
    diff = np.diff(pts, axis=1) # diff = y - x
    rect[1] = pts[np.argmin(diff)] # Top-right has smallest diff
    rect[3] = pts[np.argmax(diff)] # Bottom-left has largest diff
    return rect


def find_sudoku_grid_contour(
    img: np.ndarray,
    debug_dir: Optional[Path] = None
) -> Optional[np.ndarray]:
    """
    Find the largest 4-point contour approximating the Sudoku grid boundary.
    Returns the contour points (4, 2) or None if not found.
    """
    if img is None or img.size == 0:
        print("[Contour Finder] Invalid input image.")
        return None

    gray = (
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.ndim == 3 and img.shape[2] == 3 else img.copy()
    )
    if gray.ndim == 3: # Handle case like RGBA input
        gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / "00_gray.png"), gray)

    # Preprocessing: Blur and Threshold
    # GaussianBlur helps reduce noise before thresholding
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    if debug_dir: cv2.imwrite(str(debug_dir / "01_blur.png"), blur)

    # Adaptive thresholding is generally robust to lighting changes
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # Method
        cv2.THRESH_BINARY_INV,         # Threshold type (invert to get black lines on white)
        15,                            # Block size (must be odd) - adjust based on image size/line thickness
        4                              # Constant C subtracted from mean
    )
    if debug_dir: cv2.imwrite(str(debug_dir / "02_thresh.png"), thresh)

    # Morphological closing to connect broken lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    if debug_dir: cv2.imwrite(str(debug_dir / "03_closed.png"), closed)

    # Find contours
    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_EXTERNAL,        # Retrieve only outer contours
        cv2.CHAIN_APPROX_SIMPLE   # Compress horizontal/vertical segments
    )

    if not contours:
        print("[Contour Finder] No contours found.")
        return None

    # Sort contours by area (descending) and filter small ones
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    min_area = gray.size * 0.05 # Require contour to be at least 5% of image area

    if debug_dir:
        img_contours = img.copy() if img.ndim == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 1) # Draw all contours found
        cv2.imwrite(str(debug_dir / "04_all_contours.png"), img_contours)


    # Find the largest quadrilateral contour that resembles a square
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area:
            # print(f"[Contour Finder] Remaining contours too small (area {area} < {min_area}).")
            break # No need to check smaller contours

        peri = cv2.arcLength(cnt, True)
        # Approximate the contour shape to a polygon
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True) # Epsilon factor determines accuracy

        # Check if the approximation has 4 vertices and is convex
        if len(approx) == 4 and cv2.isContourConvex(approx):
            # Check aspect ratio of the bounding box
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h) if h > 0 else 0

            # Allow some tolerance for perspective distortion
            if 0.7 < aspect_ratio < 1.3:
                if debug_dir:
                    img_found = img.copy() if img.ndim == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    cv2.drawContours(img_found, [approx], -1, (0, 0, 255), 3)
                    cv2.imwrite(str(debug_dir / f"05_found_contour_{i}.png"), img_found)
                # print(f"[Contour Finder] Found potential grid contour (index {i}, area {area:.0f}, aspect {aspect_ratio:.2f}).")
                return approx.reshape(4, 2).astype("float32") # Return the corner points

    print("[Contour Finder] No suitable Sudoku grid contour found.")
    return None


def rectify_grid(
    img: np.ndarray,
    contour: np.ndarray,
    size: int = DEFAULT_RECTIFIED_SIZE
) -> Optional[np.ndarray]:
    """Perspective-correct the Sudoku grid to a square of given size."""
    if contour is None or contour.shape != (4, 2):
        print("[Rectifier] Invalid contour provided.")
        return None
    if img is None or img.size == 0:
        print("[Rectifier] Invalid image provided.")
        return None

    try:
        # Order the contour points: TL, TR, BR, BL
        src_pts = _order_points(contour)

        # Define the destination points for the square image
        dst_pts = np.array([
            [0, 0],             # Top-left
            [size - 1, 0],      # Top-right
            [size - 1, size - 1], # Bottom-right
            [0, size - 1],      # Bottom-left
        ], dtype="float32")

        # Calculate the perspective transform matrix
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # Apply the perspective warp
        warped = cv2.warpPerspective(img, matrix, (size, size))
        return warped
    except Exception as e:
        print(f"[Rectifier] Error during perspective warp: {e}")
        return None


def split_into_cells(
    rectified: np.ndarray
) -> Tuple[Optional[List[np.ndarray]], Optional[np.ndarray]]:
    """
    Split the rectified grid into GRID_SIZE × GRID_SIZE cell images.
    Applies a small margin removal.
    Returns (cells, rectified_image) or (None, None) on failure.
    """
    if rectified is None or rectified.shape[0] != rectified.shape[1]:
        print("[Splitter] Invalid rectified image provided.")
        return None, None

    h, w = rectified.shape[:2]
    if h == 0 or w == 0:
        print("[Splitter] Rectified image has zero dimension.")
        return None, None

    cell_sz_h = h // GRID_SIZE
    cell_sz_w = w // GRID_SIZE
    if cell_sz_h == 0 or cell_sz_w == 0:
        print("[Splitter] Calculated cell size is zero.")
        return None, None

    # Calculate margin to remove grid lines (e.g., 4% of cell size)
    margin_y = max(1, int(cell_sz_h * 0.04))
    margin_x = max(1, int(cell_sz_w * 0.04))

    cells: List[np.ndarray] = []
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            # Calculate cell boundaries with margin
            y0 = row * cell_sz_h + margin_y
            x0 = col * cell_sz_w + margin_x
            y1 = (row + 1) * cell_sz_h - margin_y
            x1 = (col + 1) * cell_sz_w - margin_x

            # Ensure coordinates are valid and extract cell
            if y0 < y1 and x0 < x1:
                cell = rectified[y0:y1, x0:x1]
                cells.append(cell)
            else:
                # Handle cases where margin is too large for cell size (should be rare)
                # Fallback: extract cell without margin
                y0_nomargin = row * cell_sz_h
                x0_nomargin = col * cell_sz_w
                y1_nomargin = (row + 1) * cell_sz_h
                x1_nomargin = (col + 1) * cell_sz_w
                cell = rectified[y0_nomargin:y1_nomargin, x0_nomargin:x1_nomargin]
                # Add a blank cell if even no-margin extraction fails
                if cell is None or cell.size == 0:
                     cell_shape = (cell_sz_h, cell_sz_w) + rectified.shape[2:] if rectified.ndim > 2 else (cell_sz_h, cell_sz_w)
                     cell = np.zeros(cell_shape, rectified.dtype) # Create blank cell
                cells.append(cell)


    if len(cells) != GRID_SIZE * GRID_SIZE:
        print(f"[Splitter] Incorrect number of cells extracted: {len(cells)}")
        return None, rectified # Return None for cells but keep rectified image

    return cells, rectified


def extract_cells_from_image(
    img_or_path,
    size: int = DEFAULT_RECTIFIED_SIZE,
    debug: bool = False
) -> Tuple[Optional[List[np.ndarray]], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    High-level function: read image, find grid, rectify, and split into cells.
    Returns (cells, rectified_image, contour) or (None, None, None) on failure.
    """
    if isinstance(img_or_path, (str, Path)):
        img_path = str(img_or_path)
        if not Path(img_path).exists():
            print(f"[Extractor] Image file not found: {img_path}")
            return None, None, None
        img = cv2.imread(img_path)
        if img is None:
            print(f"[Extractor] Failed to read image: {img_path}")
            return None, None, None
    elif isinstance(img_or_path, np.ndarray):
        img = img_or_path.copy()
    else:
        print("[Extractor] Invalid input type (must be path or numpy array).")
        return None, None, None

    if img.size == 0:
        print("[Extractor] Input image is empty.")
        return None, None, None

    try:
        debug_dir = Path(f"debug_extract_{Path(img_path).stem}_{os.getpid()}") if debug and isinstance(img_or_path, (str, Path)) else None
        if debug and not debug_dir: debug_dir = Path(f"debug_extract_np_{os.getpid()}")

        # 1. Find Grid Contour
        contour = find_sudoku_grid_contour(img, debug_dir)
        if contour is None:
            print("[Extractor] Failed to find Sudoku contour.")
            return None, None, None # Contour finding failed

        # 2. Rectify Grid
        rectified = rectify_grid(img, contour, size=size)
        if rectified is None:
            print("[Extractor] Failed to rectify grid.")
            return None, None, contour # Rectification failed, return contour found

        if debug_dir:
            cv2.imwrite(str(debug_dir / "06_rectified.png"), rectified)

        # 3. Split into Cells
        cells, rectified_output = split_into_cells(rectified)
        if cells is None:
            print("[Extractor] Failed to split rectified grid into cells.")
            # Return rectified image even if splitting fails, maybe useful for debugging
            return None, rectified_output, contour

        if debug_dir:
            # Save a montage of extracted cells
            cell_h, cell_w = cells[0].shape[:2]
            montage = np.zeros((GRID_SIZE * cell_h, GRID_SIZE * cell_w) + cells[0].shape[2:], dtype=cells[0].dtype)
            for i, cell_img in enumerate(cells):
                r, c = divmod(i, GRID_SIZE)
                if cell_img.shape[:2] == (cell_h, cell_w): # Ensure shape consistency
                    montage[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = cell_img
            cv2.imwrite(str(debug_dir / "07_cells_montage.png"), montage)


        # Success
        return cells, rectified_output, contour

    except Exception as e:
        print(f"[Extractor] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None
