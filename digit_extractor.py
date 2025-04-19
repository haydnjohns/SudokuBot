# digit_extractor.py
import os
from pathlib import Path

import cv2
import numpy as np

def _order_points(pts):
    """Orders 4 points: top-left, top-right, bottom-right, bottom-left."""
    pts = np.array(pts, dtype="float32")
    if pts.shape != (4, 2):
         try:
              pts = pts.reshape(4, 2)
         except ValueError:
              raise ValueError(f"Input 'pts' must be reshapeable to (4, 2). Got shape {pts.shape}")

    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Top-left
    rect[2] = pts[np.argmax(s)] # Bottom-right
    diff = np.diff(pts, axis=1) # y - x
    rect[1] = pts[np.argmin(diff)] # Top-right
    rect[3] = pts[np.argmax(diff)] # Bottom-left
    return rect

def find_sudoku_grid_contour(image):
    """
    Pre-processes the image and returns the 4-point contour of the Sudoku grid.

    Args:
        image (np.ndarray): Input BGR image.

    Returns:
        np.ndarray: 4x2 float32 array of corner points, or None if not found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0) # Slightly less blur than before?
    thresh = cv2.adaptiveThreshold(blur, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 3) # Invert

    # Optional: Morphological operations to clean up thresholding
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel) # Close small gaps

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # print("Warning: No contours found.")
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Find the largest quadrilateral contour
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True) # Adjust epsilon if needed

        if len(approx) == 4:
            # Check if the contour area is reasonably large
            min_area_ratio = 0.1 # Require contour to be at least 10% of image area
            if cv2.contourArea(approx) > image.shape[0] * image.shape[1] * min_area_ratio:
                 return approx.reshape(4, 2).astype("float32")

    # print("Warning: No suitable 4-point contour found.")
    return None


def rectify_grid(image, contour, target_size=450):
    """
    Applies perspective transform to get a top-down view of the grid.

    Args:
        image (np.ndarray): Original BGR image.
        contour (np.ndarray): 4x2 float32 array of grid corners.
        target_size (int): Desired size (width and height) of the rectified grid.

    Returns:
        np.ndarray: The rectified BGR image, or None if contour is invalid.
    """
    if contour is None or contour.shape != (4, 2):
        return None

    ordered_corners = _order_points(contour)

    dst_pts = np.array([
        [0, 0],
        [target_size - 1, 0],
        [target_size - 1, target_size - 1],
        [0, target_size - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(ordered_corners, dst_pts)
    rectified = cv2.warpPerspective(image, matrix, (target_size, target_size))
    return rectified


def extract_cells_from_rectified(rectified_grid, cell_border_fraction=0.05):
    """
    Splits the rectified grid image into 81 cell images.

    Args:
        rectified_grid (np.ndarray): Top-down view of the Sudoku grid (BGR).
        cell_border_fraction (float): Fraction of cell size to remove from each border
                                      to reduce impact of grid lines.

    Returns:
        list[np.ndarray]: A list of 81 BGR cell images (row-major order).
                          Returns empty list if input is invalid.
    """
    if rectified_grid is None or rectified_grid.ndim != 3:
        return []

    h, w = rectified_grid.shape[:2]
    if h == 0 or w == 0 or h != w: # Expect square grid
        print(f"Warning: Invalid rectified grid shape {rectified_grid.shape}")
        return []

    cell_size = h // 9
    cells = []

    border_h = int(cell_size * cell_border_fraction)
    border_w = int(cell_size * cell_border_fraction)

    for r in range(9):
        for c in range(9):
            y0 = r * cell_size + border_h
            y1 = (r + 1) * cell_size - border_h
            x0 = c * cell_size + border_w
            x1 = (c + 1) * cell_size - border_w

            # Ensure coordinates are valid
            y0, x0 = max(0, y0), max(0, x0)
            y1, x1 = min(h, y1), min(w, x1)

            if y1 <= y0 or x1 <= x0: # Check if cell is valid after border removal
                # Fallback: use full cell if border makes it invalid
                y0_fb = r * cell_size
                y1_fb = (r + 1) * cell_size
                x0_fb = c * cell_size
                x1_fb = (c + 1) * cell_size
                cell_img = rectified_grid[y0_fb:y1_fb, x0_fb:x1_fb]
                # print(f"Warning: Cell ({r},{c}) too small after border removal, using full cell.")
            else:
                cell_img = rectified_grid[y0:y1, x0:x1]

            if cell_img is None or cell_img.size == 0:
                 # Add a placeholder (e.g., small black square) if extraction fails
                 print(f"Warning: Failed to extract cell ({r},{c}), adding placeholder.")
                 cell_img = np.zeros((max(1, cell_size // 2), max(1, cell_size // 2), 3), dtype=np.uint8)


            cells.append(cell_img)

    if len(cells) != 81:
        print(f"Warning: Expected 81 cells, but extracted {len(cells)}")
        # Pad with placeholders if necessary? Or return empty? Let's return what we have.

    return cells


def extract_digits(image_path_or_array, target_grid_size=450, cell_border_frac=0.05):
    """
    High-level function to find grid, rectify, and extract cells from an image.

    Args:
        image_path_or_array (str | np.ndarray): Path to the image or the image array.
        target_grid_size (int): Size for the rectified grid.
        cell_border_frac (float): Border fraction to remove from cells.

    Returns:
        tuple: (cells, rectified_grid, contour)
            - cells (list[np.ndarray]): List of 81 extracted BGR cell images. Empty list on failure.
            - rectified_grid (np.ndarray | None): The 450x450 rectified grid image. None on failure.
            - contour (np.ndarray | None): The 4x2 contour found in the original image. None on failure.
    """
    if isinstance(image_path_or_array, str) or isinstance(image_path_or_array, Path):
        image = cv2.imread(str(image_path_or_array))
        if image is None:
            print(f"Error: Could not load image {image_path_or_array}")
            return [], None, None
    elif isinstance(image_path_or_array, np.ndarray):
        image = image_path_or_array
    else:
        raise TypeError("Input must be a file path (str/Path) or a numpy array.")

    contour = find_sudoku_grid_contour(image)
    if contour is None:
        # print("Extraction failed: Could not find grid contour.")
        return [], None, None

    rectified = rectify_grid(image, contour, target_size=target_grid_size)
    if rectified is None:
        # print("Extraction failed: Could not rectify grid.")
        return [], None, contour

    cells = extract_cells_from_rectified(rectified, cell_border_fraction=cell_border_frac)
    if not cells:
        # print("Extraction failed: Could not extract cells from rectified grid.")
        return [], rectified, contour # Return rectified grid even if cell extraction failed

    return cells, rectified, contour


# --- Example Usage ---
if __name__ == "__main__":
    # Assumes sudoku_renderer.py is in the same directory and can generate an image
    try:
        from sudoku_renderer import SudokuRenderer
        print("[INFO] Using SudokuRenderer to generate a test image...")
        renderer = SudokuRenderer(warp_intensity=0.2)
        test_img, _, _ = renderer.render_sudoku(difficulty=0.5)
        cv2.imwrite("temp_extractor_test.png", test_img)
        img_path = "temp_extractor_test.png"
    except ImportError:
        print("[WARN] SudokuRenderer not found. Please provide a test image path.")
        # Try a default path if the generated one isn't available
        img_path = "sample_images/digit_5_img_0.png" # Use one from previous example if exists
        if not Path(img_path).exists():
             print(f"[ERROR] Test image {img_path} not found. Extractor cannot be tested.")
             exit()


    print(f"[INFO] Running digit extraction on: {img_path}")
    extracted_cells, rectified_image, found_contour = extract_digits(img_path)

    if extracted_cells and rectified_image is not None:
        print(f"[INFO] Successfully extracted {len(extracted_cells)} cells.")

        # Display the rectified grid
        cv2.imshow("Rectified Grid", rectified_image)
        cv2.waitKey(10) # Short wait

        # Display a few extracted cells
        print("[INFO] Displaying first 9 extracted cells...")
        cell_display_size = 100
        first_row_img = np.zeros((cell_display_size, cell_display_size * 9, 3), dtype=np.uint8)
        for i in range(9):
            if i < len(extracted_cells):
                cell = extracted_cells[i]
                if cell is not None and cell.size > 0:
                     resized_cell = cv2.resize(cell, (cell_display_size, cell_display_size))
                     first_row_img[:, i*cell_display_size:(i+1)*cell_display_size] = resized_cell
                else:
                     # Draw black square if cell is None or empty
                     cv2.rectangle(first_row_img, (i*cell_display_size, 0), ((i+1)*cell_display_size, cell_display_size), (0,0,0), -1)


        cv2.imshow("First 9 Extracted Cells", first_row_img)
        print("[INFO] Press any key to close cell windows.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("[ERROR] Digit extraction failed.")
        if found_contour is not None:
             print("   - Grid contour was found, but rectification or cell splitting failed.")
             # Optionally display original image with contour
             orig_image = cv2.imread(img_path)
             cv2.drawContours(orig_image, [found_contour.astype(int)], -1, (0, 255, 0), 2)
             cv2.imshow("Contour Found (Extraction Failed)", orig_image)
             cv2.waitKey(0)
             cv2.destroyAllWindows()
        else:
             print("   - Grid contour was not found.")

    # Clean up temp file
    if Path("temp_extractor_test.png").exists():
        os.remove("temp_extractor_test.png")

    print("[INFO] Extractor testing complete.")