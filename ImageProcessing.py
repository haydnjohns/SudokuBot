import sys
from pathlib import Path

import cv2
import numpy as np

from digit_classifier import DigitClassifier

# --------------------------------------------------------------------------- #
# 1. helper utilities
# --------------------------------------------------------------------------- #

def load_image(image_path):
    """Load an image from disk; raise if it cannot be opened."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return img

def _order_points(pts):
    """
    Return 4 points ordered: top‑left, top‑right, bottom‑right, bottom‑left.
    """
    pts = np.array(pts, dtype="float32")
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered

# --------------------------------------------------------------------------- #
# 2. sudoku grid localisation
# --------------------------------------------------------------------------- #

def find_sudoku_grid_contour(image):
    """
    Pre‑process the picture and return the 4‑point contour of the outer Sudoku
    grid as float32. Raises ValueError if no suitable contour is found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blur, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in image")

    # pick the largest 4‑sided contour
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype("float32")
    raise ValueError("Could not find a 4‑point Sudoku grid contour")

# --------------------------------------------------------------------------- #
# 3. perspective transform
# --------------------------------------------------------------------------- #

def warp_perspective_transform(image, grid_contour, target_size=450):
    """
    Apply a bird’s‐eye warp so that the Sudoku board becomes a perfect
    square of side length `target_size` pixels.
    """
    src = _order_points(grid_contour)
    dst = np.array([[0, 0],
                    [target_size-1, 0],
                    [target_size-1, target_size-1],
                    [0, target_size-1]],
                   dtype="float32")
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (target_size, target_size))

# --------------------------------------------------------------------------- #
# 4. split and recognise
# --------------------------------------------------------------------------- #

def extract_and_recognize_digits(grid_image, classifier):
    """
    Split the warped grid into 81 cells, recognise each digit
    using the provided classifier, and return a 9×9 NumPy array.
    """
    if grid_image.ndim == 3:
        gray = cv2.cvtColor(grid_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = grid_image

    h, w = gray.shape
    cell_h, cell_w = h // 9, w // 9
    board = np.zeros((9, 9), dtype=int)

    for r in range(9):
        for c in range(9):
            y0, y1 = r*cell_h, (r+1)*cell_h
            x0, x1 = c*cell_w, (c+1)*cell_w
            cell = gray[y0:y1, x0:x1]
            board[r, c] = classifier.recognise(cell)

    return board

# --------------------------------------------------------------------------- #
# 5. main
# --------------------------------------------------------------------------- #

def main(path):
    original = load_image(path)
    contour = find_sudoku_grid_contour(original)
    warped  = warp_perspective_transform(original, contour, target_size=450)

    classifier = DigitClassifier()
    board = extract_and_recognize_digits(warped, classifier)

    print("Detected board (0 = empty):")
    print(board)

    # Optional: show the warped grid
    cv2.imshow("Warped Sudoku", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else "sample_images/digit_5_img_0.png"
    main(img_path)