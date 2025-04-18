#!/usr/bin/env python3
"""
End‑to‑end Sudoku grid detection and digit recognition.

usage:  python sudoku_reader.py  sudoku_image.png
"""

import sys

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# 1. helper utilities
# --------------------------------------------------------------------------- #

def load_image(image_path):
    """Load an image from disk; raise if it cannot be opened."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def _order_points(pts):
    """
    Return 4 points ordered:  top‑left, top‑right, bottom‑right, bottom‑left.
    The incoming `pts` may be in any order.
    """
    pts = np.array(pts, dtype="float32")
    s   = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = pts[np.argmin(s)]        # top‑left  = smallest sum
    ordered[2] = pts[np.argmax(s)]        # bottom‑right = largest sum
    ordered[1] = pts[np.argmin(diff)]     # top‑right = smallest diff
    ordered[3] = pts[np.argmax(diff)]     # bottom‑left = largest diff
    return ordered


# --------------------------------------------------------------------------- #
# 2. sudoku grid localisation
# --------------------------------------------------------------------------- #

def find_sudoku_grid_contour(image):
    """
    Pre‑process the picture and return the 4‑point contour of the outer Sudoku
    grid (as float32).  Raises ValueError if no suitable contour can be found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    thresh = cv2.adaptiveThreshold(blur, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY,
                                   11, 2)
    thresh = cv2.bitwise_not(thresh)

    # Strengthen the grid lines a little so the contour is closed.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No contours found in image")

    # Pick the largest 4‑sided contour.
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        peri = cv2.arcLength(cnt, closed=True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, closed=True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype("float32")

    raise ValueError("Could not find a 4‑point Sudoku grid contour")


# --------------------------------------------------------------------------- #
# 3. perspective transform
# --------------------------------------------------------------------------- #

def warp_perspective_transform(image, grid_contour, target_size=450):
    """
    Apply a bird’s‑eye warp so that the Sudoku board becomes a perfect
    square of side length `target_size` pixels.
    """
    src = _order_points(grid_contour)

    dst = np.array([[0, 0],
                    [target_size - 1, 0],
                    [target_size - 1, target_size - 1],
                    [0, target_size - 1]],
                   dtype="float32")

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (target_size, target_size))
    return warped


# --------------------------------------------------------------------------- #
# 4. digit recognition
# --------------------------------------------------------------------------- #

def _train_knn_digit_classifier():
    """
    Train a simple KNN on OpenCV's 'digits.png' sample (5000 images of size
    20×20).  Returns the trained cv2.ml.KNearest object.
    """
    sample_path = cv2.samples.findFile("digits.png")
    digits_img  = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
    if digits_img is None:
        raise FileNotFoundError("Could not load OpenCV sample 'digits.png'")

    # Split into 20×20 cells: 50 rows × 100 cols = 5000 samples.
    rows = np.vsplit(digits_img, 50)
    cells = [np.hsplit(r, 100) for r in rows]
    cells = np.array(cells)  # shape = (50, 100, 20, 20)

    train = cells.reshape(-1, 400).astype(np.float32)   # 5000 × 400

    # Labels: 500 of each digit 0‑9.
    k = np.arange(10)
    labels = np.repeat(k, 500)[:, np.newaxis].astype(np.float32)

    knn = cv2.ml.KNearest_create()
    knn.train(train, cv2.ml.ROW_SAMPLE, labels)
    return knn


def _recognise_single_digit(cell, knn):
    """
    Try to recognise the digit present in the given cell (grayscale NumPy
    array).  Returns an int 0‑9; 0 means “no digit”.
    """
    h, w = cell.shape[:2]

    # Binary image: white digit on black background.
    thresh = cv2.threshold(cell, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Keep only the biggest blob inside the cell.
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < (h * w) * 0.02:     # too small → probably empty
        return 0

    x, y, w0, h0 = cv2.boundingRect(cnt)
    digit_roi = thresh[y:y + h0, x:x + w0]

    # Normalise to 20×20 like the training data.
    digit_roi = cv2.resize(digit_roi, (20, 20))
    sample = digit_roi.reshape(1, 400).astype(np.float32)

    _, result, _, _ = knn.findNearest(sample, k=5)
    return int(result[0][0])


def extract_and_recognize_digits(grid_image):
    """
    Split the warped grid into its 81 cells, recognise each digit,
    and return a 9×9 NumPy array containing the board (0 = blank).
    """
    gray_grid = cv2.cvtColor(grid_image, cv2.COLOR_BGR2GRAY)
    h, w = gray_grid.shape[:2]
    cell_h = h // 9
    cell_w = w // 9

    # Train the digit classifier once.
    knn = _train_knn_digit_classifier()

    board = np.zeros((9, 9), dtype=int)

    for row in range(9):
        for col in range(9):
            y0, y1 = row * cell_h, (row + 1) * cell_h
            x0, x1 = col * cell_w, (col + 1) * cell_w
            cell = gray_grid[y0:y1, x0:x1]
            digit = _recognise_single_digit(cell, knn)
            board[row, col] = digit

    return board


# --------------------------------------------------------------------------- #
# 5. main
# --------------------------------------------------------------------------- #

def main(path):
    # 1: load
    original = load_image(path)

    # 2: detect the outer contour
    contour = find_sudoku_grid_contour(original)

    # 3: perspective warp
    warped = warp_perspective_transform(original, contour, target_size=450)

    # 4: split and recognise
    board = extract_and_recognize_digits(warped)

    # 5: show & print
    print("Detected board (0 = empty):")
    print(board)

    # Optional: visual feedback
    cv2.imshow("Warped Sudoku", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python sudoku_reader.py  <sudoku_image>")
        sys.exit(1)
    main(sys.argv[1])