#!/usr/bin/env python3
"""
End‑to‑end Sudoku grid detection and digit recognition.

usage:  python sudoku_reader.py  sudoku_image.png
"""
import sys
import urllib
import urllib.request
from pathlib import Path

import cv2
import numpy as np

_DIGITS_URL = ("https://raw.githubusercontent.com/opencv/opencv/"
               "master/samples/data/digits.png")


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
# 4. MUCH better digit recognition
# --------------------------------------------------------------------------- #
#
# (1)  deskew()       – mutates every sample so the mass centre sits on
#                       the vertical symmetry axis.
# (2)  hog()          – 16‑bin Histogram‑of‑Oriented‑Gradients, identical
#                       to the OpenCV sample implementation.
# (3)  _train_svm()   – trains once, then stores a ‘digits_svm.yml’ next to
#                       the script so the next run just loads it.
# (4)  _recognise_single_digit() – new, much more robust pipeline.
#

def _deskew(img):
    m = cv2.moments(img)
    if abs(m["mu02"]) < 1e-2:          # nearly no y‑variation → skip
        return img.copy()
    skew = m["mu11"] / m["mu02"]
    M = np.float32([[1, skew, -0.5 * 20 * skew],
                    [0, 1,             0      ]])
    img = cv2.warpAffine(img, M, (20, 20),
                         flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


def _hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    bins = np.int32(16 * ang / 360)           # 16 bins
    bin_cells = []
    mag_cells = []
    # 4 blocks of 10×10 pixels each
    for i in range(2):
        for j in range(2):
            bin_cells.append(bins[i*10:(i+1)*10, j*10:(j+1)*10])
            mag_cells.append(mag[i*10:(i+1)*10, j*10:(j+1)*10])

    hist = [np.bincount(b.ravel(),
                        m.ravel(),
                        16) for b, m in zip(bin_cells, mag_cells)]
    return np.hstack(hist).astype(np.float32)


def _train_svm(model_path="digits_svm.yml"):
    model_file = Path(__file__).with_name(model_path)
    if model_file.exists():
        svm = cv2.ml.SVM_load(str(model_file))
        return svm

    # ------------------------------------------------------------------ #
    # 1.  read training image (download if necessary – identical code)
    here = Path(__file__).resolve().parent
    local_copy = here / "digits.png"
    if local_copy.exists():
        sample_path = str(local_copy)
    else:
        sample_path = cv2.samples.findFile("digits.png", required=False)
        if not sample_path or not Path(sample_path).exists():
            print("digits.png not found – downloading it ...")
            urllib.request.urlretrieve(_DIGITS_URL, local_copy)
            sample_path = str(local_copy)

    digits_img = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
    if digits_img is None:
        raise FileNotFoundError("Could not load digits.png")

    # ------------------------------------------------------------------ #
    # 2.  split -> deskew -> HOG
    rows  = np.vsplit(digits_img, 50)            # 50 rows
    cells = [np.hsplit(r, 100) for r in rows]    # 100 columns
    cells = np.array(cells, dtype=np.uint8)      # (50,100,20,20)

    train_data = []
    for img in cells.reshape(-1, 20, 20):
        img = _deskew(img)
        train_data.append(_hog(img))
    train_data = np.vstack(train_data)

    labels = np.repeat(np.arange(10), 500)[:, None]

    # ------------------------------------------------------------------ #
    # 3.  train SVM
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setC(2.5)
    svm.setGamma(0.05)
    svm.train(train_data, cv2.ml.ROW_SAMPLE, labels)
    svm.save(str(model_file))
    print(f"SVM trained and cached at {model_file}")
    return svm


_SVM = None          # will be initialised on first call of recognise()


def _recognise_single_digit(cell):
    """
    Try to recognise the digit present in `cell` (BGR or grayscale image).
    Returns an int 0‑9; 0 means “no digit detected”.
    """
    global _SVM
    if _SVM is None:
        _SVM = _train_svm()

    if len(cell.shape) == 3:
        cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

    h, w = cell.shape[:2]
    margin = int(0.12 * min(h, w))        # drop ~12 % border → bye bye grid
    cell = cell[margin:h - margin, margin:w - margin]

    # binarise & strip really thin stuff (grid leftovers)
    thresh = cv2.adaptiveThreshold(cell, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV,
                                   11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 0.02 * h * w:   # still almost empty
        return 0

    x, y, w0, h0 = cv2.boundingRect(cnt)
    digit_roi = thresh[y:y + h0, x:x + w0]

    # put the digit on a 20×20 canvas, keeping aspect ratio
    canvas = np.zeros((20, 20), dtype=np.uint8)
    roi_h, roi_w = digit_roi.shape
    scale = 18.0 / max(roi_h, roi_w)
    roi = cv2.resize(digit_roi, (int(roi_w * scale), int(roi_h * scale)))
    dy = (20 - roi.shape[0]) // 2
    dx = (20 - roi.shape[1]) // 2
    canvas[dy:dy + roi.shape[0], dx:dx + roi.shape[1]] = roi

    canvas = _deskew(canvas)
    sample = _hog(canvas).reshape(1, -1)
    _, result = _SVM.predict(sample)
    return int(result[0, 0])


def extract_and_recognize_digits(grid_image):
    """
    Split the warped grid into 81 cells, recognise each digit,
    and return a 9×9 NumPy array (0 = blank).
    """
    if grid_image.ndim == 2:               # already gray?
        gray_grid = grid_image
    else:
        gray_grid = cv2.cvtColor(grid_image, cv2.COLOR_BGR2GRAY)

    h, w = gray_grid.shape
    cell_h, cell_w = h // 9, w // 9

    board = np.zeros((9, 9), dtype=int)
    for r in range(9):
        for c in range(9):
            y0, y1 = r * cell_h, (r + 1) * cell_h
            x0, x1 = c * cell_w, (c + 1) * cell_w
            cell = gray_grid[y0:y1, x0:x1]
            board[r, c] = _recognise_single_digit(cell)
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
    path = "sodoku_image.png" if len(sys.argv) == 1 else sys.argv[1]
    main(path)