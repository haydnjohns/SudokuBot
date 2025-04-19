# digit_extractor.py
"""
Find a Sudoku grid in an image, rectify it and split it into 81 cell images.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

GRID_SIZE = 9
DEFAULT_RECTIFIED_SIZE = 450


# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #
def _order_points(pts: np.ndarray) -> np.ndarray:
    """Return points ordered TL, TR, BR, BL."""
    pts = pts.reshape(4, 2).astype("float32")
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = pts[:, 1] - pts[:, 0]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# --------------------------------------------------------------------------- #
#  Core functionality                                                         #
# --------------------------------------------------------------------------- #
def find_sudoku_grid_contour(
    img: np.ndarray, *, debug_dir: Path | None = None
) -> np.ndarray:
    """Return a four‑point contour of the outer Sudoku boundary."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        4,
    )
    closed = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=2,
    )

    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    min_area = gray.size * 0.1
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            break
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)
            if 0.8 < w / h < 1.2:
                return approx.reshape(4, 2).astype("float32")

    raise ValueError("Sudoku grid not found.")


def rectify_grid(
    img: np.ndarray, contour: np.ndarray, *, size: int = DEFAULT_RECTIFIED_SIZE
) -> np.ndarray | None:
    """Perspective‑correct the Sudoku grid."""
    try:
        src = _order_points(contour)
        dst = np.array(
            [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
            dtype="float32",
        )
        m = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img, m, (size, size))
    except cv2.error:
        return None


def split_into_cells(
    rectified: np.ndarray,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Return list with 81 cells (cropped)."""
    h, w = rectified.shape[:2]
    if h != w:
        size = max(h, w)
        rectified = cv2.resize(rectified, (size, size))
        h = w = size

    cell_size = h // GRID_SIZE
    margin = int(cell_size * 0.04)
    cells: list[np.ndarray] = []

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            y0 = r * cell_size + margin
            x0 = c * cell_size + margin
            y1 = (r + 1) * cell_size - margin
            x1 = (c + 1) * cell_size - margin
            if y0 >= y1 or x0 >= x1:
                cell = np.zeros((cell_size, cell_size), rectified.dtype)
            else:
                cell = rectified[y0:y1, x0:x1]
            cells.append(cell)

    return cells, rectified


def extract_cells_from_image(
    img_or_path, *, size: int = DEFAULT_RECTIFIED_SIZE, debug: bool = False
):
    """High‑level convenience function."""
    if isinstance(img_or_path, (str, Path)):
        img = cv2.imread(str(img_or_path))
        if img is None:
            return None, None, None
    elif isinstance(img_or_path, np.ndarray):
        img = img_or_path.copy()
    else:
        return None, None, None

    try:
        contour = find_sudoku_grid_contour(img, debug_dir=Path(f"debug_{os.getpid()}") if debug else None)
        rectified = rectify_grid(img, contour, size=size)
        if rectified is None:
            return None, None, None
        cells, rectified = split_into_cells(rectified)
        return cells, rectified, contour
    except Exception as exc:
        print(f"[Extractor] {exc}")
        return None, None, None
