"""
Locate a Sudoku board inside a photo and split it into the 81  cells.
The most expensive part – finding the outer 4‑point contour – is robust
enough for the synthetic pictures produced by `sudoku_renderer.py`,
but usually also works on real smartphone photos.
"""
from __future__ import annotations
import cv2
import numpy as np


def _order_pts(pts: np.ndarray) -> np.ndarray:
    """clock‑wise starting TL, TR, BR, BL"""
    s = pts.sum(1)
    diff = np.diff(pts, 1)
    ordered = np.zeros((4, 2), np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


def locate_grid_outline(img_bgr: np.ndarray) -> np.ndarray:
    """return 4×2 float32 array of the outer Sudoku contour"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    th = cv2.adaptiveThreshold(blur, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)
    th = cv2.bitwise_not(th)
    th = cv2.dilate(th, np.ones((3, 3), np.uint8), iterations=2)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    raise ValueError("No 4‑point contour found – is there a Sudoku in the picture?")


def warp_grid(img_bgr: np.ndarray,
              contour: np.ndarray,
              side: int = 450) -> np.ndarray:
    """Bird’s‑eye view so every cell is a perfect square."""
    src = _order_pts(contour)
    dst = np.float32([[0, 0], [side-1, 0], [side-1, side-1], [0, side-1]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img_bgr, M, (side, side), flags=cv2.INTER_LINEAR)


def split_into_cells(warped: np.ndarray) -> list[np.ndarray]:
    """Return the 81 cell images in row‑major order (BGR)."""
    side = warped.shape[0]
    step = side // 9
    cells = []
    for r in range(9):
        for c in range(9):
            y0, y1 = r*step, (r+1)*step
            x0, x1 = c*step, (c+1)*step
            cells.append(warped[y0:y1, x0:x1].copy())
    return cells