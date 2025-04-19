"""
High‑level “one‑liner” API.

    python sudoku_recogniser.py  path/to/photo.jpg
prints the recognised board and highlights which cells were judged
low‑confidence.
"""
from __future__ import annotations
import sys
import cv2
import numpy as np
import digit_extractor as de
from digit_classifier import DigitClassifier


def recognise_sudoku(image_path: str,
                     min_conf: float = 0.70) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    outline = de.locate_grid_outline(img)
    warped = de.warp_grid(img, outline, side=450)
    cells = de.split_into_cells(warped)

    clf = DigitClassifier()       # load cached model or raise
    board = np.zeros((9, 9), np.int8)
    unsure = np.zeros((9, 9), np.bool_)

    for i, cell in enumerate(cells):
        d, conf = clf.predict(cell)
        r, c = divmod(i, 9)
        board[r, c] = d
        unsure[r, c] = conf < min_conf

    return board, unsure


# ----------------------------------------------------------- #
if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else "example.jpg"

    b, u = recognise_sudoku(img_path)
    print("\nRecognised board (0 = empty):")
    print(b)
    print("\nLow‑confidence mask (True = unsure):")
    print(u.astype(int))