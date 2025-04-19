# sudoku_recogniser.py
"""
Command‑line utility: detect a Sudoku grid in an image and recognise its digits.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from digit_classifier import (
    EMPTY_LABEL,
    MODEL_FILENAME,
    DigitClassifier,
)
from digit_extractor import GRID_SIZE, extract_cells_from_image

FINAL_CONFIDENCE_THRESHOLD = 0.80


# --------------------------------------------------------------------------- #
#  Pretty printing                                                            #
# --------------------------------------------------------------------------- #
def print_sudoku_grid(
    grid: np.ndarray,
    conf: np.ndarray | None = None,
    threshold: float = FINAL_CONFIDENCE_THRESHOLD,
) -> None:
    """Nicely print a 9×9 Sudoku grid."""
    grid = np.asarray(grid)
    if grid.shape != (GRID_SIZE, GRID_SIZE):
        print("[print_sudoku_grid] invalid shape")
        return

    for r in range(GRID_SIZE):
        if r and r % 3 == 0:
            print("|-------+-------+-------|")

        line = []
        for c in range(GRID_SIZE):
            d = grid[r, c]
            if d == 0:
                token = "."
            else:
                token = str(d)
                if conf is not None and conf[r, c] < threshold:
                    token = "?"
            line.append(token)

            if (c + 1) % 3 == 0 and c != GRID_SIZE - 1:
                line.append("|")
        print(" ".join(line))
    print()


# --------------------------------------------------------------------------- #
#  Overlay helper                                                             #
# --------------------------------------------------------------------------- #
def display_results_on_image(
    rectified: np.ndarray, grid: np.ndarray
) -> np.ndarray | None:
    """Draw recognised digits onto the rectified grid image."""
    if rectified is None:
        return None

    img = (
        cv2.cvtColor(rectified, cv2.COLOR_GRAY2BGR)
        if rectified.ndim == 2
        else rectified.copy()
    )
    h, w = img.shape[:2]
    cell_h, cell_w = h // GRID_SIZE, w // GRID_SIZE

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            d = grid[r, c]
            if d == 0:
                continue
            text = str(d)
            (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            x = c * cell_w + (cell_w - tw) // 2
            y = r * cell_h + (cell_h + th) // 2
            cv2.putText(
                img,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
    return img


# --------------------------------------------------------------------------- #
#  Recognition pipeline                                                       #
# --------------------------------------------------------------------------- #
def recognise_sudoku(
    img_path: Path,
    classifier: DigitClassifier,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Extract cells and run the classifier."""
    print(f"Processing {img_path} ...")
    start = time.time()
    cells, rectified, _ = extract_cells_from_image(img_path)
    if cells is None:
        raise RuntimeError("Extraction failed.")

    pred = np.zeros((GRID_SIZE, GRID_SIZE), int)
    conf = np.zeros_like(pred, float)

    for i, cell in enumerate(cells):
        r, c = divmod(i, GRID_SIZE)
        d, cf = classifier.recognise(cell, confidence_threshold=0.1)
        conf[r, c] = cf
        if d and cf >= FINAL_CONFIDENCE_THRESHOLD:
            pred[r, c] = d

    print(f"Done in {time.time() - start:.2f}s")
    return pred, conf, rectified


# --------------------------------------------------------------------------- #
#  Entry‑point                                                                #
# --------------------------------------------------------------------------- #
def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: sudoku_recogniser.py <image>")
        sys.exit(0)

    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print("Image not found.")
        sys.exit(1)

    clf = DigitClassifier(model_path=MODEL_FILENAME)
    if clf.model is None:
        print("Model missing – training required.")
        clf.train()

    grid, conf, rectified = recognise_sudoku(img_path, clf)
    print_sudoku_grid(grid, conf)

    res = display_results_on_image(rectified, grid)
    if res is not None:
        cv2.imshow("Result", res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
