"""
Command-line utility to detect a Sudoku grid in an image and recognise its digits.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np

import digit_classifier
from digit_extractor import GRID_SIZE, extract_cells_from_image

FINAL_CONFIDENCE_THRESHOLD = 0.80


def print_sudoku_grid(
    grid: np.ndarray,
    conf: Optional[np.ndarray] = None,
    threshold: float = FINAL_CONFIDENCE_THRESHOLD
) -> None:
    """Nicely print a 9×9 Sudoku grid, marking low‑confidence digits as '?'. """
    grid = np.asarray(grid)
    if grid.shape != (GRID_SIZE, GRID_SIZE):
        print("[print_sudoku_grid] invalid shape")
        return

    for r in range(GRID_SIZE):
        if r > 0 and r % 3 == 0:
            print("|-----+-------+-----|")

        tokens = []
        for c in range(GRID_SIZE):
            d = grid[r, c]
            if d == 0:
                tok = "."
            else:
                tok = str(d)
                if conf is not None and conf[r, c] < threshold:
                    tok = "?"
            tokens.append(tok)
            if (c + 1) % 3 == 0 and c < GRID_SIZE - 1:
                tokens.append("|")
        print(" ".join(tokens))
    print()


def display_results_on_image(
    rectified: Optional[np.ndarray],
    grid: np.ndarray
) -> Optional[np.ndarray]:
    """
    Draw recognised digits onto the rectified grid image and return it.
    """
    if rectified is None:
        return None

    img = (
        cv2.cvtColor(rectified, cv2.COLOR_GRAY2BGR)
        if rectified.ndim == 2 else rectified.copy()
    )
    h, w = img.shape[:2]
    cell_h, cell_w = h // GRID_SIZE, w // GRID_SIZE

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            d = grid[r, c]
            if d == 0:
                continue
            text = str(d)
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2
            )
            x = c * cell_w + (cell_w - tw) // 2
            y = r * cell_h + (cell_h + th) // 2
            cv2.putText(
                img, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA
            )
    return img


def recognise_sudoku(
    img_path: Path,
    classifier: digit_classifier.DigitClassifier
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Extract cells and run the classifier on each one.
    Returns (predicted_grid, confidence_grid, rectified_image).
    """
    print(f"Processing {img_path} ...")
    start = time.time()
    cells, rectified, _ = extract_cells_from_image(img_path)
    if cells is None or rectified is None:
        raise RuntimeError("Extraction failed.")

    pred = np.zeros((GRID_SIZE, GRID_SIZE), int)
    conf = np.zeros_like(pred, float)

    for i, cell in enumerate(cells):
        r, c = divmod(i, GRID_SIZE)
        d, cf = classifier.recognise(cell, confidence_threshold=0.1)
        conf[r, c] = cf
        if d != 0 and cf >= FINAL_CONFIDENCE_THRESHOLD:
            pred[r, c] = d

    elapsed = time.time() - start
    print(f"Done in {elapsed:.2f}s")
    return pred, conf, rectified


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: sudoku_recogniser.py <image_path>")
        sys.exit(0)

    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print(f"Image not found: {img_path}")
        sys.exit(1)

    classifier = digit_classifier.DigitClassifier(model_path=digit_classifier.MODEL_FILENAME)
    if classifier.model is None:
        print("Model missing—training required.")
        classifier.train()

    grid, conf, rectified = recognise_sudoku(img_path, classifier)
    print_sudoku_grid(grid, conf)

    result_img = display_results_on_image(rectified, grid)
    if result_img is not None:
        cv2.imshow("Sudoku Recognition", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
