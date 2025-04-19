"""
Script to generate example Sudoku images and demonstrate the extraction pipeline.

Generates a few synthetic Sudoku images, saves the original, the rectified grid,
and extracts and saves each individual cell image into separate subfolders.
"""

import shutil
from pathlib import Path

import cv2
import numpy as np

from sudoku_renderer import SudokuRenderer
from digit_extractor import GRID_SIZE, extract_cells_from_image

# --- Configuration ---
NUM_EXAMPLES = 5
OUTPUT_DIR = Path("example_pipeline")
RECTIFIED_SIZE = 450  # Match the size used in digit_extractor

# --- Main Execution ---
if __name__ == "__main__":
    # Clear previous results
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    renderer = SudokuRenderer()

    print(f"Generating {NUM_EXAMPLES} example Sudokus and extracting cells to {OUTPUT_DIR}...")

    for i in range(NUM_EXAMPLES):
        example_dir = OUTPUT_DIR / f"example_{i+1:02d}"
        example_dir.mkdir()

        print(f"\n--- Processing Example {i+1}/{NUM_EXAMPLES} ---")

        # 1. Generate a synthetic Sudoku image
        img, gt_grid, _ = renderer.render_sudoku(allow_empty=True)
        if img is None:
            print(f"Could not render example {i+1}. Skipping.")
            continue

        # Save the original rendered image
        img_path = example_dir / "original_sudoku.png"
        cv2.imwrite(str(img_path), img)
        print(f"Saved original image to {img_path}")

        # 2. Extract cells from the image
        cells, rectified, _ = extract_cells_from_image(img, size=RECTIFIED_SIZE, debug=False)

        if cells is None or rectified is None:
            print(f"Extraction failed for example {i+1}.")
            continue

        # Save the rectified grid image
        rectified_path = example_dir / "rectified_grid.png"
        cv2.imwrite(str(rectified_path), rectified)
        print(f"Saved rectified grid to {rectified_path}")

        # Save each cell image
        cells_dir = example_dir / "cells"
        cells_dir.mkdir()
        print(f"Saving individual cells to {cells_dir}")
        for cell_idx, cell_img in enumerate(cells):
            r, c = divmod(cell_idx, GRID_SIZE)
            cell_path = cells_dir / f"cell_{r}_{c}.png"
            cv2.imwrite(str(cell_path), cell_img)

        print(f"Successfully processed example {i+1}.")

    print("\nPipeline demonstration complete.")
