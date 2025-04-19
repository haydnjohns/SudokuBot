#!/usr/bin/env python3
"""
Generate synthetic Sudoku examples and extract pipeline stages.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from sudoku_renderer import SudokuRenderer
from digit_extractor import GRID_SIZE, rectify_grid, split_into_cells

def main():
    parser = argparse.ArgumentParser(
        description="Generate data‐pipeline example images."
    )
    parser.add_argument(
        "--num", "-n",
        type=int, default=10,
        help="Number of examples to generate."
    )
    parser.add_argument(
        "--outdir", "-o",
        type=Path, default=Path("examples"),
        help="Root output directory."
    )
    args = parser.parse_args()

    renderer = SudokuRenderer()

    for i in range(1, args.num + 1):
        ex_dir = args.outdir / f"example_{i:02d}"
        cells_dir = ex_dir / "cells"
        ex_dir.mkdir(parents=True, exist_ok=True)
        cells_dir.mkdir(parents=True, exist_ok=True)

        # 1) Render a synthetic Sudoku (warped image + ground truth + corners)
        warped, gt_grid, corners = renderer.render_sudoku()

        # Save the original warped image
        orig_path = ex_dir / "original.png"
        cv2.imwrite(str(orig_path), warped)

        # Save the ground truth grid
        gt_path = ex_dir / "gt.npy"
        np.save(gt_path, gt_grid)

        # 2) Rectify the grid
        rectified = rectify_grid(warped, corners)
        if rectified is None:
            print(f"[Example {i:02d}] rectify failed, skipping.")
            continue
        rect_path = ex_dir / "rectified.png"
        cv2.imwrite(str(rect_path), rectified)

        # 3) Split into individual cells
        cells, _ = split_into_cells(rectified)
        expected = GRID_SIZE * GRID_SIZE
        if len(cells) != expected:
            print(f"[Example {i:02d}] expected {expected} cells, got {len(cells)}, skipping.")
            continue

        # Save each cell image
        for idx, cell in enumerate(cells):
            r, c = divmod(idx, GRID_SIZE)
            cell_path = cells_dir / f"cell_{r}_{c}.png"
            cv2.imwrite(str(cell_path), cell)

        print(f"[Example {i:02d}] generated at {ex_dir}")

if __name__ == "__main__":
    main()