"""
Synthetic Sudoku image generator for on‑the‑fly training data.
Uses only computer fonts (no MNIST).
"""

import random
from pathlib import Path
from typing import Tuple, Optional, List

import cv2
import numpy as np

GRID_SIZE = 9
BASE_IMAGE_SIZE = 1000
CELL_SIZE = BASE_IMAGE_SIZE // GRID_SIZE


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order four points as TL, TR, BR, BL."""
    pts = pts.reshape(4, 2).astype("float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = pts[:, 1] - pts[:, 0]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


class SudokuRenderer:
    """
    Render a random (or specified) Sudoku grid to a synthetic image,
    using a variety of computer fonts.
    """

    def __init__(self) -> None:
        # A selection of built‑in OpenCV fonts
        self.fonts: List[int] = [
            cv2.FONT_HERSHEY_SIMPLEX,
            cv2.FONT_HERSHEY_COMPLEX,
            cv2.FONT_HERSHEY_DUPLEX,
            cv2.FONT_HERSHEY_TRIPLEX,
            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        ]

    def render_sudoku(
        self,
        grid_spec: Optional[List[List[Optional[int]]]] = None,
        *,
        allow_empty: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a synthetic Sudoku image.
        Returns (image, ground_truth_grid, warped_corners).
        """
        # Prepare ground truth grid
        if grid_spec is None:
            gt = np.zeros((GRID_SIZE, GRID_SIZE), int)
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    if allow_empty and random.random() < 0.4:
                        continue
                    gt[r, c] = random.randint(1, 9)
        else:
            gt = np.array([[d or 0 for d in row] for row in grid_spec], int)

        # Create background
        bg = tuple(random.randint(200, 240) for _ in range(3))
        img = np.full((BASE_IMAGE_SIZE, BASE_IMAGE_SIZE, 3), bg, np.uint8)

        # Draw grid lines
        for i in range(GRID_SIZE + 1):
            major = (i % 3 == 0)
            thickness = random.randint(3 if major else 1, 5 if major else 3)
            y = i * CELL_SIZE
            x = i * CELL_SIZE
            cv2.line(img, (0, y), (BASE_IMAGE_SIZE, y), (0, 0, 0), thickness)
            cv2.line(img, (x, 0), (x, BASE_IMAGE_SIZE), (0, 0, 0), thickness)

        # Draw digits using only fonts
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                d = gt[r, c]
                if d == 0:
                    continue

                # Compute target size for digit
                scale = random.uniform(0.5, 0.8)
                tgt = int(CELL_SIZE * scale)
                # Center of the cell with slight random offset
                center_x = c * CELL_SIZE + CELL_SIZE // 2
                center_y = r * CELL_SIZE + CELL_SIZE // 2
                dx = int(random.uniform(-0.1, 0.1) * CELL_SIZE)
                dy = int(random.uniform(-0.1, 0.1) * CELL_SIZE)
                cx, cy = center_x + dx, center_y + dy

                # Randomly pick font and thickness
                font = random.choice(self.fonts)
                thickness = random.randint(1, 3)
                # Add small variability to desired height
                desired_height = int(tgt * random.uniform(0.8, 1.2))
                font_scale = cv2.getFontScaleFromHeight(font, desired_height, thickness)

                text = str(d)
                (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
                x = cx - tw // 2
                y = cy + th // 2

                cv2.putText(
                    img,
                    text,
                    (x, y),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness,
                    cv2.LINE_AA,
                )

        # Add noise
        noise = np.random.normal(0, random.uniform(5, 20), img.shape).astype(np.float32)
        noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Random perspective warp
        h, w = noisy.shape[:2]
        orig = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
        shift = random.uniform(0.05, 0.2)
        max_dx, max_dy = w * shift, h * shift
        dst = np.array(
            [
                [random.uniform(0, max_dx), random.uniform(0, max_dy)],
                [w - 1 - random.uniform(0, max_dx), random.uniform(0, max_dy)],
                [
                    w - 1 - random.uniform(-max_dx * 0.2, max_dx),
                    h - 1 - random.uniform(0, max_dy),
                ],
                [random.uniform(-max_dx * 0.2, max_dx), h - 1 - random.uniform(0, max_dy)],
            ],
            dtype="float32",
        )
        M = cv2.getPerspectiveTransform(orig, dst)
        out_w = int(dst[:, 0].max()) + 1
        out_h = int(dst[:, 1].max()) + 1
        warped = cv2.warpPerspective(
            noisy, M, (out_w, out_h), borderMode=cv2.BORDER_REPLICATE
        )

        return warped, gt, dst


def generate_and_save_test_example(
    prefix: str = "epoch_test_sudoku",
    force: bool = False
) -> Tuple[str, np.ndarray]:
    """
    Generate or load a fixed Sudoku test example for epoch callbacks.
    Returns (image_path, ground_truth_grid).
    """
    import cv2 as _cv2, numpy as _np

    img_path = Path(f"{prefix}.png")
    gt_path = Path(f"{prefix}_gt.npy")

    if not force and img_path.exists() and gt_path.exists():
        return str(img_path), _np.load(gt_path)

    renderer = SudokuRenderer()
    grid_spec = [
        [None, None, 3, None, None, 6, None, 8, None],
        [8, None, 1, None, 3, None, 5, None, 4],
        [None, 4, None, 8, None, 7, None, 1, None],
        [1, None, None, 4, None, 5, None, None, 9],
        [None, 7, None, None, 2, None, None, 4, None],
        [5, None, None, 7, None, 1, None, None, 3],
        [None, 8, None, 5, None, 3, None, 9, None],
        [7, None, 4, None, 9, None, 1, None, 8],
        [None, 1, None, 6, None, None, 4, None, None],
    ]
    img, gt, _ = renderer.render_sudoku(grid_spec=grid_spec)
    _cv2.imwrite(str(img_path), img)
    _np.save(gt_path, gt)
    return str(img_path), gt
