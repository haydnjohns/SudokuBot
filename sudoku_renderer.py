"""
Synthetic Sudoku image generator for on‑the‑fly training data.
"""

import random
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
import keras

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


def _load_mnist_digits() -> dict[int, list[np.ndarray]]:
    """
    Download MNIST once and bucket images by label for rendering.
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    images = np.concatenate([x_train, x_test])
    labels = np.concatenate([y_train, y_test])

    buckets: dict[int, list[np.ndarray]] = {i: [] for i in range(10)}
    for img, lbl in zip(images, labels):
        inv = cv2.bitwise_not(img)
        padded = cv2.copyMakeBorder(
            inv, 4, 4, 4, 4,
            cv2.BORDER_CONSTANT, value=255
        )
        buckets[int(lbl)].append(padded)
    return buckets


class SudokuRenderer:
    """
    Render a random (or specified) Sudoku grid to a synthetic image.
    """

    def __init__(self) -> None:
        self.mnist_buckets: Optional[dict[int, list[np.ndarray]]] = None

    def _digit_source(self, digit: int) -> Tuple[Optional[np.ndarray], str]:
        if self.mnist_buckets is None:
            self.mnist_buckets = _load_mnist_digits()

        sources = []
        if self.mnist_buckets[digit]:
            sources.append("mnist")
        sources.append("font")
        choice = random.choice(sources)

        if choice == "mnist":
            img = random.choice(self.mnist_buckets[digit])
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img, "mnist"
        return None, "font"

    def render_sudoku(
        self,
        grid_spec: Optional[list[list[int | None]]] = None,
        *,
        allow_empty: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a synthetic Sudoku image.
        Returns (image, ground_truth_grid, warped_corners).
        """
        if grid_spec is None:
            gt = np.zeros((GRID_SIZE, GRID_SIZE), int)
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    if allow_empty and random.random() < 0.4:
                        continue
                    gt[r, c] = random.randint(1, 9)
        else:
            gt = np.array([[d or 0 for d in row] for row in grid_spec], int)

        bg = tuple(random.randint(200, 240) for _ in range(3))
        img = np.full((BASE_IMAGE_SIZE, BASE_IMAGE_SIZE, 3), bg, np.uint8)

        # draw grid lines
        for i in range(GRID_SIZE + 1):
            major = (i % 3 == 0)
            thickness = random.randint(3 if major else 1, 5 if major else 3)
            cv2.line(img, (0, i*CELL_SIZE), (BASE_IMAGE_SIZE, i*CELL_SIZE), (0, 0, 0), thickness)
            cv2.line(img, (i*CELL_SIZE, 0), (i*CELL_SIZE, BASE_IMAGE_SIZE), (0, 0, 0), thickness)

        # draw digits
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                d = gt[r, c]
                if d == 0:
                    continue

                src_img, src_type = self._digit_source(d)
                scale = random.uniform(0.5, 0.8)
                tgt = int(CELL_SIZE * scale)
                center_x = c*CELL_SIZE + CELL_SIZE//2
                center_y = r*CELL_SIZE + CELL_SIZE//2
                dx = int(random.uniform(-0.1, 0.1)*CELL_SIZE)
                dy = int(random.uniform(-0.1, 0.1)*CELL_SIZE)
                cx, cy = center_x + dx, center_y + dy

                if src_type == "mnist":
                    digit = cv2.resize(src_img, (tgt, tgt))
                    angle = random.uniform(-10, 10)
                    M = cv2.getRotationMatrix2D((tgt/2, tgt/2), angle, 1)
                    digit = cv2.warpAffine(
                        digit, M, (tgt, tgt),
                        borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255)
                    )
                    mask = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY_INV)

                    x0 = max(0, cx - tgt//2)
                    y0 = max(0, cy - tgt//2)
                    roi = img[y0:y0+tgt, x0:x0+tgt]
                    m_inv = cv2.bitwise_not(mask[:roi.shape[0], :roi.shape[1]])
                    bg_region = cv2.bitwise_and(roi, roi, mask=m_inv)
                    fg_region = cv2.bitwise_and(digit, digit, mask=mask[:roi.shape[0], :roi.shape[1]])
                    img[y0:y0+roi.shape[0], x0:x0+roi.shape[1]] = cv2.add(bg_region, fg_region)
                else:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    thickness = random.randint(1, 3)
                    font_scale = cv2.getFontScaleFromHeight(font, tgt, thickness) * 0.8
                    text = str(d)
                    tw, th = cv2.getTextSize(text, font, font_scale, thickness)[0]
                    x = cx - tw//2
                    y = cy + th//2
                    cv2.putText(img, text, (x, y),
                                font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        # add noise
        noise = np.random.normal(0, random.uniform(5,20), img.shape).astype(np.float32)
        noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # random perspective warp
        h, w = noisy.shape[:2]
        orig = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype="float32")
        shift = random.uniform(0.05, 0.2)
        max_dx, max_dy = w*shift, h*shift
        dst = np.array([
            [random.uniform(0, max_dx), random.uniform(0, max_dy)],
            [w-1-random.uniform(0, max_dx), random.uniform(0, max_dy)],
            [w-1-random.uniform(-max_dx*0.2, max_dx), h-1-random.uniform(0, max_dy)],
            [random.uniform(-max_dx*0.2, max_dx), h-1-random.uniform(0, max_dy)],
        ], dtype="float32")
        M = cv2.getPerspectiveTransform(orig, dst)
        out_w = int(dst[:,0].max()) + 1
        out_h = int(dst[:,1].max()) + 1
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
    img_path = Path(f"{prefix}.png")
    gt_path = Path(f"{prefix}_gt.npy")

    if not force and img_path.exists() and gt_path.exists():
        return str(img_path), np.load(gt_path)

    renderer = SudokuRenderer()
    grid_spec = [
        [None, None, 3,    None, None, 6,    None, 8,    None],
        [8,    None, 1,    None, 3,    None, 5,    None, 4   ],
        [None, 4,    None, 8,    None, 7,    None, 1,    None],
        [1,    None, None, 4,    None, 5,    None, None, 9   ],
        [None, 7,    None, None, 2,    None, None, 4,    None],
        [5,    None, None, 7,    None, 1,    None, None, 3   ],
        [None, 8,    None, 5,    None, 3,    None, 9,    None],
        [7,    None, 4,    None, 9,    None, 1,    None, 8   ],
        [None, 1,    None, 6,    None, None, 4,    None, None],
    ]
    img, gt, _ = renderer.render_sudoku(grid_spec=grid_spec)
    cv2.imwrite(str(img_path), img)
    np.save(gt_path, gt)
    return str(img_path), gt
