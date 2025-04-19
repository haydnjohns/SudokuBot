"""
Produce a synthetic hand‑held photo of a Sudoku grid.

The heavy‑lifting is done in the class `SudokuRenderer`.  The public
method that most callers want is
      renderer.render(board=None, *, perspective=True)
which returns

    • img_bgr  – np.uint8 (h, w, 3) picture (OpenCV ordering BGR)
    • board    – the 9×9 np.int8 array that was actually rendered
                 (0 = empty, 1‑9 = digit)

If you pass `board=None` a random board with ≈60 % empty cells is drawn.
"""
from __future__ import annotations
from pathlib import Path
import random
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


_FONT_CACHE: list[ImageFont.FreeTypeFont] | None = None


def _load_fonts(size: int = 48) -> list[ImageFont.FreeTypeFont]:
    """
    Load every *.ttf found in ./data/fonts/
    Fallback: if no fonts exist, Pillow's default font is used.
    """
    global _FONT_CACHE
    if _FONT_CACHE is not None:
        return _FONT_CACHE

    fonts_dir = Path(__file__).with_suffix("").parent / "data" / "fonts"
    fonts: list[ImageFont.FreeTypeFont] = []
    for p in fonts_dir.glob("*.ttf"):
        try:
            fonts.append(ImageFont.truetype(str(p), size))
        except Exception as e:
            print(f"[WARN] could not load font {p}: {e}")
    if not fonts:
        fonts.append(ImageFont.load_default())
    _FONT_CACHE = fonts
    return fonts


# --------------------------------------------------------------------- #
class SudokuRenderer:
    def __init__(
        self,
        img_size: int = 1000,            # base square canvas before warp
        cell_margin: float = 0.15,       # % of cell left as padding
        line_thickness: tuple[int, int] = (1, 5),  # (thin, thick) px
        bg_col_range: tuple[int, int] = (200, 240),
        rng: random.Random | None = None,
    ):
        self.img_size = img_size
        self.cell = img_size // 9
        self.cell_margin = cell_margin
        self.thin, self.thick = line_thickness
        self.bg_range = bg_col_range
        self.rng = rng or random.Random()

        # MNIST will be lazily loaded only if/when asked for digits
        self._mnist_digits: dict[int, list[np.ndarray]] | None = None

    # ---------------- public API ------------------------------------ #
    def render(
        self,
        board: np.ndarray | None = None,
        *,
        handwritten_ratio: float = 0.6,
        perspective: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Draw `board` (9×9 int array, 0=empty) on a paper‑like canvas.

        handwritten_ratio : 0→only font digits, 1→only MNIST digits
        perspective       : if True apply random 3‑D warp
        """
        if board is None:
            board = self._random_board()

        canvas = np.full(
            (self.img_size, self.img_size, 3),
            self.rng.randint(*self.bg_range),
            np.uint8,
        )

        # ------------------------------------------------------------- #
        # 1. grid lines
        # ------------------------------------------------------------- #
        for i in range(10):
            t = self.thick if i % 3 == 0 else self.thin
            # horizontal
            cv2.line(
                canvas,
                (0, i * self.cell),
                (self.img_size, i * self.cell),
                (0, 0, 0),
                t,
            )
            # vertical
            cv2.line(
                canvas,
                (i * self.cell, 0),
                (i * self.cell, self.img_size),
                (0, 0, 0),
                t,
            )

        # ------------------------------------------------------------- #
        # 2. fill digits
        # ------------------------------------------------------------- #
        for r in range(9):
            for c in range(9):
                d = int(board[r, c])
                if d == 0:
                    continue

                if self.rng.random() < handwritten_ratio:
                    glyph = self._mnist_digit(d)
                else:
                    glyph = self._font_digit(d)

                # random scale 75‑100 %
                scale = self.rng.uniform(0.75, 1.0)
                g_h, g_w = glyph.shape[:2]
                sz = int(scale * min(self.cell, self.cell) * (1 - self.cell_margin))
                glyph = cv2.resize(glyph, (sz, sz), interpolation=cv2.INTER_AREA)

                # paste centrally with a small random offset
                x0 = c * self.cell + (self.cell - glyph.shape[1]) // 2
                y0 = r * self.cell + (self.cell - glyph.shape[0]) // 2
                x0 += self.rng.randint(-self.cell // 15, self.cell // 15)
                y0 += self.rng.randint(-self.cell // 15, self.cell // 15)

                # white background, so simply dark‑ink the glyph
                mask = glyph < 200  # glyph drawn in black
                canvas[y0 : y0 + glyph.shape[0], x0 : x0 + glyph.shape[1]][mask] = (
                    0,
                    0,
                    0,
                )

        # ------------------------------------------------------------- #
        # 3. Add small Gaussian image noise
        # ------------------------------------------------------------- #
        noise = self.rng.normalvariate(0, 8)
        canvas = np.clip(
            canvas.astype(np.int16) + self.rng.normalvariate(0, 8), 0, 255
        ).astype(np.uint8)

        # ------------------------------------------------------------- #
        # 4. random 3‑D perspective tilt
        # ------------------------------------------------------------- #
        if perspective:
            canvas = self._perspective_warp(canvas)

        return canvas, board

    # ---------------- helpers --------------------------------------- #
    def _random_board(self, fill: float = 0.4) -> np.ndarray:
        """
        fill – probability each cell contains a digit
        """
        board = np.zeros((9, 9), np.int8)
        for r in range(9):
            for c in range(9):
                if self.rng.random() < fill:
                    board[r, c] = self.rng.randint(1, 9)
        return board

    # --------------- digit drawing helpers -------------------------- #
    def _mnist_digit(self, digit: int) -> np.ndarray:
        # lazy MNIST load
        if self._mnist_digits is None:
            from keras.datasets import mnist

            (x_tr, y_tr), _ = mnist.load_data()
            self._mnist_digits = {d: [] for d in range(10)}
            for im, lab in zip(x_tr, y_tr):
                self._mnist_digits[int(lab)].append(im)
        sample = self.rng.choice(self._mnist_digits[digit])
        # invert so black on white
        return 255 - sample

    def _font_digit(self, digit: int) -> np.ndarray:
        fonts = _load_fonts()
        font = self.rng.choice(fonts)
        img_pil = Image.new("L", (64, 64), 255)
        draw = ImageDraw.Draw(img_pil)
        w, h = draw.textbbox((0, 0), str(digit), font=font)[2:]
        draw.text(((64 - w) / 2, (64 - h) / 2), str(digit), font=font, fill=0)
        return np.array(img_pil)

    # --------------- perspective tilt ------------------------------- #
    # --------------- perspective tilt ------------------------------- #
    def _perspective_warp(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        # four source pts at the corners
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        # Define max displacement based on image size (original logic)
        # Using max_x for both x and y simplifies, adjust if separate needed
        max_offset = min(w, h) * 0.12 # Use a single max offset relative to min dim

        # Generate an array of random offsets with the same shape as src (4, 2)
        # Values will be between -max_offset and +max_offset
        random_offsets = np.random.uniform(
            low=-max_offset,
            high=max_offset,
            size=src.shape
        )
        # Add the random offsets to the original corner points
        # Ensure the result is float32, consistent with src and required by cv2.getPerspectiveTransform
        dst = (src + random_offsets).astype(np.float32)

        M = cv2.getPerspectiveTransform(src, dst)
        # Specify border mode and value explicitly for clarity and control
        # Using white border to simulate paper extending beyond the grid
        out = cv2.warpPerspective(
            img,
            M,
            (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255) # White border
        )
        return out