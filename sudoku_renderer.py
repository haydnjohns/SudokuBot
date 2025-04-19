# sudoku_renderer.py
import cv2
import numpy as np
import random
from pathlib import Path
import urllib.request
import os

# --- Constants ---
_DIGITS_URL = ("https://raw.githubusercontent.com/opencv/opencv/"
               "master/samples/data/digits.png")
_DIGITS_PNG_PATH = Path(__file__).resolve().parent / "digits.png"
_DEFAULT_FONTS = [
    cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN,
    cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    cv2.FONT_HERSHEY_SIMPLEX, # Add more common ones for higher probability
    cv2.FONT_HERSHEY_SIMPLEX,
]

class SudokuRenderer:
    """
    Generates synthetic Sudoku puzzle images simulating photos of paper puzzles.

    Handles loading digit images (handwritten + fonts), composing the grid,
    applying perspective warp, noise, and other augmentations.
    """
    def __init__(self,
                 base_canvas_size=1000,
                 grid_line_thickness_range=(1, 5),
                 cell_padding_fraction=0.15, # Padding around digit within cell
                 digit_source_ratio=(0.5, 0.5), # (handwritten, font) probability
                 fonts=None,
                 noise_level_range=(5, 20),
                 blur_kernel_range=(0, 3), # 0 means no blur, otherwise odd kernel size (e.g., 1->3x3, 3->7x7)
                 warp_intensity=0.15, # Max corner shift as fraction of size
                 paper_color_range=((200, 200, 200), (245, 245, 245)), # BGR range
                 line_color=(0, 0, 0),
                 digit_color=(0, 0, 0)):
        """
        Initializes the SudokuRenderer.

        Args:
            base_canvas_size: Initial size of the grid image before warping.
            grid_line_thickness_range: Min/max thickness for grid lines.
            cell_padding_fraction: Min padding around digit relative to cell size.
            digit_source_ratio: Probability tuple for using (handwritten, font) digits.
            fonts: List of cv2 font constants to use. Defaults to _DEFAULT_FONTS.
            noise_level_range: Min/max std dev for Gaussian noise.
            blur_kernel_range: Min/max kernel size index for Gaussian blur (0=off).
            warp_intensity: Controls the maximum perspective distortion.
            paper_color_range: Min/max BGR values for the paper background.
            line_color: BGR color for grid lines.
            digit_color: BGR color for digits.
        """
        self.base_canvas_size = base_canvas_size
        self.grid_line_thickness_range = grid_line_thickness_range
        self.cell_padding_fraction = cell_padding_fraction
        self.digit_source_ratio = digit_source_ratio
        self.fonts = fonts if fonts else _DEFAULT_FONTS
        self.noise_level_range = noise_level_range
        self.blur_kernel_range = blur_kernel_range
        self.warp_intensity = warp_intensity
        self.paper_color_range = paper_color_range
        self.line_color = line_color
        self.digit_color = digit_color
        self.rng = np.random.default_rng()

        self.handwritten_digits = self._load_handwritten_digits()

    def _load_handwritten_digits(self):
        """Loads the 5000 handwritten digits from digits.png."""
        if not _DIGITS_PNG_PATH.exists():
            print(f"Downloading {_DIGITS_URL}...")
            try:
                urllib.request.urlretrieve(_DIGITS_URL, str(_DIGITS_PNG_PATH))
            except Exception as e:
                print(f"Error downloading digits.png: {e}")
                return {i: [] for i in range(10)} # Return empty dict on failure

        img = cv2.imread(str(_DIGITS_PNG_PATH), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error reading {_DIGITS_PNG_PATH}")
            return {i: [] for i in range(10)}

        cells = [np.hsplit(r, 100) for r in np.vsplit(img, 50)]
        digits_20x20 = np.array(cells, dtype=np.uint8).reshape(-1, 20, 20) # 5000x20x20
        labels = np.repeat(np.arange(10), 500)

        digit_map = {i: [] for i in range(10)}
        for img_cell, label in zip(digits_20x20, labels):
             # Invert (black digit on white bg) and convert to BGR for consistency
             inverted_cell = cv2.bitwise_not(img_cell)
             bgr_cell = cv2.cvtColor(inverted_cell, cv2.COLOR_GRAY2BGR)
             digit_map[label].append(bgr_cell)

        print(f"Loaded {sum(len(v) for v in digit_map.values())} handwritten digit samples.")
        return digit_map

    def _get_random_digit_image(self, digit):
        """Gets a random image for the specified digit (1-9)."""
        if not (1 <= digit <= 9):
            raise ValueError("Digit must be between 1 and 9.")

        use_handwritten = self.rng.random() < self.digit_source_ratio[0]

        if use_handwritten and self.handwritten_digits.get(digit):
            # Choose a random handwritten sample
            img = random.choice(self.handwritten_digits[digit]).copy()
            # Handwritten digits are already BGR (white bg, black digit)
            # We might need to adjust color later if digit_color is not black
            if self.digit_color != (0, 0, 0):
                 img[np.where((img == [0,0,0]).all(axis=2))] = self.digit_color
            return img
        else:
            # Generate using a font
            font_face = random.choice(self.fonts)
            # Estimate size needed - start large, then scale down
            font_scale_initial = 5.0
            thickness = self.rng.integers(1, 4)
            text = str(digit)

            (text_w, text_h), baseline = cv2.getTextSize(text, font_face, font_scale_initial, thickness)

            # Create image slightly larger than text
            margin = 10
            img_h = text_h + baseline + 2 * margin
            img_w = text_w + 2 * margin
            img = np.full((img_h, img_w, 3), (255, 255, 255), dtype=np.uint8) # White background

            # Calculate text origin (bottom-left)
            org_x = margin
            org_y = margin + text_h # Y is measured from top

            cv2.putText(img, text, (org_x, org_y), font_face, font_scale_initial,
                        self.digit_color, thickness, cv2.LINE_AA)

            # Optional: Add slight rotation/shear to font digits? (More complex)
            # For now, just return the clean font rendering
            return img


    def render_sudoku(self, grid_spec=None, difficulty=0.5):
        """
        Generates a synthetic Sudoku image.

        Args:
            grid_spec (Optional[np.ndarray]): A 9x9 numpy array with digits (1-9)
                or 0/None for empty cells. If None, a random grid is generated.
            difficulty (float): If grid_spec is None, controls the fraction of
                cells filled (0.0 to 1.0).

        Returns:
            tuple: (rendered_image, ground_truth_grid, corners_in_rendered)
                - rendered_image: The final BGR image.
                - ground_truth_grid: The 9x9 numpy array used (0 for empty).
                - corners_in_rendered: 4x2 numpy array of the original grid corners
                                       in the coordinate system of the rendered image.
        """
        # 1. Determine Ground Truth Grid
        if grid_spec is None:
            ground_truth_grid = np.zeros((9, 9), dtype=int)
            num_filled = int(81 * difficulty)
            indices = self.rng.choice(81, num_filled, replace=False)
            rows, cols = np.unravel_index(indices, (9, 9))
            for r, c in zip(rows, cols):
                ground_truth_grid[r, c] = self.rng.integers(1, 10) # Digits 1-9
        else:
            if not isinstance(grid_spec, np.ndarray) or grid_spec.shape != (9, 9):
                raise ValueError("grid_spec must be a 9x9 numpy array.")
            # Replace None with 0 for consistency
            ground_truth_grid = np.nan_to_num(grid_spec.astype(float)).astype(int)

        # 2. Create Base Canvas
        bg_color = self.rng.integers(self.paper_color_range[0], self.paper_color_range[1], size=3, endpoint=True).tolist()
        image = np.full((self.base_canvas_size, self.base_canvas_size, 3), bg_color, dtype=np.uint8)
        cell_draw_size = self.base_canvas_size // 9

        # 3. Draw Grid Lines
        min_thick, max_thick = self.grid_line_thickness_range
        for i in range(10):
            thickness = self.rng.integers(min_thick, max_thick + 1)
            if i % 3 == 0: # Thicker lines for 3x3 blocks
                thickness = self.rng.integers(max(min_thick, 2), max_thick + 1) # Ensure thicker is >= min
            pos = i * cell_draw_size
            # Clamp position to avoid drawing outside bounds if thickness is large
            pos = max(0, min(self.base_canvas_size - 1, pos))
            # Horizontal
            cv2.line(image, (0, pos), (self.base_canvas_size, pos), self.line_color, thickness)
            # Vertical
            cv2.line(image, (pos, 0), (pos, self.base_canvas_size), self.line_color, thickness)

        # 4. Place Digits
        for r in range(9):
            for c in range(9):
                digit = ground_truth_grid[r, c]
                if digit == 0: # Skip empty cells
                    continue

                digit_img = self._get_random_digit_image(digit)
                if digit_img is None: continue # Skip if image loading failed

                # Calculate target size for the digit within the cell
                padding = int(cell_draw_size * self.cell_padding_fraction)
                max_digit_h = cell_draw_size - 2 * padding
                max_digit_w = cell_draw_size - 2 * padding

                if max_digit_h <= 0 or max_digit_w <= 0: continue # Cell too small

                # Resize digit image preserving aspect ratio
                h_orig, w_orig = digit_img.shape[:2]
                scale = min(max_digit_w / w_orig, max_digit_h / h_orig)
                new_w, new_h = int(w_orig * scale), int(h_orig * scale)

                if new_w <= 0 or new_h <= 0: continue # Resized too small

                resized_digit = cv2.resize(digit_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # Calculate placement position (top-left corner) within the cell
                cell_top = r * cell_draw_size
                cell_left = c * cell_draw_size

                # Center the digit within the available space (cell - padding)
                offset_x = (cell_draw_size - new_w) // 2
                offset_y = (cell_draw_size - new_h) // 2

                # Add small random offset
                max_jitter = padding // 2
                jitter_x = self.rng.integers(-max_jitter, max_jitter + 1)
                jitter_y = self.rng.integers(-max_jitter, max_jitter + 1)

                paste_x = cell_left + offset_x + jitter_x
                paste_y = cell_top + offset_y + jitter_y

                # Ensure placement is within bounds
                paste_x = max(0, paste_x)
                paste_y = max(0, paste_y)

                # Calculate end coordinates, careful not to exceed image bounds
                end_row = min(paste_y + new_h, self.base_canvas_size)
                end_col = min(paste_x + new_w, self.base_canvas_size)
                # Adjust height/width of digit slice if clipping occurred
                slice_h = end_row - paste_y
                slice_w = end_col - paste_x

                if slice_h <= 0 or slice_w <= 0: continue # Nothing to paste

                # Create mask for transparency (assuming white background in digit img)
                # Convert resized digit to grayscale for masking
                gray_digit = cv2.cvtColor(resized_digit[:slice_h, :slice_w], cv2.COLOR_BGR2GRAY)
                # Mask is where the digit is NOT white (i.e., the digit itself)
                mask = cv2.threshold(gray_digit, 240, 255, cv2.THRESH_BINARY_INV)[1]

                # Paste using mask
                roi = image[paste_y:end_row, paste_x:end_col]
                # Ensure mask has 3 channels if needed, or apply per channel
                mask_inv = cv2.bitwise_not(mask)
                img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                img_fg = cv2.bitwise_and(resized_digit[:slice_h, :slice_w], resized_digit[:slice_h, :slice_w], mask=mask)

                image[paste_y:end_row, paste_x:end_col] = cv2.add(img_bg, img_fg)


        # 5. Apply Noise
        std_dev = self.rng.uniform(self.noise_level_range[0], self.noise_level_range[1])
        noise = self.rng.normal(0, std_dev, image.shape).astype(np.float32)
        noisy_image = image.astype(np.float32) + noise
        image = np.clip(noisy_image, 0, 255).astype(np.uint8)

        # 6. Apply Blur
        blur_k_idx = self.rng.integers(self.blur_kernel_range[0], self.blur_kernel_range[1] + 1)
        if blur_k_idx > 0:
            ksize = blur_k_idx * 2 + 1 # Ensure odd kernel size (3, 5, 7...)
            image = cv2.GaussianBlur(image, (ksize, ksize), 0)

        # 7. Apply Perspective Warp
        h, w = image.shape[:2]
        original_corners = np.array([
            [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]
        ], dtype="float32")

        max_shift_x = w * self.warp_intensity
        max_shift_y = h * self.warp_intensity

        # Encourage lower-angle views: shift top corners more inward, bottom corners less so
        # or shift top corners down, bottom corners up slightly
        top_shift_x = max_shift_x * self.rng.uniform(0.5, 1.0)
        top_shift_y = max_shift_y * self.rng.uniform(0.5, 1.0) # Shift top corners down
        bottom_shift_x = max_shift_x * self.rng.uniform(0.1, 0.5)
        bottom_shift_y = max_shift_y * self.rng.uniform(0.0, 0.3) # Shift bottom corners up slightly

        shifted_corners = np.array([
            [self.rng.uniform(0, top_shift_x), self.rng.uniform(0, top_shift_y)], # Top-left
            [w - 1 - self.rng.uniform(0, top_shift_x), self.rng.uniform(0, top_shift_y)], # Top-right
            [w - 1 - self.rng.uniform(-bottom_shift_x/2, bottom_shift_x), h - 1 - self.rng.uniform(0, bottom_shift_y)], # Bottom-right
            [self.rng.uniform(-bottom_shift_x/2, bottom_shift_x), h - 1 - self.rng.uniform(0, bottom_shift_y)]  # Bottom-left
        ], dtype="float32")

        # Ensure corners form a valid quadrilateral (basic check: prevent self-intersection)
        # A more robust check might be needed, but this prevents extreme cases.
        shifted_corners = np.clip(shifted_corners, -w*0.1, w*1.1) # Allow slight overshoot

        matrix = cv2.getPerspectiveTransform(original_corners, shifted_corners)
        # Determine output size - let's keep it the same as the base canvas for simplicity
        # Or calculate bounds of shifted corners? Keep it simple for now.
        warped_image = cv2.warpPerspective(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # The 'corners_in_rendered' are simply the 'shifted_corners'
        corners_in_rendered = shifted_corners

        return warped_image, ground_truth_grid, corners_in_rendered

# --- Example Usage ---
if __name__ == "__main__":
    print("[INFO] Initializing Sudoku Renderer...")
    renderer = SudokuRenderer(warp_intensity=0.2, digit_source_ratio=(0.7, 0.3)) # More handwritten

    print("[INFO] Generating random Sudoku image...")
    # Example: Generate a random grid with ~40% filled cells
    rendered_img, truth_grid, corners = renderer.render_sudoku(difficulty=0.4)

    print("[INFO] Ground Truth Grid (0=empty):")
    print(truth_grid)
    # print("\n[INFO] Corners in Rendered Image:")
    # print(corners)

    # Draw detected corners on the image for visualization
    vis_img = rendered_img.copy()
    for i in range(4):
        pt1 = tuple(corners[i].astype(int))
        pt2 = tuple(corners[(i + 1) % 4].astype(int))
        cv2.line(vis_img, pt1, pt2, (0, 255, 0), 2)
        cv2.circle(vis_img, pt1, 5, (0, 0, 255), -1)

    print("[INFO] Displaying generated Sudoku image with corners...")
    cv2.imshow("Generated Sudoku", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Example: Generate a specific grid
    print("\n[INFO] Generating specific Sudoku image...")
    specific_grid = np.array([
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ])
    rendered_img_spec, _, _ = renderer.render_sudoku(grid_spec=specific_grid)
    cv2.imshow("Specific Sudoku", rendered_img_spec)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("[INFO] Renderer testing complete.")