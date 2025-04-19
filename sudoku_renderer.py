# sudoku_renderer.py
import cv2
import numpy as np
import random
import os
from pathlib import Path
import keras # For MNIST dataset loading

# --- Constants ---
GRID_SIZE = 9
DEFAULT_BASE_IMAGE_SIZE = 1000 # Initial size before warp
DEFAULT_CELL_DRAW_SIZE = DEFAULT_BASE_IMAGE_SIZE // GRID_SIZE
MNIST_IMG_SIZE = 28 # MNIST digits are 28x28

# --- Helper Functions ---
def _order_points(pts):
    """Orders 4 points: top-left, top-right, bottom-right, bottom-left."""
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def _load_mnist_digits():
    """Loads MNIST digits and organizes them by label."""
    print("Loading MNIST dataset...")
    try:
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    except Exception as e:
        print(f"Error loading MNIST: {e}")
        print("Please ensure you have an internet connection or the dataset is cached.")
        # Fallback: Create empty dictionary if MNIST fails
        return {i: [] for i in range(10)}

    mnist_digits = {i: [] for i in range(10)}
    for img, label in zip(np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test))):
        # Invert MNIST (black background to white background like paper)
        # and add a border to prevent digits touching edges after resize
        img_inverted = cv2.bitwise_not(img)
        # Pad slightly before potential resizing later
        img_padded = cv2.copyMakeBorder(img_inverted, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255) # White border
        mnist_digits[label].append(img_padded)
    print(f"Loaded {sum(len(v) for v in mnist_digits.values())} MNIST digits.")
    return mnist_digits

class SudokuRenderer:
    """
    Generates synthetic Sudoku images with various configurable parameters.
    """
    def __init__(self,
                 base_image_size=DEFAULT_BASE_IMAGE_SIZE,
                 use_mnist=True,
                 use_fonts=True,
                 font_faces=None,
                 line_thickness_range=(1, 5),
                 digit_size_range=(0.5, 0.8), # Relative to cell size
                 digit_rotation_range=(-10, 10), # Degrees
                 digit_offset_range=(-0.1, 0.1), # Relative to cell size
                 perspective_warp_range=(0.05, 0.20), # Fraction of image size
                 noise_level_range=(5, 20), # Std dev for Gaussian noise
                 background_color_range=((200, 240), (200, 240), (200, 240)) # BGR ranges
                 ):
        """
        Initializes the SudokuRenderer.

        Args:
            base_image_size: Size of the square canvas before perspective warp.
            use_mnist: Whether to use MNIST digits.
            use_fonts: Whether to use OpenCV fonts for digits.
            font_faces: List of cv2 font constants to use if use_fonts is True.
            line_thickness_range: Min/max thickness for grid lines.
            digit_size_range: Min/max scale factor for digits relative to cell size.
            digit_rotation_range: Min/max rotation angle for digits.
            digit_offset_range: Min/max random offset for digits relative to cell size.
            perspective_warp_range: Min/max intensity for perspective distortion.
            noise_level_range: Min/max standard deviation for Gaussian noise.
            background_color_range: Tuple of (min, max) for each BGR channel.
        """
        if not use_mnist and not use_fonts:
            raise ValueError("Must use at least one digit source (MNIST or fonts).")

        self.base_image_size = base_image_size
        self.cell_draw_size = base_image_size // GRID_SIZE
        self.use_mnist = use_mnist
        self.use_fonts = use_fonts
        self.line_thickness_range = line_thickness_range
        self.digit_size_range = digit_size_range
        self.digit_rotation_range = digit_rotation_range
        self.digit_offset_range = digit_offset_range
        self.perspective_warp_range = perspective_warp_range
        self.noise_level_range = noise_level_range
        self.background_color_range = background_color_range

        if use_fonts:
            self.font_faces = font_faces or [
                cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN,
                cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX,
                cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_SIMPLEX # Weight simplex more
            ]
        else:
            self.font_faces = []

        if use_mnist:
            self.mnist_digits = _load_mnist_digits()
        else:
            self.mnist_digits = {i: [] for i in range(10)} # Empty dict if not used

    def _get_random_digit_image(self, digit):
        """Selects a random image for the given digit (1-9) from available sources."""
        sources = []
        if self.use_mnist and digit in self.mnist_digits and self.mnist_digits[digit]:
            sources.append("mnist")
        if self.use_fonts:
            sources.append("font")

        if not sources:
             # Fallback: render with font even if use_fonts was false, if MNIST failed/empty
             if not self.font_faces: self.font_faces = [cv2.FONT_HERSHEY_SIMPLEX] # Ensure one font
             source = "font"
        else:
            source = random.choice(sources)

        if source == "mnist":
            # Select a random MNIST image for this digit
            img = random.choice(self.mnist_digits[digit])
            # Ensure it's BGR for consistency
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img, "mnist"
        else: # source == "font"
            # Render using OpenCV font - we'll do this directly on the canvas later
            return None, "font"


    def render_sudoku(self, grid_spec=None, allow_empty=True):
        """
        Generates a synthetic Sudoku image based on the grid specification.

        Args:
            grid_spec (list[list[int | None]] | None): A 9x9 list of lists.
                Each element is an int (1-9) or None/0 for an empty cell.
                If None, a grid with random digits (respecting allow_empty) is generated.
            allow_empty (bool): If grid_spec is None, allows generating empty cells.

        Returns:
            tuple: (warped_image, ground_truth_grid, warped_corners)
                - warped_image (np.ndarray): The generated BGR image.
                - ground_truth_grid (np.ndarray): 9x9 NumPy array of the digits placed (0 for empty).
                - warped_corners (np.ndarray): (4, 2) array of corner coordinates in the warped image.
        """
        # 1. Create Ground Truth Grid if not provided
        if grid_spec is None:
            ground_truth_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    if allow_empty and random.random() < 0.4: # ~40% empty cells
                        ground_truth_grid[r, c] = 0
                    else:
                        ground_truth_grid[r, c] = random.randint(1, 9)
        else:
            # Convert input grid_spec to numpy array, ensuring None becomes 0
            ground_truth_grid = np.array([[d if d is not None else 0 for d in row] for row in grid_spec], dtype=int)
            if ground_truth_grid.shape != (GRID_SIZE, GRID_SIZE):
                raise ValueError(f"grid_spec must be a {GRID_SIZE}x{GRID_SIZE} list or array.")

        # 2. Create Base Canvas
        bg_color = tuple(random.randint(min_val, max_val) for min_val, max_val in self.background_color_range)
        image = np.full((self.base_image_size, self.base_image_size, 3), bg_color, dtype=np.uint8)

        # 3. Draw Grid Lines
        line_color = (0, 0, 0) # Black
        min_line, max_line = self.line_thickness_range
        for i in range(GRID_SIZE + 1):
            thickness = random.randint(min_line, max_line -1) # Normal lines
            if i % 3 == 0: # Thicker lines for 3x3 blocks
                thickness = random.randint(max(min_line, max_line -2), max_line)

            # Horizontal
            cv2.line(image, (0, i * self.cell_draw_size), (self.base_image_size, i * self.cell_draw_size), line_color, thickness)
            # Vertical
            cv2.line(image, (i * self.cell_draw_size, 0), (i * self.cell_draw_size, self.base_image_size), line_color, thickness)

        # 4. Place Digits in Cells
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                digit = ground_truth_grid[r, c]
                if digit == 0: # Skip empty cells
                    continue

                digit_img, source = self._get_random_digit_image(digit)

                # Calculate target size for the digit within the cell
                scale = random.uniform(*self.digit_size_range)
                target_h = int(self.cell_draw_size * scale)
                target_w = int(self.cell_draw_size * scale) # Keep aspect ratio roughly square for simplicity

                # Cell center coordinates
                cell_center_x = c * self.cell_draw_size + self.cell_draw_size // 2
                cell_center_y = r * self.cell_draw_size + self.cell_draw_size // 2

                # Random offset
                offset_x = int(random.uniform(*self.digit_offset_range) * self.cell_draw_size)
                offset_y = int(random.uniform(*self.digit_offset_range) * self.cell_draw_size)
                final_center_x = cell_center_x + offset_x
                final_center_y = cell_center_y + offset_y

                # Random rotation
                angle = random.uniform(*self.digit_rotation_range)

                if source == "mnist" and digit_img is not None:
                    # Resize MNIST image
                    resized_digit = cv2.resize(digit_img, (target_w, target_h), interpolation=cv2.INTER_AREA)

                    # Rotate the digit image
                    M = cv2.getRotationMatrix2D((target_w / 2, target_h / 2), angle, 1.0)
                    rotated_digit = cv2.warpAffine(resized_digit, M, (target_w, target_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)) # White background

                    # Create a mask for blending (assuming white background in rotated_digit)
                    gray_digit = cv2.cvtColor(rotated_digit, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(gray_digit, 250, 255, cv2.THRESH_BINARY_INV) # Mask is black digit on white bg

                    # Calculate top-left corner for pasting
                    paste_x = final_center_x - target_w // 2
                    paste_y = final_center_y - target_h // 2

                    # Ensure paste coordinates are within bounds
                    paste_x = max(0, paste_x)
                    paste_y = max(0, paste_y)

                    # Get the region of interest (ROI) on the main image
                    roi = image[paste_y : paste_y + target_h, paste_x : paste_x + target_w]

                    # Adjust digit/mask size if pasting goes out of bounds
                    roi_h, roi_w = roi.shape[:2]
                    if roi_h != target_h or roi_w != target_w:
                        rotated_digit = rotated_digit[:roi_h, :roi_w]
                        mask = mask[:roi_h, :roi_w]

                    # Blend using the mask
                    inv_mask = cv2.bitwise_not(mask)
                    img_bg = cv2.bitwise_and(roi, roi, mask=inv_mask)
                    img_fg = cv2.bitwise_and(rotated_digit, rotated_digit, mask=mask)
                    dst = cv2.add(img_bg, img_fg)
                    image[paste_y : paste_y + target_h, paste_x : paste_x + target_w] = dst

                elif source == "font":
                    font_face = random.choice(self.font_faces)
                    digit_str = str(digit)
                    # Adjust font scale dynamically to roughly match target size
                    # This is approximate and might need tuning
                    font_scale = cv2.getFontScaleFromHeight(font_face, target_h, thickness=2) # Estimate scale
                    font_thickness = random.randint(1, 3)
                    text_color = (random.randint(0,50), random.randint(0,50), random.randint(0,50)) # Dark color

                    (text_width, text_height), baseline = cv2.getTextSize(digit_str, font_face, font_scale, font_thickness)

                    # Calculate origin for putText (bottom-left)
                    origin_x = final_center_x - text_width // 2
                    origin_y = final_center_y + text_height // 2

                    # Apply rotation using warpAffine for text is complex,
                    # cv2.putText doesn't support rotation directly.
                    # For simplicity, we'll skip rotation for font-based digits for now,
                    # or accept that the offset/size variation provides some diversity.
                    # TODO: Implement text rotation if crucial (e.g., render text on separate canvas, rotate, then blend)

                    cv2.putText(image, digit_str, (origin_x, origin_y), font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

        # 5. Add Gaussian Noise
        mean = 0
        std_dev = random.uniform(*self.noise_level_range)
        noise = np.random.normal(mean, std_dev, image.shape).astype(np.float32)
        noisy_image = np.clip(image.astype(np.float32) + noise, 0, 255)
        image = noisy_image.astype(np.uint8)

        # 6. Apply Perspective Warp
        h, w = image.shape[:2]
        original_corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")

        warp_intensity = random.uniform(*self.perspective_warp_range)
        max_shift_x = w * warp_intensity
        max_shift_y = h * warp_intensity

        # Create more varied perspective shifts, including potentially stronger top/bottom shifts
        # to simulate low angles
        shifted_corners = np.array([
            [random.uniform(0, max_shift_x * 0.8), random.uniform(0, max_shift_y)], # Top-left
            [w - 1 - random.uniform(0, max_shift_x * 0.8), random.uniform(0, max_shift_y)], # Top-right
            [w - 1 - random.uniform(-max_shift_x * 0.2, max_shift_x), h - 1 - random.uniform(0, max_shift_y * 0.5)], # Bottom-right
            [random.uniform(-max_shift_x * 0.2, max_shift_x), h - 1 - random.uniform(0, max_shift_y * 0.5)]  # Bottom-left
        ], dtype="float32")

        # Ensure corners don't collapse or go wildly out of bounds (basic sanity check)
        min_dist = w * 0.1 # Minimum distance between adjacent corners horizontally
        if abs(shifted_corners[0, 0] - shifted_corners[1, 0]) < min_dist or \
           abs(shifted_corners[3, 0] - shifted_corners[2, 0]) < min_dist:
            # Reset to less aggressive warp if corners get too close horizontally
             shifted_corners = original_corners + np.random.uniform(-w*0.05, w*0.05, size=(4,2))
             shifted_corners = np.clip(shifted_corners, 0, [w-1, h-1])
             shifted_corners = shifted_corners.astype("float32")


        matrix = cv2.getPerspectiveTransform(original_corners, shifted_corners)
        # Determine output size based on the max extent of shifted corners
        # This makes the warped image contain the whole grid without excessive padding
        x_coords = shifted_corners[:, 0]
        y_coords = shifted_corners[:, 1]
        out_w = int(np.ceil(max(x_coords)))
        out_h = int(np.ceil(max(y_coords)))
        out_w = max(out_w, 100) # Ensure minimum size
        out_h = max(out_h, 100)

        # Adjust transform matrix for the new output size if needed (usually not necessary here)
        # matrix[0, 2] -= min(x_coords) # Shift origin if min_x < 0 (not typical here)
        # matrix[1, 2] -= min(y_coords) # Shift origin if min_y < 0

        warped_image = cv2.warpPerspective(image, matrix, (out_w, out_h), borderMode=cv2.BORDER_REPLICATE)

        # The 'warped_corners' should correspond to the original grid corners in the *new* warped image coordinates
        # Since warpPerspective calculates coordinates relative to the output size, the shifted_corners are already correct.
        final_warped_corners = shifted_corners

        return warped_image, ground_truth_grid, final_warped_corners

# --- Example Usage (for testing) ---
if __name__ == "__main__":
    print("Testing SudokuRenderer...")
    renderer = SudokuRenderer(use_mnist=True, use_fonts=True)

    # Example 1: Generate a random grid
    print("Generating random Sudoku image...")
    random_img, random_gt, random_corners = renderer.render_sudoku(allow_empty=True)
    print("Ground Truth Grid (Random):")
    print(random_gt)
    print("Warped Corners (Random):")
    print(random_corners)
    cv2.imwrite("rendered_sudoku_random.png", random_img)
    print("Saved random_img to rendered_sudoku_random.png")

    # Example 2: Generate a specific grid
    print("\nGenerating specific Sudoku image...")
    specific_grid_spec = [
        [5, 3, None, None, 7, None, None, None, None],
        [6, None, None, 1, 9, 5, None, None, None],
        [None, 9, 8, None, None, None, None, 6, None],
        [8, None, None, None, 6, None, None, None, 3],
        [4, None, None, 8, None, 3, None, None, 1],
        [7, None, None, None, 2, None, None, None, 6],
        [None, 6, None, None, None, None, 2, 8, None],
        [None, None, None, 4, 1, 9, None, None, 5],
        [None, None, None, None, 8, None, None, 7, 9]
    ]
    specific_img, specific_gt, specific_corners = renderer.render_sudoku(grid_spec=specific_grid_spec)
    print("Ground Truth Grid (Specific):")
    print(specific_gt)
    cv2.imwrite("rendered_sudoku_specific.png", specific_img)
    print("Saved specific_img to rendered_sudoku_specific.png")

    # Draw corners on the specific image for visualization
    img_with_corners = specific_img.copy()
    for i, p in enumerate(specific_corners):
        pt = tuple(p.astype(int))
        cv2.circle(img_with_corners, pt, 10, (0, 0, 255), -1) # Red circles
        cv2.putText(img_with_corners, str(i), (pt[0]+5, pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2) # Blue text
    cv2.imwrite("rendered_sudoku_specific_corners.png", img_with_corners)
    print("Saved specific_img with corners to rendered_sudoku_specific_corners.png")

    print("\nRenderer test complete.")