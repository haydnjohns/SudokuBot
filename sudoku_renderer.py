# sudoku_renderer.py
import cv2
import numpy as np
import random
import os
from pathlib import Path
import keras # For MNIST dataset loading

# --- Constants ---
GRID_SIZE = 9
DEFAULT_BASE_IMAGE_SIZE = 1000 # Initial canvas size before perspective warp
DEFAULT_CELL_DRAW_SIZE = DEFAULT_BASE_IMAGE_SIZE // GRID_SIZE
MNIST_IMG_SIZE = 28 # Standard size of MNIST digit images

# --- Helper Functions ---
def _order_points(pts):
    """
    Orders 4 points found for a contour: top-left, top-right, bottom-right, bottom-left.
    (Identical to the one in digit_extractor.py, kept here for renderer independence if needed)
    """
    if pts.shape != (4, 2):
         try: pts = pts.reshape(4, 2)
         except ValueError: raise ValueError(f"Input 'pts' must be reshapeable to (4, 2). Got shape {pts.shape}")
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Top-left
    rect[2] = pts[np.argmax(s)] # Bottom-right
    diff_yx = pts[:, 1] - pts[:, 0]
    rect[1] = pts[np.argmin(diff_yx)] # Top-right
    rect[3] = pts[np.argmax(diff_yx)] # Bottom-left
    return rect

def _load_mnist_digits():
    """
    Loads the MNIST dataset using Keras and organizes digit images by label (0-9).
    Applies basic preprocessing (inversion, padding).

    Returns:
        dict: A dictionary where keys are digits (0-9) and values are lists of
              corresponding preprocessed MNIST image arrays (NumPy). Returns an empty
              dictionary if loading fails.
    """
    print("Loading MNIST dataset...")
    try:
        # Load MNIST data (train and test sets)
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        all_images = np.concatenate((x_train, x_test))
        all_labels = np.concatenate((y_train, y_test))
    except Exception as e:
        print(f"[Error] Failed to load MNIST dataset: {e}")
        print("Please ensure Keras/TensorFlow is installed and you have an internet connection.")
        # Return an empty structure if loading fails
        return {i: [] for i in range(10)}

    mnist_digits = {i: [] for i in range(10)}
    for img, label in zip(all_images, all_labels):
        # Preprocess MNIST images for rendering:
        # 1. Invert: MNIST is black digit on white bg, we want black on white paper-like bg
        img_inverted = cv2.bitwise_not(img)
        # 2. Pad: Add a white border to prevent digits touching cell edges after resize/rotation
        img_padded = cv2.copyMakeBorder(img_inverted, 4, 4, 4, 4, # Padding size (top, bottom, left, right)
                                        cv2.BORDER_CONSTANT, value=255) # White border
        mnist_digits[label].append(img_padded)

    print(f"Loaded and preprocessed {len(all_images)} MNIST digits.")
    return mnist_digits

# --- Sudoku Renderer Class ---
class SudokuRenderer:
    """
    Generates synthetic Sudoku images with various augmentations like
    perspective warp, noise, different digit styles (MNIST/fonts), etc.
    """
    def __init__(self,
                 base_image_size=DEFAULT_BASE_IMAGE_SIZE,
                 use_mnist=True,
                 use_fonts=True,
                 font_faces=None,
                 line_thickness_range=(1, 5),
                 digit_size_range=(0.5, 0.8), # Relative to cell size
                 digit_rotation_range=(-10, 10), # Degrees
                 digit_offset_range=(-0.1, 0.1), # Relative to cell center
                 perspective_warp_range=(0.05, 0.20), # Fraction of image size for corner shifts
                 noise_level_range=(5, 20), # Std dev range for Gaussian noise
                 background_color_range=((200, 240), (200, 240), (200, 240)) # BGR min/max ranges
                 ):
        """
        Initializes the SudokuRenderer with configuration parameters.

        Args:
            base_image_size (int): Size of the square canvas before perspective warp.
            use_mnist (bool): Allow using MNIST digits for rendering.
            use_fonts (bool): Allow using OpenCV system fonts for rendering.
            font_faces (list | None): List of cv2 font constants (e.g., cv2.FONT_HERSHEY_SIMPLEX).
                                      Defaults to a predefined list if None.
            line_thickness_range (tuple): (min, max) thickness for grid lines.
            digit_size_range (tuple): (min, max) scale factor for digits relative to cell size.
            digit_rotation_range (tuple): (min, max) rotation angle in degrees for digits.
            digit_offset_range (tuple): (min, max) random offset for digits relative to cell center.
            perspective_warp_range (tuple): (min, max) intensity factor for perspective distortion.
            noise_level_range (tuple): (min, max) standard deviation for Gaussian noise.
            background_color_range (tuple): Tuple of ((min_B, max_B), (min_G, max_G), (min_R, max_R)).
        """
        if not use_mnist and not use_fonts:
            raise ValueError("Must enable at least one digit source (use_mnist=True or use_fonts=True).")

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

        # Setup font faces if fonts are enabled
        if use_fonts:
            self.font_faces = font_faces or [
                cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN,
                cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX,
                cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                # Weight Simplex slightly more by including it multiple times
                cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_SIMPLEX
            ]
        else:
            self.font_faces = []

        # Lazy loading for MNIST data: load only when first accessed
        self._mnist_digits = None if use_mnist else {} # Use empty dict if MNIST disabled

    @property
    def mnist_digits(self):
        """Property to access MNIST digits, triggers loading on first access if enabled."""
        if self.use_mnist and self._mnist_digits is None:
            self._mnist_digits = _load_mnist_digits()
        # Return the loaded digits or an empty dict if MNIST is disabled/failed
        return self._mnist_digits if self._mnist_digits is not None else {}

    def _get_random_digit_image(self, digit):
        """
        Selects a random image source (MNIST or font) and returns either the
        MNIST image array or None (indicating font should be used).

        Args:
            digit (int): The digit (1-9) to render.

        Returns:
            tuple: (image_array | None, source_type_str)
                   - np.ndarray: A BGR MNIST digit image if MNIST is chosen.
                   - None: If font rendering is chosen.
                   - str: "mnist" or "font" indicating the chosen source.
        """
        available_sources = []
        # Check if MNIST is enabled and has images for the requested digit
        if self.use_mnist and digit in self.mnist_digits and self.mnist_digits[digit]:
            available_sources.append("mnist")
        # Check if fonts are enabled
        if self.use_fonts and self.font_faces:
            available_sources.append("font")

        if not available_sources:
            # Fallback if somehow both sources are unavailable (should be prevented by __init__)
            print(f"[Warning] No digit sources available for digit {digit}. Check configuration.")
            # Default to font rendering attempt if possible, otherwise return None
            if self.font_faces:
                 return None, "font"
            else:
                 return None, "none" # Indicate failure

        # Choose randomly between available sources
        chosen_source = random.choice(available_sources)

        if chosen_source == "mnist":
            # Select a random instance of the digit from the loaded MNIST data
            img = random.choice(self.mnist_digits[digit])
            # Ensure image is BGR (MNIST is loaded as grayscale)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img, "mnist"
        else: # chosen_source == "font"
            return None, "font"


    def render_sudoku(self, grid_spec=None, allow_empty=True):
        """
        Generates a synthetic Sudoku image based on a grid specification or randomly.

        Args:
            grid_spec (list[list[int | None]] | np.ndarray | None):
                A 9x9 specification of the grid. Integers 1-9 represent digits,
                while 0 or None represent empty cells. If None, a random grid is generated.
            allow_empty (bool): If grid_spec is None, controls whether random generation
                                includes empty cells (True) or fills all cells (False).

        Returns:
            tuple: (warped_image, ground_truth_grid, warped_corners)
                - warped_image (np.ndarray | None): The generated BGR image, or None on failure.
                - ground_truth_grid (np.ndarray): 9x9 NumPy array of the digits placed (0 for empty).
                - warped_corners (np.ndarray | None): (4, 2) array of corner coordinates in the
                                                     final warped image, or None on failure.
        """
        try:
            # 1. Determine Ground Truth Grid
            if grid_spec is None:
                # Generate a random grid
                ground_truth_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
                for r in range(GRID_SIZE):
                    for c in range(GRID_SIZE):
                        # Fill cell randomly based on allow_empty probability
                        if allow_empty and random.random() < 0.4: # Approx 40% empty cells
                            ground_truth_grid[r, c] = 0
                        else:
                            ground_truth_grid[r, c] = random.randint(1, 9)
            else:
                # Use the provided grid specification
                try:
                    # Convert spec to numpy array, handling None as 0
                    ground_truth_grid = np.array([[d if d is not None else 0 for d in row] for row in grid_spec], dtype=int)
                    if ground_truth_grid.shape != (GRID_SIZE, GRID_SIZE):
                        raise ValueError(f"grid_spec must be {GRID_SIZE}x{GRID_SIZE}.")
                except (TypeError, ValueError) as e:
                    print(f"[Error] Invalid grid_spec provided: {e}")
                    return None, None, None

            # 2. Create Base Canvas
            # Random background color within specified ranges
            bg_color = tuple(random.randint(min_val, max_val) for min_val, max_val in self.background_color_range)
            image = np.full((self.base_image_size, self.base_image_size, 3), bg_color, dtype=np.uint8)

            # 3. Draw Grid Lines
            line_color = (0, 0, 0) # Black lines
            min_line, max_line = self.line_thickness_range
            for i in range(GRID_SIZE + 1):
                # Make major grid lines (every 3rd) potentially thicker
                is_major_line = (i % 3 == 0)
                thickness = random.randint(max(min_line, max_line - 2), max_line) if is_major_line else random.randint(min_line, max(min_line, max_line - 1))
                thickness = max(1, thickness) # Ensure thickness is at least 1

                # Draw horizontal line
                pt1_h = (0, i * self.cell_draw_size)
                pt2_h = (self.base_image_size, i * self.cell_draw_size)
                cv2.line(image, pt1_h, pt2_h, line_color, thickness)
                # Draw vertical line
                pt1_v = (i * self.cell_draw_size, 0)
                pt2_v = (i * self.cell_draw_size, self.base_image_size)
                cv2.line(image, pt1_v, pt2_v, line_color, thickness)

            # 4. Place Digits onto the Canvas
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    digit = ground_truth_grid[r, c]
                    if digit == 0: continue # Skip empty cells

                    # Get digit image (MNIST) or signal to use font
                    digit_img_mnist, source_type = self._get_random_digit_image(digit)

                    # Randomize digit properties
                    scale = random.uniform(*self.digit_size_range)
                    target_h = int(self.cell_draw_size * scale)
                    target_w = int(self.cell_draw_size * scale) # Keep aspect ratio for font scaling later
                    angle = random.uniform(*self.digit_rotation_range)
                    offset_x = int(random.uniform(*self.digit_offset_range) * self.cell_draw_size)
                    offset_y = int(random.uniform(*self.digit_offset_range) * self.cell_draw_size)

                    # Calculate target position (center of cell + offset)
                    cell_center_x = c * self.cell_draw_size + self.cell_draw_size // 2
                    cell_center_y = r * self.cell_draw_size + self.cell_draw_size // 2
                    final_center_x = cell_center_x + offset_x
                    final_center_y = cell_center_y + offset_y

                    if source_type == "mnist" and digit_img_mnist is not None:
                        # --- Render using MNIST digit ---
                        # Resize the MNIST image
                        resized_digit = cv2.resize(digit_img_mnist, (target_w, target_h), interpolation=cv2.INTER_AREA)
                        # Rotate the resized digit
                        M = cv2.getRotationMatrix2D((target_w / 2, target_h / 2), angle, 1.0)
                        # Use white border to fill areas exposed by rotation
                        rotated_digit = cv2.warpAffine(resized_digit, M, (target_w, target_h),
                                                       borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

                        # Create a mask from the rotated digit (assuming black digit on white bg)
                        gray_digit = cv2.cvtColor(rotated_digit, cv2.COLOR_BGR2GRAY)
                        # Threshold to get mask (digit is black/dark, background is white)
                        # Adjust threshold value (e.g., 250) if needed based on MNIST preprocessing
                        _, mask = cv2.threshold(gray_digit, 250, 255, cv2.THRESH_BINARY_INV)

                        # Calculate paste position (top-left corner)
                        paste_x = max(0, final_center_x - target_w // 2)
                        paste_y = max(0, final_center_y - target_h // 2)

                        # Define Region of Interest (ROI) on the main image canvas
                        roi = image[paste_y : paste_y + target_h, paste_x : paste_x + target_w]
                        roi_h, roi_w = roi.shape[:2]

                        # Adjust digit/mask size if ROI is smaller than target (due to edge proximity)
                        if roi_h != target_h or roi_w != target_w:
                            rotated_digit = rotated_digit[:roi_h, :roi_w]
                            mask = mask[:roi_h, :roi_w]

                        # Blend the digit onto the canvas using the mask
                        if mask.shape[0] == roi_h and mask.shape[1] == roi_w: # Final check for size match
                            inv_mask = cv2.bitwise_not(mask)
                            img_bg = cv2.bitwise_and(roi, roi, mask=inv_mask) # Keep background where mask is off
                            img_fg = cv2.bitwise_and(rotated_digit, rotated_digit, mask=mask) # Keep digit where mask is on
                            dst = cv2.add(img_bg, img_fg) # Combine background and foreground
                            image[paste_y : paste_y + roi_h, paste_x : paste_x + roi_w] = dst
                        # else: print warning or skip if sizes mismatch significantly

                    elif source_type == "font":
                        # --- Render using OpenCV font ---
                        font_face = random.choice(self.font_faces)
                        digit_str = str(digit)
                        font_thickness = random.randint(1, 3)
                        # Estimate font scale to roughly match target height
                        # This is approximate and might need fine-tuning per font
                        font_scale = cv2.getFontScaleFromHeight(font_face, target_h, thickness=font_thickness) * 0.8
                        # Random dark color for the text
                        text_color = (random.randint(0,50), random.randint(0,50), random.randint(0,50))

                        # Get text size to center it accurately
                        (text_width, text_height), baseline = cv2.getTextSize(digit_str, font_face, font_scale, font_thickness)
                        # Calculate text origin (bottom-left corner for cv2.putText)
                        origin_x = final_center_x - text_width // 2
                        origin_y = final_center_y + text_height // 2

                        # Draw the text (rotation is not applied for simplicity with fonts)
                        cv2.putText(image, digit_str, (origin_x, origin_y), font_face, font_scale,
                                    text_color, font_thickness, cv2.LINE_AA)

            # 5. Add Gaussian Noise
            mean = 0
            std_dev = random.uniform(*self.noise_level_range)
            # Generate noise with same shape as image, ensure float type for addition
            noise = np.random.normal(mean, std_dev, image.shape).astype(np.float32)
            # Add noise and clip values to valid range [0, 255]
            noisy_image_float = image.astype(np.float32) + noise
            noisy_image = np.clip(noisy_image_float, 0, 255).astype(np.uint8)
            image = noisy_image # Use the noisy image for subsequent steps

            # 6. Apply Perspective Warp
            h, w = image.shape[:2]
            # Define original corners of the flat image
            original_corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")

            # Calculate maximum shift based on warp intensity range
            warp_intensity = random.uniform(*self.perspective_warp_range)
            max_shift_x = w * warp_intensity
            max_shift_y = h * warp_intensity

            # Generate random shifts for each corner within calculated limits
            # Adjust limits per corner to create realistic perspective (e.g., top corners shift less horizontally)
            shifted_corners = np.array([
                [random.uniform(0, max_shift_x * 0.8), random.uniform(0, max_shift_y)], # Top-left
                [w - 1 - random.uniform(0, max_shift_x * 0.8), random.uniform(0, max_shift_y)], # Top-right
                [w - 1 - random.uniform(-max_shift_x * 0.2, max_shift_x), h - 1 - random.uniform(0, max_shift_y * 0.5)], # Bottom-right
                [random.uniform(-max_shift_x * 0.2, max_shift_x), h - 1 - random.uniform(0, max_shift_y * 0.5)] # Bottom-left
            ], dtype="float32")

            # Sanity check: prevent extreme collapses (e.g., top edge becoming very short)
            min_edge_length = w * 0.1 # Minimum allowed length for top/bottom edges relative to width
            if abs(shifted_corners[0, 0] - shifted_corners[1, 0]) < min_edge_length or \
               abs(shifted_corners[3, 0] - shifted_corners[2, 0]) < min_edge_length:
                 # If collapsed, apply a simpler, less intense random jitter instead
                 print("[Renderer WARN] Extreme perspective warp detected, applying simple jitter instead.")
                 jitter_amount = w * 0.05
                 shifted_corners = original_corners + np.random.uniform(-jitter_amount, jitter_amount, size=(4,2))
                 # Ensure corners stay within image bounds after jitter
                 shifted_corners[:, 0] = np.clip(shifted_corners[:, 0], 0, w - 1)
                 shifted_corners[:, 1] = np.clip(shifted_corners[:, 1], 0, h - 1)
                 shifted_corners = shifted_corners.astype("float32")

            # Calculate the perspective transformation matrix
            matrix = cv2.getPerspectiveTransform(original_corners, shifted_corners)

            # Determine the output size needed to contain the warped image
            x_coords, y_coords = shifted_corners[:, 0], shifted_corners[:, 1]
            out_w = max(100, int(np.ceil(max(x_coords)))) # Ensure minimum size
            out_h = max(100, int(np.ceil(max(y_coords))))

            # Apply the perspective warp
            # Use BORDER_REPLICATE to avoid black areas at edges after warp
            warped_image = cv2.warpPerspective(image, matrix, (out_w, out_h), borderMode=cv2.BORDER_REPLICATE)

            # The final corner positions in the warped image are the shifted_corners
            final_warped_corners = shifted_corners

            return warped_image, ground_truth_grid, final_warped_corners

        except Exception as e:
            print(f"[Error] Failed during Sudoku rendering: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None


# --- Test Example Generation ---
def generate_and_save_test_example(filename_prefix="epoch_test_sudoku", force_regenerate=False):
    """
    Generates a consistent Sudoku image and its ground truth grid, saving them
    to files. Used for repeatable testing, e.g., in the EpochTestCallback.

    Args:
        filename_prefix (str): Base name for the output image (.png) and ground truth (.npy) files.
        force_regenerate (bool): If True, always generates new files, overwriting existing ones.

    Returns:
        tuple: (image_path_str, ground_truth_grid)
               - str: Path to the generated (or existing) image file.
               - np.ndarray: The 9x9 ground truth grid.

    Raises:
        RuntimeError: If image rendering fails or saving/loading fails unexpectedly.
    """
    img_path = Path(f"{filename_prefix}.png")
    gt_path = Path(f"{filename_prefix}_gt.npy")

    # Check if files exist and regeneration is not forced
    if not force_regenerate and img_path.exists() and gt_path.exists():
        print(f"Using existing test example: {img_path}, {gt_path}")
        try:
            # Load existing ground truth
            ground_truth_grid = np.load(gt_path)
            # Validate shape
            if ground_truth_grid.shape == (GRID_SIZE, GRID_SIZE):
                return str(img_path), ground_truth_grid
            else:
                print(f"[Warning] Existing ground truth file {gt_path} has incorrect shape {ground_truth_grid.shape}. Regenerating.")
        except Exception as e:
            print(f"[Warning] Error loading existing ground truth {gt_path}: {e}. Regenerating.")

    # Generate new test example
    print(f"Generating new test example: {img_path}, {gt_path}")

    # Define a fixed, reasonably complex Sudoku grid for consistency
    test_grid_spec = [
        [None, None, 3, None, None, 6, None, 8, None],
        [8, None, 1, None, 3, None, 5, None, 4],
        [None, 4, None, 8, None, 7, None, 1, None],
        [1, None, None, 4, None, 5, None, None, 9],
        [None, 7, None, None, 2, None, None, 4, None],
        [5, None, None, 7, None, 1, None, None, 3],
        [None, 8, None, 5, None, 3, None, 9, None],
        [7, None, 4, None, 9, None, 1, None, 8],
        [None, 1, None, 6, None, None, 4, None, None]
    ]

    # Use default renderer settings for consistency
    renderer = SudokuRenderer()
    rendered_img, ground_truth_grid, _ = renderer.render_sudoku(grid_spec=test_grid_spec)

    if rendered_img is None or ground_truth_grid is None:
        raise RuntimeError("Failed to render the test Sudoku image.")

    # Save the generated image and ground truth grid
    try:
        cv2.imwrite(str(img_path), rendered_img)
        np.save(gt_path, ground_truth_grid)
        print(f"Saved new test example image to {img_path}")
        print(f"Saved new test example ground truth to {gt_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save test example files: {e}")

    return str(img_path), ground_truth_grid


# --- Example Usage (__main__) ---
if __name__ == "__main__":
    print("Testing SudokuRenderer...")

    # Ensure the standard test example exists (or is generated)
    try:
        print("\nGenerating/Verifying standard test example...")
        generate_and_save_test_example(force_regenerate=False) # Don't force unless needed
    except Exception as e:
        print(f"[ERROR] Could not generate or verify the standard test example: {e}")

    # Initialize renderer with both MNIST and fonts enabled
    renderer = SudokuRenderer(use_mnist=True, use_fonts=True)

    # Example 1: Generate a random grid with empty cells allowed
    print("\nGenerating random Sudoku image (with empty cells)...")
    random_img, random_gt, random_corners = renderer.render_sudoku(allow_empty=True)
    if random_img is not None:
        print("Ground Truth Grid (Random):")
        print(random_gt)
        # print("Warped Corners (Random):") # Optional: print corner coords
        # print(random_corners)
        save_path_random = "rendered_sudoku_random.png"
        cv2.imwrite(save_path_random, random_img)
        print(f"Saved random Sudoku image to '{save_path_random}'")
    else:
        print("Failed to generate random Sudoku image.")

    # Example 2: Generate a specific, predefined grid
    print("\nGenerating specific Sudoku image...")
    # Standard example puzzle
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
    if specific_img is not None:
        print("Ground Truth Grid (Specific):")
        print(specific_gt)
        save_path_specific = "rendered_sudoku_specific.png"
        cv2.imwrite(save_path_specific, specific_img)
        print(f"Saved specific Sudoku image to '{save_path_specific}'")

        # Optional: Draw detected corners on the specific image for visualization
        img_with_corners = specific_img.copy()
        if specific_corners is not None:
            corner_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)] # BGR: Red, Green, Blue, Yellow
            for i, p in enumerate(specific_corners):
                pt = tuple(p.astype(int))
                color = corner_colors[i % len(corner_colors)]
                cv2.circle(img_with_corners, pt, 10, color, -1) # Draw filled circle
                # Add text label near the corner
                cv2.putText(img_with_corners, str(i), (pt[0] + 10, pt[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3, cv2.LINE_AA) # White text w/ black outline
                cv2.putText(img_with_corners, str(i), (pt[0] + 10, pt[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)

            save_path_corners = "rendered_sudoku_specific_corners.png"
            cv2.imwrite(save_path_corners, img_with_corners)
            print(f"Saved specific image with corners visualized to '{save_path_corners}'")
    else:
        print("Failed to generate specific Sudoku image.")

    print("\nRenderer test complete.")
