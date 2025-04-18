import cv2
import numpy as np
import random
# skimage is not needed if we are not processing cells
# from skimage.segmentation import clear_border
import os # For directory creation and path joining
import shutil # For cleaning up the sample directory if needed

# --- Constants ---
# NUM_IMAGES_PER_DIGIT = 1 # We only need 1 image per digit for the sample
GRID_SIZE = 9
# Base size for generating the initial image (before perspective warp)
BASE_IMAGE_SIZE = 1000
# Target size for the rectified Sudoku grid (square) after perspective correction
# RECTIFIED_GRID_SIZE = 900 # Not strictly needed for saving the warped image
# CELL_SIZE = RECTIFIED_GRID_SIZE // GRID_SIZE # Not needed
# Target size for the final processed cell images for the dataset
# FINAL_CELL_SIZE = (28, 28) # Not needed
# Directory to save sample images
SAMPLE_IMAGE_DIR = "sample_images" # Changed name to avoid confusion

# --- Helper Functions ---

# order_points is only needed for rectify_perspective, which we aren't using here.
# It can be kept if you might use it later, or removed. Let's keep it for now.
def order_points(pts):
    """
    Orders the 4 points of a contour/quadrilateral
    in top-left, top-right, bottom-right, bottom-left order.

    Args:
        pts (np.ndarray): Array of 4 points (x, y). Should be shape (4, 2).

    Returns:
        np.ndarray: Array of 4 points ordered correctly, shape (4, 2).
    """
    # Ensure input is shape (4, 2)
    if pts.shape != (4, 2):
         # Attempt to reshape if it's like (4, 1, 2) from findContours
         try:
              pts = pts.reshape(4, 2)
         except ValueError:
              raise ValueError(f"Input 'pts' must be reshapeable to (4, 2). Got shape {pts.shape}")

    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Top-left (smallest sum)
    rect[2] = pts[np.argmax(s)] # Bottom-right (largest sum)

    diff = np.array([p[1] - p[0] for p in pts])
    rect[1] = pts[np.argmin(diff)] # Top-right (smallest difference y-x)
    rect[3] = pts[np.argmax(diff)] # Bottom-left (largest difference y-x)

    return rect

# --- Synthetic Data Generation Functions ---

def generate_sudoku_image(digit):
    """
    Generates a single synthetic Sudoku image with all cells containing the same digit,
    adds noise, variable grid lines, and perspective warp.

    Args:
        digit (int): The digit (1-9) to fill the grid with.

    Returns:
        tuple: (warped_image, warped_corners_array)
               - warped_image (np.ndarray): The generated BGR image with perspective warp.
               - warped_corners_array (np.ndarray): The (4, 2) array of corner coordinates
                                                    corresponding to the original grid corners
                                                    in the warped image.
    """
    # 1. Create Base Canvas (Light Gray)
    background_color = (random.randint(200, 230), random.randint(200, 230), random.randint(200, 230)) # BGR
    image = np.full((BASE_IMAGE_SIZE, BASE_IMAGE_SIZE, 3), background_color, dtype=np.uint8)
    cell_draw_size = BASE_IMAGE_SIZE // GRID_SIZE

    # 2. Draw Grid Lines (Variable Thickness)
    line_color = (0, 0, 0) # Black
    for i in range(GRID_SIZE + 1):
        thickness = random.randint(1, 2)
        if i % 3 == 0: # Thicker lines for 3x3 blocks
            thickness = random.randint(3, 5)
        # Horizontal lines
        start_point_h = (0, i * cell_draw_size)
        end_point_h = (BASE_IMAGE_SIZE, i * cell_draw_size)
        cv2.line(image, start_point_h, end_point_h, line_color, thickness)
        # Vertical lines
        start_point_v = (i * cell_draw_size, 0)
        end_point_v = (i * cell_draw_size, BASE_IMAGE_SIZE)
        cv2.line(image, start_point_v, end_point_v, line_color, thickness)

    # 3. Place Digits in Cells (Random Font Size & Offset)
    digit_str = str(digit)
    font_face = random.choice([cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN,
                               cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX,
                               cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX])
    text_color = (0, 0, 0) # Black

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            font_scale = random.uniform(1.8, 3.0) # Randomize size
            font_thickness = random.randint(1, 3)  # Randomize thickness

            # Get text size to help centering
            (text_width, text_height), baseline = cv2.getTextSize(digit_str, font_face, font_scale, font_thickness)

            # Calculate base position (center of the cell)
            cell_center_x = c * cell_draw_size + cell_draw_size // 2
            cell_center_y = r * cell_draw_size + cell_draw_size // 2

            # Calculate bottom-left corner for cv2.putText (origin)
            origin_x = cell_center_x - text_width // 2
            origin_y = cell_center_y + text_height // 2 # Baseline is below text

            # Add random offset
            offset_x = random.randint(-cell_draw_size // 8, cell_draw_size // 8)
            offset_y = random.randint(-cell_draw_size // 8, cell_draw_size // 8)

            final_origin = (origin_x + offset_x, origin_y + offset_y)

            cv2.putText(image, digit_str, final_origin, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

    # 4. Add Gaussian Noise
    mean = 0
    std_dev = random.uniform(5, 15) # Randomize noise level
    noise = np.random.normal(mean, std_dev, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255) # Ensure values are within valid range
    image = noisy_image.astype(np.uint8)

    # 5. Apply Perspective Warp
    h, w = image.shape[:2]
    original_corners = np.array([
        [0, 0],         # Top-left
        [w - 1, 0],     # Top-right
        [w - 1, h - 1], # Bottom-right
        [0, h - 1]      # Bottom-left
    ], dtype="float32")

    max_shift_x = w * 0.1
    max_shift_y = h * 0.1

    shifted_corners = np.array([
        [random.uniform(0, max_shift_x), random.uniform(0, max_shift_y)],
        [w - 1 - random.uniform(0, max_shift_x), random.uniform(0, max_shift_y)],
        [w - 1 - random.uniform(-max_shift_x/2, max_shift_x/2), h - 1 - random.uniform(0, max_shift_y/2)],
        [random.uniform(-max_shift_x/2, max_shift_x/2), h - 1 - random.uniform(0, max_shift_y/2)]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(original_corners, shifted_corners)
    warped_image = cv2.warpPerspective(image, matrix, (w, h))

    return warped_image, shifted_corners


# --- Image Processing Functions ---
# These functions are NOT needed for saving the full warped image,
# but are kept here if you want to use the script for cell extraction later.

def rectify_perspective(warped_image, warped_corners, output_size):
    # ... (implementation remains the same as before) ...
    dst_pts = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1]
    ], dtype="float32")
    ordered_warped_pts = order_points(warped_corners)
    matrix = cv2.getPerspectiveTransform(ordered_warped_pts, dst_pts)
    rectified = cv2.warpPerspective(warped_image, matrix, (output_size, output_size))
    return rectified

def extract_cells(rectified_grid_image):
    # ... (implementation remains the same as before) ...
    cells = []
    current_cell_size = rectified_grid_image.shape[0] // GRID_SIZE
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            start_row = r * current_cell_size
            start_col = c * current_cell_size
            end_row = start_row + current_cell_size
            end_col = start_col + current_cell_size
            cell = rectified_grid_image[start_row:end_row, start_col:end_col]
            cells.append(cell)
    return cells

def preprocess_cell(cell_image, target_size=(28, 28)):
    # ... (implementation remains the same as before, including skimage import if used) ...
    # Note: If you remove skimage import, you need to handle the try/except block here
    # or remove the clear_border step.
    gray_cell = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    blurred_cell = cv2.GaussianBlur(gray_cell, (5, 5), 0)
    thresh_cell = cv2.adaptiveThreshold(blurred_cell, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    processed_cell = cv2.dilate(thresh_cell, kernel, iterations=1)
    # Optional clear_border step (requires skimage)
    # try:
    #     pad_width = 3
    #     padded_cell = cv2.copyMakeBorder(processed_cell, pad_width, pad_width, pad_width, pad_width, cv2.BORDER_CONSTANT, value=0)
    #     cleared_padded = clear_border(padded_cell)
    #     processed_cell = cleared_padded[pad_width:-pad_width, pad_width:-pad_width]
    # except NameError: # If clear_border wasn't imported or failed
    #      print("[WARN] clear_border not available or failed. Skipping.")
    # except Exception as e:
    #      print(f"[WARN] Error during clear_border: {e}. Skipping.")

    resized_cell = cv2.resize(processed_cell, target_size, interpolation=cv2.INTER_AREA)
    return resized_cell

# --- Main Data Generation Pipeline ---
if __name__ == "__main__":
    print("[INFO] Starting synthetic Sudoku full image generation...")

    # Create the target directory, removing it first if it exists
    if os.path.exists(SAMPLE_IMAGE_DIR):
        print(f"[INFO] Removing existing directory: {SAMPLE_IMAGE_DIR}")
        shutil.rmtree(SAMPLE_IMAGE_DIR)
    print(f"[INFO] Creating directory: {SAMPLE_IMAGE_DIR}")
    os.makedirs(SAMPLE_IMAGE_DIR, exist_ok=True)

    # Keep track if we saved a sample for each digit
    saved_samples = {d: False for d in range(1, 10)}

    # Loop through digits 1 to 9
    for digit in range(1, 10):
        print(f"[INFO] Generating full image for digit: {digit}")
        # Generate only the first image (i=0) for each digit
        i = 0 # Image index is always 0 for these samples

        # 1. Generate Synthetic Sudoku Image + Warp
        # We only need the warped image itself for saving
        warped_img, _ = generate_sudoku_image(digit) # Ignore warped_corners

        # --- We DO NOT rectify, extract cells, or preprocess for this task ---

        # 2. Construct Filename (without cell index)
        filename = f"digit_{digit}_img_{i}.png"
        filepath = os.path.join(SAMPLE_IMAGE_DIR, filename)

        # 3. Save the full warped image
        try:
            cv2.imwrite(filepath, warped_img)
            print(f"    Saved sample image: {filepath}")
            saved_samples[digit] = True
        except Exception as e:
            print(f"[ERROR] Failed to save image {filepath}: {e}")

    print("\n[INFO] Sample full image generation complete.")
    cv2.destroyAllWindows() # Close any debug windows left open

    # Verify if all samples were saved
    all_saved = all(saved_samples.values())
    if all_saved:
        print(f"[INFO] Successfully saved one sample full image for each digit (1-9) in '{SAMPLE_IMAGE_DIR}'.")
    else:
        print("[WARN] Failed to save sample full images for all digits. Check logs.")
        missing = [d for d, saved in saved_samples.items() if not saved]
        print(f"Missing samples for digits: {missing}")