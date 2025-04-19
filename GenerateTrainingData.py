import cv2
import numpy as np
import random
from PIL import ImageFont, ImageDraw, Image
from skimage.segmentation import clear_border # Optional: for removing border-touching elements
import os # For potential font file handling if needed (though using OpenCV fonts here)

# --- Constants ---
NUM_IMAGES_PER_DIGIT = 10
GRID_SIZE = 9
# Base size for generating the initial image (before perspective warp)
# Make it larger than final rectified size to avoid quality loss during warping
BASE_IMAGE_SIZE = 2000
# Target size for the rectified Sudoku grid (square) after perspective correction
RECTIFIED_GRID_SIZE = 900
CELL_SIZE = RECTIFIED_GRID_SIZE // GRID_SIZE # Size of one cell after rectification (100x100)
# Target size for the final processed cell images for the dataset
FINAL_CELL_SIZE = (28, 28)

# --- Helper Functions ---

def rectify(h):

    h = h.reshape((4, 2))
    hnew = np.zeros((4, 2), dtype=np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h, axis=1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew


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

    # Difference (y - x)
    # Note: np.diff calculates difference between consecutive elements.
    # We need y-x, so calculate explicitly or use difference from sums
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
    # 1. Choose colors
    v_line = random.choice(range(50))
    line_color = (v_line, v_line, v_line)
    v_text = random.choice(range(50))
    text_color = (v_text, v_text, v_text)
    v_background = 255 - random.choice(range(155))
    background_color = (v_background, v_background, v_background)

    # 2. Create Base Canvas
    image = np.full((BASE_IMAGE_SIZE, BASE_IMAGE_SIZE, 3), background_color, dtype=np.uint8)
    cell_draw_size = BASE_IMAGE_SIZE // GRID_SIZE

    # 3. Draw Grid Lines
    inside_thickness = random.randint(1,4)
    outside_thickness = max(random.randint(2,5), 1+inside_thickness)  #ensure outside thickness is always greater
    for i in range(GRID_SIZE + 1):
        thickness = inside_thickness
        if i % 3 == 0: # Thicker lines for 3x3 blocks
            thickness = outside_thickness
        if i == 0 or i == GRID_SIZE:  # double thickness for border because it'll be cut off
            thickness = outside_thickness * 2
        # Horizontal lines
        start_point_h = (0, i * cell_draw_size)
        end_point_h = (BASE_IMAGE_SIZE, i * cell_draw_size)
        cv2.line(image, start_point_h, end_point_h, line_color, thickness)
                # Vertical lines
        start_point_v = (i * cell_draw_size, 0)
        end_point_v = (i * cell_draw_size, BASE_IMAGE_SIZE)
        cv2.line(image, start_point_v, end_point_v, line_color, thickness)

    # 4. Draw digits
    font_paths = [
        "Fonts/arial.ttf",
        "Fonts/Times New Roman.ttf",
        "Fonts/Helvetica.ttf",
        "Fonts/Cambria.ttf",
        "Fonts/Comic Sans.ttf"
    ]

    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    digit_str = str(digit)

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            font_path = random.choice(font_paths)
            font_size = font_size = random.randint(80, 160)
            font = ImageFont.truetype(font_path, font_size)

            # Get text bounding box
            bbox = font.getbbox(digit_str)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            # Cell center
            cell_center_x = c * cell_draw_size + cell_draw_size // 2
            cell_center_y = r * cell_draw_size + cell_draw_size // 2
            # Adjust origin using bbox y-offset for better vertical centering
            origin_x = cell_center_x - text_width // 2
            origin_y = cell_center_y - (text_height // 2 + bbox[1])  # Compensate for top offset
            # Apply offset
            off = 10
            offset_x = random.randint(-cell_draw_size // off, cell_draw_size // off)
            offset_y = random.randint(-cell_draw_size // off, cell_draw_size // off)
            final_origin = (origin_x + offset_x, origin_y + offset_y)
            draw.text(final_origin, digit_str, font=font, fill=text_color)

    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)  # Convert back to OpenCV format

    # 5. Place the sudoku on a larger page
    padding = 200
    outer_size = BASE_IMAGE_SIZE + 2*padding
    padded_image = np.full((outer_size, outer_size, 3), background_color, dtype=np.uint8)
    padded_image[padding:padding + BASE_IMAGE_SIZE, padding:padding + BASE_IMAGE_SIZE] = image
    image=padded_image


    # 6. Add Noise
    mean = 0
    std_dev = random.uniform(10, 30) # Randomize noise level
    noise = np.random.normal(mean, std_dev, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255) # Ensure values are within valid range
    image = noisy_image.astype(np.uint8)

    # 5. Apply Perspective Warp
    h, w = image.shape[:2]

    # Define original corners
    original_corners = np.array([
        [0, 0],  # Top-left
        [w - 1, 0],  # Top-right
        [w - 1, h - 1],  # Bottom-right
        [0, h - 1]  # Bottom-left
    ], dtype="float32")

    # Define shifted corners with specific random movement constraints
    shifted_corners = np.array([
        [random.uniform(800, 1000), random.uniform(800, 1000)],  # Top-left
        [w - 1 - random.uniform(800, 1000), random.uniform(800, 1000)],  # Top-right
        [w - 1 - random.uniform(0, 200), h - 1 - random.uniform(0, 200)],  # Bottom-right
        [random.uniform(0, 200), h - 1 - random.uniform(0, 200)]  # Bottom-left
    ], dtype="float32")

    # Get perspective transform matrix and apply warp
    matrix = cv2.getPerspectiveTransform(original_corners, shifted_corners)
    warped_image = cv2.warpPerspective(image, matrix, (w, h))
    cropped_image = warped_image[750:h-200, 200:w-200]
    h, w = cropped_image.shape[:2]

    # 7. Apply Lens Blur
    strength = random.uniform(1,2)
    direction = random.choice(['top', 'bottom', 'both'])

    # Create a blurred copy of the image
    blurred = cv2.GaussianBlur(cropped_image, (0, 0), strength)

    # Create vertical gradient mask (1 = sharp, 0 = blurred)
    if direction == 'top':
        mask = np.linspace(1, 0, h).reshape(h, 1)
    elif direction == 'both':
        mask = np.abs(np.linspace(-1, 1, h)).reshape(h, 1)
        mask = 1 - mask  # center sharp, top/bottom blurred
    else:  # 'bottom'
        mask = np.linspace(0, 1, h).reshape(h, 1)

    # Expand mask to match image channels
    mask = np.repeat(mask, w, axis=1)
    mask = np.dstack([mask] * 3)

    # Blend original and blurred images
    image = (cropped_image * mask + blurred * (1 - mask)).astype(np.uint8)

    return image, shifted_corners # shifted_corners is already (4, 2) float32




# --- Image Processing Functions ---

def rectify_perspective(warped_image, warped_corners, output_size):
    """
    Rectifies the perspective of a warped Sudoku image using known corner points.

    Args:
        warped_image (np.ndarray): The input image with perspective distortion.
        warped_corners (np.ndarray): The (4, 2) array of corner points in the warped image
                                      that correspond to the original grid corners.
        output_size (int): The desired side length of the output square image (e.g., 900).

    Returns:
        np.ndarray: The perspective-corrected square image of the Sudoku grid.
    """
    # Define the destination points for the square image
    dst_pts = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1]
    ], dtype="float32")

    # Order the input warped corners: top-left, top-right, bottom-right, bottom-left
    ordered_warped_pts = order_points(warped_corners)

    # Calculate the perspective transformation matrix (from warped to rectified)
    matrix = cv2.getPerspectiveTransform(ordered_warped_pts, dst_pts)

    # Apply the perspective warp to rectify
    rectified = cv2.warpPerspective(warped_image, matrix, (output_size, output_size))

    return rectified

def extract_cells(rectified_grid_image):
    """
    Splits the rectified Sudoku grid image into 81 individual cell images.

    Args:
        rectified_grid_image (np.ndarray): The square, perspective-corrected grid image.

    Returns:
        list: A list containing 81 NumPy arrays, each representing a cell image.
    """
    cells = []
    current_cell_size = rectified_grid_image.shape[0] // GRID_SIZE # Should match CELL_SIZE constant
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
    """
    Preprocesses an individual cell image for OCR training dataset.
    Converts to grayscale, blurs, thresholds, applies morphology, clears border,
    and resizes to the target size.

    Args:
        cell_image (np.ndarray): Image of a single Sudoku cell (expects BGR).
        target_size (tuple): The final desired (width, height) for the image.

    Returns:
        np.ndarray: Preprocessed grayscale image resized to target_size.
    """
    # 1. Convert to Grayscale
    gray_cell = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)

    # 2. Apply Gaussian Blur
    # Use a slightly larger kernel before thresholding if needed
    blurred_cell = cv2.GaussianBlur(gray_cell, (5, 5), 0)

    # 3. Apply Adaptive Thresholding (inverse binary)
    thresh_cell = cv2.adaptiveThreshold(blurred_cell, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 3) # Adjust block size/C if needed

    # 4. Morphological Operations (Optional but requested)
    # Kernel for morphology - adjust size as needed
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # Example: Opening (remove small noise) - might remove thin parts of digits
    # processed_cell = cv2.morphologyEx(thresh_cell, cv2.MORPH_OPEN, kernel)
    # Example: Closing (close small gaps) - might connect noise
    # processed_cell = cv2.morphologyEx(thresh_cell, cv2.MORPH_CLOSE, kernel)
    # Often, just thresholding is enough, or a specific sequence like dilate then erode
    # Let's apply a slight dilation to make digits thicker, potentially followed by erosion
    dilated_cell = cv2.dilate(thresh_cell, kernel, iterations=1)
    # eroded_cell = cv2.erode(dilated_cell, kernel, iterations=1) # Optional erosion after dilation
    processed_cell = dilated_cell # Using dilated result

    # 5. Clear Border Artifacts
    try:
        # Add a small temporary border before clearing, then remove it,
        # to handle cases where the digit perfectly touches the original edge.
        pad_width = 3
        padded_cell = cv2.copyMakeBorder(processed_cell, pad_width, pad_width, pad_width, pad_width, cv2.BORDER_CONSTANT, value=0)
        cleared_padded = clear_border(padded_cell)
        # Crop back to original size (before padding)
        processed_cell = cleared_padded[pad_width:-pad_width, pad_width:-pad_width]

    except ImportError:
        print("[WARN] skimage not fully installed or import failed. Skipping clear_border.")
    except Exception as e:
         print(f"[WARN] Error during clear_border: {e}. Skipping.")

    # 6. Resize to Final Target Size (e.g., 28x28)
    # Use INTER_AREA for shrinking to potentially preserve features better
    resized_cell = cv2.resize(processed_cell, target_size, interpolation=cv2.INTER_AREA)

    return resized_cell

# --- Main Data Generation Pipeline ---
if __name__ == "__main__":
    print("[INFO] Starting synthetic Sudoku training data generation...")

    all_processed_cells = []
    all_labels = []

    # Loop through digits 1 to 9
    for digit in range(1, 10):
        print(f"[INFO] Generating images for digit: {digit}")
        # Generate N images for the current digit
        for i in range(NUM_IMAGES_PER_DIGIT):
            print(f"  Generating image {i+1}/{NUM_IMAGES_PER_DIGIT} for digit {digit}...")

            # 1. Generate Synthetic Sudoku Image + Warp
            warped_img, warped_corners = generate_sudoku_image(digit)

            # DEBUG: Show generated warped image
            # if i == 0: # Show only the first generated image per digit
            cv2.imshow(f"Warped Synthetic - Digit {digit}", warped_img)
            cv2.waitKey(0) # Show for a short time

            # 2. Rectify Perspective
            rectified_grid = rectify_perspective(warped_img, warped_corners, RECTIFIED_GRID_SIZE)

            # DEBUG: Show rectified grid
            # if i == 0:
            #    cv2.imshow(f"Rectified Grid - Digit {digit}", rectified_grid)
            #    cv2.waitKey(100)

            # 3. Extract Cells
            cells = extract_cells(rectified_grid)

            # 4. Preprocess Each Cell and Store
            if len(cells) != GRID_SIZE * GRID_SIZE:
                 print(f"[WARN] Expected {GRID_SIZE*GRID_SIZE} cells, but found {len(cells)}. Skipping this image.")
                 continue

            for cell_idx, cell in enumerate(cells):
                processed_cell = preprocess_cell(cell, target_size=FINAL_CELL_SIZE)

                # DEBUG: Show processed cell
                # if i == 0 and cell_idx < 5: # Show first few cells of first image
                #     cv2.imshow(f"Processed Cell ({digit}, {i}, {cell_idx})", processed_cell)
                #     cv2.waitKey(10)

                all_processed_cells.append(processed_cell)
                all_labels.append(digit)

    print("[INFO] Data generation complete.")
    cv2.destroyAllWindows() # Close any debug windows

    # 5. Convert lists to NumPy arrays
    X_train = np.array(all_processed_cells, dtype=np.uint8) # uint8 for grayscale images (0-255)
    y_train = np.array(all_labels, dtype=np.uint8)        # uint8 for labels 1-9

    # Verify the shapes
    expected_count = GRID_SIZE * GRID_SIZE * NUM_IMAGES_PER_DIGIT * 9 # 81 * 10 * 9 = 7290
    print(f"\n--- Generated Dataset Shapes ---")
    print(f"X_train (images) shape: {X_train.shape}")
    print(f"y_train (labels) shape: {y_train.shape}")
    print(f"Expected number of samples: {expected_count}")
    if X_train.shape[0] == expected_count and y_train.shape[0] == expected_count:
        print("[INFO] Dataset shapes match expected count.")
    else:
        print("[WARN] Dataset shapes DO NOT match expected count!")

    # The NumPy arrays X_train and y_train now hold your synthetic dataset.
    # You can now use them directly for training an ML model or save them:
    # np.savez_compressed('synthetic_sudoku_dataset.npz', X_train=X_train, y_train=y_train)
    # print("\n[INFO] Dataset saved to synthetic_sudoku_dataset.npz (optional)")