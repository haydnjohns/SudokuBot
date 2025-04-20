import cv2
import numpy as np
import random
from PIL import ImageFont, ImageDraw, Image
from skimage.segmentation import clear_border
import os

# --- Constants ---
NUM_IMAGES_PER_DIGIT = 100
GRID_SIZE = 9
BASE_IMAGE_SIZE = 2000
RECTIFIED_GRID_SIZE = 900
CELL_SIZE = RECTIFIED_GRID_SIZE // GRID_SIZE
FINAL_CELL_SIZE = (28, 28)

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

def process_board(image, target_size=RECTIFIED_GRID_SIZE):
    # Step 1: Preprocess the image
    greyed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = cv2.fastNlMeansDenoising(image, None, h=20)
    image = cv2.GaussianBlur(greyed , (3, 3), 0)
    image = cv2.adaptiveThreshold(image, 255, 1, 1, 151, 3.2)
    image = cv2.bitwise_not(image)
    kernel = np.ones((2, 2), np.uint8)
    # image = cv2.dilate(image, kernel, iterations=0)
    image = cv2.erode(image, kernel, iterations=2)

    # Step 2: Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Step 3: Visualize all contours
    contour_visualization = image.copy()
    cv2.drawContours(contour_visualization, contours, -1, (0, 255, 0), 2)


    # Step 4: Filter to find the largest suitable 4-point contour (not near edges)
    biggest = None
    max_area = 0
    margin = 10  # margin from the edge of the image
    height, width = image.shape[:2]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 10000:
            x, y, w, h = cv2.boundingRect(cnt)
            if x <= margin or y <= margin or x + w >= width - margin or y + h >= height - margin:
                continue  # Skip if bounding box is too close to the image edge

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4 and area > max_area:
                biggest = approx
                max_area = area

    # Step 5: If a valid board contour was found, warp it
    if biggest is not None:
        sorted_pts = rectify(biggest)
        target = np.array([
            [0, 0],
            [target_size, 0],
            [target_size, target_size],
            [0, target_size]
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(sorted_pts, target)
        warped = cv2.warpPerspective(greyed, matrix, (target_size, target_size))
        return warped

    # Step 6: If no valid contour found, return None
    print("No suitable board contour found.")
    return None

def extract_digit(i, j, image, target_size=RECTIFIED_GRID_SIZE):
    cell_size = target_size // 9
    digit = image[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]
    thresh_digit = cv2.adaptiveThreshold(digit, 255, 1, 1, 99, 7)

    kernel = np.ones((2, 2), np.uint8)
    eroded = cv2.erode(thresh_digit, kernel, iterations=2)
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    cleared = clear_border(dilated)

    kernel = np.ones((15, 15), np.uint8)
    closed = cv2.morphologyEx(cleared, cv2.MORPH_CLOSE, kernel)


    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    cropped = thresh_digit[y:y + h, x:x + w]
    canvas = np.zeros((100, 100), dtype=np.uint8)
    cx, cy = (100 - w) // 2, (100 - h) // 2
    canvas[cy:cy + h, cx:cx + w] = cropped
    resized = cv2.resize(canvas, FINAL_CELL_SIZE, interpolation=cv2.INTER_AREA)

    # cv2.imshow('resized', resized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Skip mostly empty images
    if np.sum(resized > 30) < 20:  # A rough threshold for "empty"
        return None
    return resized

def save_digits_to_array(image, digit_label):
    digits = []
    labels = []
    for i in range(9):
        for j in range(9):
            d = extract_digit(i, j, image)
            if d is not None:
                digits.append(d)
                labels.append(digit_label)
    return digits, labels

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
    inside_thickness = random.randint(2,8)
    outside_thickness = max(random.randint(4,10), 2+inside_thickness)  #ensure outside thickness is always greater
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
            font_size = font_size = random.randint(100, 160)
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

    return image

# --- Main Data Generation Pipeline ---
if __name__ == "__main__":
    print("[INFO] Starting synthetic Sudoku training data generation...")

    all_processed_cells = []
    all_labels = []

    for digit in range(1, 10):  # Looping through digits 1-9
        print(f"[INFO] Generating for digit {digit}")
        for i in range(NUM_IMAGES_PER_DIGIT):
            print(f"  Generating image {i + 1}/{NUM_IMAGES_PER_DIGIT}")

            synthetic_img = generate_sudoku_image(digit)
            # cv2.imshow('synthetic_img', synthetic_img)
            # cv2.waitKey(0)
            rectified_img = process_board(synthetic_img)
            # cv2.imshow('rectified_img', rectified_img)
            # cv2.waitKey(0)

            if rectified_img is None:
                print("[WARN] Rectification failed. Skipping.")
                continue

            digits, labels = save_digits_to_array(rectified_img, digit)
            all_processed_cells.extend(digits)
            all_labels.extend(labels)

    x_train = np.array(all_processed_cells, dtype=np.uint8)
    y_train = np.array(all_labels, dtype=np.uint8)

    print(f"\n--- Dataset Summary ---")
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"Total samples: {x_train.shape[0]}")
    print(f"Digit distribution: {np.bincount(y_train)[1:]}")  # [0] is unused

    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    # print(X_train)

    # Optional: Save
    np.savez_compressed('sudoku_digits.npz', x_train=x_train, y_train=y_train)