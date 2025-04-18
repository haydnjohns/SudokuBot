import numpy as np
import cv2

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    original = image.copy()
    # cv2.imshow("Original", original)
    #cv2.waitKey(0)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Grayscale", gray)
    #cv2.waitKey(0)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    # cv2.imshow("Blurred", blurred)
    # cv2.waitKey(0)

    # Adaptive Threshold to binarize
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    # cv2.imshow("Thresholded", thresh)
    #cv2.waitKey(0)

    # Dilation followed by erosion (closing)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=3)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    # cv2.imshow("Dilated and Eroded", eroded)
    # cv2.waitKey(0)

    return eroded, original

def find_sudoku_contour(processed_img, original_img):
    contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sudoku_contour = None
    max_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # Look for the largest 4-sided contour
        if area > max_area and len(approx) == 4:
            sudoku_contour = approx
            max_area = area

    # Draw the contour on a copy of the original image
    if sudoku_contour is not None:
        cv2.drawContours(original_img, [sudoku_contour], -1, (0, 255, 0), 3)
        cv2.imshow("Detected Sudoku Contour", original_img)
        # print("Sudoku contour points:", sudoku_contour.reshape(4, 2))
    else:
        print("No quadrilateral contour found.")

    return sudoku_contour

def pixel_to_world_distances(points_px):

    # --- Camera parameters ---
    focal_length = 3.744  # mm
    sensor_width = 3.6  # mm
    sensor_height = 2.7  # mm
    sensor_width_px = 2592  # pixels
    sensor_height_px = 1944  # pixels
    camera_height = 82.5  # mm
    camera_angle = 23.45  # degrees
    camera_angle_rad = np.radians(camera_angle)

    # --- Convert pixel coordinates to mm relative to sensor center ---
    points_mm_centre = np.zeros((4, 2))
    points_mm_centre[:, 0] = (points_px[:, 0] - 0.5 * sensor_width_px) * (sensor_width / sensor_width_px)
    points_mm_centre[:, 1] = (points_px[:, 1] - 0.5 * sensor_height_px) * (sensor_height / sensor_height_px)

    # --- Convert to vectors in camera coordinate system ---
    points_vectors = np.hstack((np.ones((4, 1)), points_mm_centre / focal_length))  # shape (4, 3)

    # --- Rotation matrix for camera tilt (rotation about x-axis) ---
    rotation_matrix = np.array([
        [np.cos(camera_angle_rad), 0, np.sin(camera_angle_rad)],
        [0, 1, 0],
        [-np.sin(camera_angle_rad), 0, np.cos(camera_angle_rad)]
    ])

    # --- Rotate vectors to world coordinate system ---
    points_global = (rotation_matrix @ points_vectors.T).T  # shape (4, 3)

    # --- Scale vectors so z = 0 (ground level) ---
    k_values = camera_height / points_global[:, 2]

    # --- Compute real-world (X, Y) ground plane coordinates ---
    ground_coords = np.zeros((4, 2))  # [X, Y]
    ground_coords[:, 0] = k_values * points_global[:, 1]  # horizontal (left-right)
    ground_coords[:, 1] = k_values * points_global[:, 0]  # forward (depth)

    return ground_coords


image_path = "/Users/haydnjohns/photo4.jpg"

# PREPROCESS
processed, original = preprocess_image(image_path)

# FIND PIXEL COORDINATES
image_height = 1944
pixel_points = find_sudoku_contour(processed, original)
pixel_points = pixel_points.reshape(4, 2).astype(np.float32)
pixel_points[:, 1] = image_height - pixel_points[:, 1]
print(f"\nPixel coordinates:\n {pixel_points}")

# FIND GROUND COORDINATES
ground_coords = pixel_to_world_distances(pixel_points)

# DISPLAY RESULTS
print("\nReal-world ground coordinates (mm):")
for i, (x, y) in enumerate(ground_coords):
    print(f"Point {i+1}: Horizontal = {x:.2f} mm, Depth = {y:.2f} mm")