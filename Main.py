import numpy as np
import cv2
import math
from time import sleep
from scipy.optimize import least_squares
from gpiozero import OutputDevice
from skimage.segmentation import clear_border
import threading  # Allows motors to run simultaneously
from ai_edge_litert.interpreter import Interpreter
import subprocess

def capture_image():
    """
    Captures a photo from the Raspberry Pi camera using libcamera-jpeg
    and returns it as a NumPy array (OpenCV image).
    """
    # Run libcamera-jpeg and capture JPEG data from stdout
    result = subprocess.run(
        ['libcamera-jpeg', '-o', '-', '--width', '2592', '--height', '1944', '--nopreview'],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL  # You can change to subprocess.PIPE if you want logs
    )

    # Convert binary output to NumPy array
    image_bytes = np.frombuffer(result.stdout, dtype=np.uint8)

    # Decode JPEG image to OpenCV format
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError("Failed to capture or decode image from camera")

    return image

def preprocess_image(image, debug=False):
    image = cv2.imread(image)
    # image = cv2.rotate(image, cv2.ROTATE_180)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    original = image.copy()
    gray=gray.copy()
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=3)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    if debug:
        cv2.imshow('Original', original)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow('Eroded', eroded)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return eroded, original, gray

def find_sudoku_contour(processed_img, original_img):
    contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    sudoku_contour = None

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if area > max_area and len(approx) == 4:
            sudoku_contour = approx
            max_area = area

    if sudoku_contour is not None:
        cv2.drawContours(original_img, [sudoku_contour], -1, (0, 255, 0), 3)
        # cv2.imshow("Detected Sudoku Contour", original_img)
        # cv2.waitKey(1)
        # cv2.destroyAllWindows()
    return sudoku_contour

def sort_corners(contour):
    contour = contour.reshape((4, 2))

    # sort by y (top to bottom)
    contour_sorted = contour[np.argsort(contour[:, 1]), :]

    # now take top-most two and bottom-most two
    top_two = contour_sorted[:2, :]
    bottom_two = contour_sorted[2:, :]

    # sort left to right
    top_left, top_right = top_two[np.argsort(top_two[:, 0]), :]
    bottom_left, bottom_right = bottom_two[np.argsort(bottom_two[:, 0]), :]
    contours_sorted = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

    return  contours_sorted

def pixel_to_world_distances(contour_sorted, debug=False):
    # --- Camera and image parameters ---
    stamper_setback = 16  # mm (offset in robot's forward direction)
    camera_height = 85    # mm (height from lens center to ground)
    camera_angle = 26.35  # degrees (tilt down from horizontal)
    camera_angle_rad = -np.radians(camera_angle)

    # --- Intrinsic camera matrix and distortion coefficients ---
    camera_matrix = np.array([
        [2650.9964176101485, 0.0, 1638.4541745042789],
        [0.0, 2644.434621507731, 1106.6683119862594],
        [0.0, 0.0, 1.0]],
        dtype=np.float32
    )

    dist_coeffs = np.array([
        0.19930490026457645,
        -0.2903695599399088,
        -0.007406225233953261,
        0.005002318775684634,
        -0.5143858238250185],
        dtype=np.float32
    )

    N = len(contour_sorted)

    # --- Undistort and reproject to pixel coordinates ---
    undistorted = cv2.undistortPoints(
        contour_sorted.reshape(-1, 1, 2),
        camera_matrix,
        dist_coeffs,
        P=camera_matrix  # Reproject to pixel space
    ).reshape(-1, 2)

    if debug:
        print(f"\n[Step 0] Undistorted pixel coordinates:\n{undistorted}")

    # --- Convert pixel coordinates to normalized camera coordinates ---
    homog_points = cv2.convertPointsToHomogeneous(undistorted).reshape(-1, 3)  # (x, y, 1)
    points_camera = (np.linalg.inv(camera_matrix) @ homog_points.T).T  # shape (N, 3)

    if debug:
        print(f"\n[Step 1] Rays in camera coordinates:\n{points_camera}")

    # --- Rotate camera rays to world frame (accounting for tilt down) ---
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(camera_angle_rad), -np.sin(camera_angle_rad)],
        [0, np.sin(camera_angle_rad),  np.cos(camera_angle_rad)]
    ])
    points_global = (rotation_matrix @ points_camera.T).T

    if debug:
        print(f"\n[Step 2] Rays in global coordinates (tilted down):\n{points_global}")

    # --- Project rays onto ground plane (Y = -camera_height) ---
    k_values = -camera_height / points_global[:, 1]  # scale factors

    ground_coords = np.zeros((N, 2))
    ground_coords[:, 0] = - k_values * points_global[:, 0]                # X axis (left-right)
    ground_coords[:, 1] = - k_values * points_global[:, 2] + stamper_setback  # Y axis (forward-back)

    if debug:
        print(f"\n[Step 3] Real-world coordinates of points on ground (mm):\n{ground_coords}")

    return ground_coords

def pixel_to_world_distances2(contour_sorted, debug=False):
    # --- Camera and image parameters ---
    stamper_setback = 16  # mm
    focal_length = 3.04  # mm
    sensor_width = 3.68    # mm
    sensor_height = 2.76   # mm
    sensor_width_px = 3280
    sensor_height_px = 2464
    camera_height = 85  # mm
    camera_angle = 26.51  # degrees (from vertical?)
    camera_angle_rad = np.radians(camera_angle)

    # --- Intrinsic camera matrix and distortion coefficients ---
    camera_matrix = np.array([
        [2650.9964176101485, 0.0, 1638.4541745042789],
        [0.0, 2644.434621507731, 1106.6683119862594],
        [0.0, 0.0, 1.0]],
        dtype=np.float32
    )

    dist_coeffs = np.array([
        0.19930490026457645,
        -0.2903695599399088,
        -0.007406225233953261,
        0.005002318775684634,
        -0.5143858238250185],
        dtype=np.float32
    )

    # --- Undistort the input pixel points ---
    undistorted = cv2.undistortPoints(
        contour_sorted.reshape(-1, 1, 2),
        camera_matrix,
        dist_coeffs,
        P=camera_matrix  # Output in pixel coordinates
    ).reshape(-1, 2)

    # undistorted = contour_sorted
    if debug:
        print(f"\n[Step 0] Undistorted pixel coordinates:\n{undistorted}")

    # --- Convert pixel coordinates to mm relative to the center of the sensor ---
    points_mm_centre = np.zeros((4, 2))
    points_mm_centre[:, 0] = ((undistorted[:, 0] - 0.5 * sensor_width_px) *
                              (sensor_width / sensor_width_px))  # X axis
    points_mm_centre[:, 1] = ((sensor_height_px - undistorted[:, 1] - 0.5 * sensor_height_px) *
                              (sensor_height / sensor_height_px))  # Y axis, accounting for fact that y pixels are measured from the top of the image

    if debug:
        print(f"\n[Step 1] Pixel coordinates relative to sensor centre (mm):\n{points_mm_centre}")

    # --- Construct 3D rays in camera coordinates ---
    points_camera = np.zeros((4, 3))
    points_camera[:, 0] = points_mm_centre[:, 0]  # X
    points_camera[:, 1] = points_mm_centre[:, 1]  # Y
    points_camera[:, 2] = focal_length            # Z (depth from lens center)

    if debug:
        print(f"\n[Step 2] Rays in camera coordinates (mm):\n{points_camera}")

    # --- Rotate rays to world coordinates (camera tilt) ---
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(camera_angle_rad), -np.sin(camera_angle_rad)],
        [0, np.sin(camera_angle_rad),  np.cos(camera_angle_rad)]
    ])
    points_global = (rotation_matrix @ points_camera.T).T  # shape (4, 3)

    if debug:
        print(f"\n[Step 3] Rays in global coordinates (mm):\n{points_global}")

    # --- Project onto ground plane (scale vectors so Z=0) ---
    k_values = -camera_height / points_global[:, 1]  # Scaling factors to flatten to ground

    if debug:
        print(f"\n[Step 4] Scaling factors to project rays to ground plane:\n{k_values}")

    ground_coords = np.zeros((4, 2))
    ground_coords[:, 0] = k_values * points_global[:, 0]  # X (left-right)
    ground_coords[:, 1] = k_values * points_global[:, 2] + stamper_setback  # Y (depth)

    if debug:
        print(f"\n[Step 5] Real-world coordinates of square corners (mm):\n{ground_coords}")

    return ground_coords

def fit_rectangle_least_squares(points):
    """
    Fit a rectangle to 4 points by minimizing the squared distance
    to the rectangle corners, enforcing right angles.

    Args:
        points (np.ndarray): shape (4,2) input points.

    Returns:
        np.ndarray: shape (4,2) fitted rectangle corners.
    """
    points = np.asarray(points)

    # Initial guess:
    # Use first point as P0
    P0 = points[0]
    # v: vector from P0 to P1
    v = points[1] - P0
    # w: vector from P0 to P3
    w = points[3] - P0

    # Parameter vector: [P0_x, P0_y, v_x, v_y, w_x, w_y]
    x0 = np.hstack((P0, v, w))

    def residuals(params):
        P0 = params[0:2]
        v = params[2:4]
        w = params[4:6]

        # Enforce orthogonality by penalizing dot product
        orth_penalty = np.dot(v, w)

        # Construct rectangle corners
        P = np.array([
            P0,
            P0 + v,
            P0 + v + w,
            P0 + w
        ])

        # Compute distances from input points to these corners
        dists = P - points
        res = dists.flatten()

        # Add orthogonality penalty weighted (e.g., times 100)
        res = np.hstack((res, orth_penalty * 100))

        return res

    result = least_squares(residuals, x0)

    P0 = result.x[0:2]
    v = result.x[2:4]
    w = result.x[4:6]

    fitted_corners = np.array([
        P0,
        P0 + v,
        P0 + v + w,
        P0 + w
    ])

    # 👇 Compute final error explicitly here
    orth_penalty = np.dot(v, w)
    residual_vec = (fitted_corners - points).flatten()
    final_error = np.sum(residual_vec**2) + (orth_penalty * 100)**2

    return fitted_corners, final_error

def generate_real_world_coordinates(world_distances):

    # Step 1: Calculate average grid width and height
    top_left, top_right, bottom_right, bottom_left = world_distances

    top_width = np.linalg.norm(top_left - top_right)
    bottom_width = np.linalg.norm(bottom_left - bottom_right)
    average_grid_width = (top_width + bottom_width) / 2

    left_height = np.linalg.norm(top_left - bottom_left)
    print(f"left_height: {left_height}")
    right_height = np.linalg.norm(top_right - bottom_right)
    print(f"right_height: {right_height}")
    average_grid_height = (left_height + right_height) / 2

    # Step 2: Assume square grid based on average size
    # average_grid_size = (average_grid_width + average_grid_height) / 2

    destination_points = np.array([
        [0, average_grid_height],  # Top-left
        [average_grid_width, average_grid_height],  # Top-right
        [average_grid_width, 0],  # Bottom-right
        [0, 0]  # Bottom-left
    ])

    # Step 3: Compute transformation
    source_points = np.array(world_distances)
    destination_points = np.array(destination_points)

    source_centroid = np.mean(source_points, axis=0)
    destination_centroid = np.mean(destination_points, axis=0)

    source_centered = source_points - source_centroid
    destination_centered = destination_points - destination_centroid

    covariance_matrix = source_centered.T @ destination_centered
    u_matrix, singular_values, v_transpose = np.linalg.svd(covariance_matrix)

    rotation_matrix = v_transpose.T @ u_matrix.T

    # Handle reflection
    if np.linalg.det(rotation_matrix) < 0:
        v_transpose[1, :] *= -1
        rotation_matrix = v_transpose.T @ u_matrix.T

    translation_vector = destination_centroid - rotation_matrix @ source_centroid

    # Step 4: Calculate robot's new position
    robot_camera_original_position = np.array([0, 0])  # In robot's own frame
    robot_camera_position_new = rotation_matrix @ robot_camera_original_position + translation_vector
    robot_stamp_position_new = robot_camera_position_new.copy()
    robot_stamp_position_new[1] -= 9  # Subtract 9mm from y (forward/backward) only

    # Step 5: Calculate robot heading
    # Extract rotation angle from rotation matrix
    heading_radians = - (np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
    robot_heading_deg = np.degrees(heading_radians)

    # Step 6: Calculate cell size
    horizontal_cell_size = average_grid_width / 9
    vertical_cell_size = average_grid_height / 9



    return robot_heading_deg, robot_stamp_position_new, horizontal_cell_size, vertical_cell_size

def warp_image(image, contour_sorted, target_size):

    # Define target corners for output image
    destination_corners = np.array([
        [0, 0],
        [target_size - 1, 0],
        [target_size - 1, target_size - 1],
        [0, target_size - 1]
    ], dtype="float32")

    transformation_matrix = cv2.getPerspectiveTransform(contour_sorted, destination_corners)
    warped_image = cv2.warpPerspective(image, transformation_matrix, (target_size, target_size))

    return warped_image

def extract_digit(i, j, warped_image, target_size=900):
    cell_size_actual = target_size // 9
    cell_size_buffered = target_size // 8  # ~12.5% larger for buffer

    # Center of the cell
    center_x = j * cell_size_actual + cell_size_actual // 2
    center_y = i * cell_size_actual + cell_size_actual // 2

    # Calculate crop bounds
    half_size = cell_size_buffered // 2
    x1 = max(center_x - half_size, 0)
    y1 = max(center_y - half_size, 0)
    x2 = min(center_x + half_size, warped_image.shape[1])
    y2 = min(center_y + half_size, warped_image.shape[0])

    digit = warped_image[y1:y2, x1:x2]
    # cv2.imshow('digit', digit)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    thresh_digit = cv2.adaptiveThreshold(digit, 255, 1, 1, 99, 7)
    # cv2.imshow('thresh', thresh_digit)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    kernel = np.ones((2, 2), np.uint8)
    eroded = cv2.erode(thresh_digit, kernel, iterations=0)
    dilated = cv2.dilate(eroded, kernel, iterations=0)
    cleared = clear_border(dilated)
    # cv2.imshow('cleared', cleared)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # kernel = np.ones((15, 15), np.uint8)
    # closed = cv2.morphologyEx(cleared, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('closed', closed)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    contours, _ = cv2.findContours(cleared, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    cropped = cleared[y:y + h, x:x + w]
    canvas = np.zeros((100, 100), dtype=np.uint8)
    cx, cy = (100 - w) // 2, (100 - h) // 2
    canvas[cy:cy + h, cx:cx + w] = cropped
    resized = cv2.resize(canvas, (28, 28), interpolation=cv2.INTER_AREA)
    # cv2.imshow('resized', resized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if np.sum(resized > 30) < 20:
        return None
    return resized

def predict_digit(cell_img, interpreter):
    # Preprocess input
    cell_img = cell_img.reshape(1, 28, 28, 1).astype("float32") / 255.0

    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], cell_img)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Return predicted digit (add 1 to match label system)
    return np.argmax(output_data) + 1

def build_virtual_sudoku(warped_image):
    virtual_board = []
    instructions = []

    for i in range(9):
        row = []
        for j in range(9):
            digit_img = extract_digit(i, j, warped_image)
            if digit_img is not None:
                # digit = predict_digit(digit_img)
                digit = predict_digit(digit_img, interpreter)
                row.append(digit)
                instructions.append(f"{i+1},{j+1},{digit}")
            else:
                row.append(0)
        virtual_board.append(row)

    virtual_board = np.array(virtual_board, dtype=np.int32)  # << Convert to real numpy array here

    return virtual_board, instructions

def print_virtual_board(board):

    # Print the top border first
    print("\n+ - - - + - - - + - - - +")

    for i in range(9):
        # Print horizontal separator for every 3rd row (after 3 rows)
        if i % 3 == 0 and i != 0:
            print("+ - - - + - - - + - - - +")

        # Initialize an empty row string
        row = ""

        for j in range(9):
            # Check if it's the first cell, if so, don't add the leading "|"
            if j % 3 == 0 and j != 0:
                row += "| "

            # Add the cell value to the row string
            if board[i][j] == 0:
                row += ". "
            else:
                row += f"{board[i][j]} "

        # Ensure there is a space between the last digit and the vertical line
        print(f"| {row.strip()} |")  # Added space before the last vertical bar

    # Print the bottom border after the last row
    print("+ - - - + - - - + - - - +")

def solve_sudoku(grid):
    """
    Solves a 9x9 Sudoku puzzle represented by a NumPy ndarray using Dancing Links (Algorithm X).
    Corrected version addressing IndexError.

    Args:
        grid: A 9x9 NumPy ndarray representing the Sudoku puzzle,
              with 0 for empty cells. dtype should be integer-compatible.

    Returns:
        A 9x9 NumPy ndarray representing the solved Sudoku grid (dtype=int),
        or None if no solution exists or the input is invalid.
    """
    N = 9
    N_SQ = N * N
    NUM_CONSTRAINTS = N_SQ * 4 # Cell(81), Row(81), Col(81), Box(81)
    ROOT_IDX = 0 # CORRECTED: Root node is the first node added (index 0)

    # --- Node Structure Indices ---
    # Each node is represented as a list: [L, R, U, D, CH, RID]
    L, R, U, D, CH, RID = 0, 1, 2, 3, 4, 5 # Indices for node attributes

    # --- Input Validation ---
    if not isinstance(grid, np.ndarray) or grid.shape != (N, N):
        raise ValueError(f"Input grid must be a {N}x{N} NumPy ndarray.")
    if not np.issubdtype(grid.dtype, np.integer):
         print(f"Warning: Grid dtype is {grid.dtype}, converting to int.", file=sys.stderr)
         grid = grid.astype(int)
    if np.any((grid < 0) | (grid > N)):
        print("Error: Grid contains values outside the range 0-9.", file=sys.stderr)
        return None

    # --- DLX Data Structure Initialization ---
    nodes = []
    col_sizes = [0] * NUM_CONSTRAINTS
    header_node_indices = [-1] * NUM_CONSTRAINTS
    row_starts = {}

    # --- Build DLX Matrix ---

    # 1. Create Root Node (at index ROOT_IDX = 0)
    # Root node points to itself initially. CH points to self, RID is 0.
    nodes.append([ROOT_IDX, ROOT_IDX, ROOT_IDX, ROOT_IDX, ROOT_IDX, 0])

    # 2. Create Column Header Nodes
    for j in range(NUM_CONSTRAINTS):
        # Header node links horizontally between root and other headers.
        # Vertically (U/D) points to itself initially. CH points to self.
        # RID is negative column index (-1 to -324) for identification.
        last_header_idx = nodes[ROOT_IDX][L] # Get index of node currently left of root
        h_idx = len(nodes)                   # Index for the new header node
        header_node_indices[j] = h_idx
        # Add node: [L=last_header, R=root, U=self, D=self, CH=self, RID=-(j+1)]
        nodes.append([last_header_idx, ROOT_IDX, h_idx, h_idx, h_idx, -(j + 1)])
        # Update horizontal links: Root <-> New Header <-> Last Header
        nodes[last_header_idx][R] = h_idx    # Previous node's R points to new header
        nodes[ROOT_IDX][L] = h_idx           # Root's L points to new header

    # 3. Create Choice Nodes (representing placing digit d in cell r,c)
    for r in range(N):
        for c in range(N):
            box_idx = (r // 3) * 3 + (c // 3)
            for digit in range(1, N + 1):
                d_idx = digit - 1 # 0-8 index for digit
                row_id = (r, c, digit)

                # Calculate the 4 constraint column indices this choice satisfies
                cell_col = r * N + c
                row_col = N_SQ + r * N + d_idx
                col_col = N_SQ * 2 + c * N + d_idx
                box_col = N_SQ * 3 + box_idx * N + d_idx
                col_indices = [cell_col, row_col, col_col, box_col]

                first_node_in_row_idx = -1
                prev_node_in_row_idx = -1

                # Create a node for each of the 4 constraints satisfied
                for col_idx in col_indices:
                    header_idx = header_node_indices[col_idx]
                    up_node_idx = nodes[header_idx][U] # Node currently above header

                    # Create new node
                    new_node_idx = len(nodes)
                    # Add node: [L=?, R=?, U=up_node, D=header, CH=header, RID=row_id]
                    nodes.append([-1, -1, up_node_idx, header_idx, header_idx, row_id])

                    # Link vertically within the column
                    nodes[up_node_idx][D] = new_node_idx # Old Up's Down points to New
                    nodes[header_idx][U] = new_node_idx  # Header's Up points to New
                    col_sizes[col_idx] += 1              # Increment column size

                    # Link horizontally within the row (circularly)
                    if first_node_in_row_idx == -1:
                        first_node_in_row_idx = new_node_idx
                        nodes[new_node_idx][L] = new_node_idx # Points to self initially
                        nodes[new_node_idx][R] = new_node_idx
                    else:
                        # Link New <-> Prev and New <-> First
                        nodes[new_node_idx][L] = prev_node_in_row_idx
                        nodes[new_node_idx][R] = first_node_in_row_idx
                        nodes[prev_node_in_row_idx][R] = new_node_idx
                        nodes[first_node_in_row_idx][L] = new_node_idx

                    prev_node_in_row_idx = new_node_idx

                # Store the index of one node from this row for easy lookup later
                if first_node_in_row_idx != -1:
                    row_starts[row_id] = first_node_in_row_idx

    # --- Cover/Uncover Operations ---
    def cover(header_idx):
        """Removes a column (header) and all rows intersecting it."""
        # Remove header from horizontal list
        header_l_idx = nodes[header_idx][L]
        header_r_idx = nodes[header_idx][R]
        nodes[header_l_idx][R] = header_r_idx
        nodes[header_r_idx][L] = header_l_idx

        # Iterate down the column, removing rows associated with this column
        row_node_idx = nodes[header_idx][D] # Start from node below header
        while row_node_idx != header_idx:
            # Iterate right through the row (nodes in other columns)
            node_in_row_idx = nodes[row_node_idx][R]
            while node_in_row_idx != row_node_idx:
                # Remove node from its column's vertical list
                node_u_idx, node_d_idx = nodes[node_in_row_idx][U], nodes[node_in_row_idx][D]
                node_c_hdr_idx = nodes[node_in_row_idx][CH] # Header index of the *other* column
                col_idx_to_decrement = -nodes[node_c_hdr_idx][RID] - 1 # Get col index (0-323)

                nodes[node_u_idx][D] = node_d_idx
                nodes[node_d_idx][U] = node_u_idx
                # Decrement size of the column this node belongs to
                if 0 <= col_idx_to_decrement < NUM_CONSTRAINTS:
                     col_sizes[col_idx_to_decrement] -= 1

                node_in_row_idx = nodes[node_in_row_idx][R] # Move right
            row_node_idx = nodes[row_node_idx][D] # Move down the original column

    def uncover(header_idx):
        """Restores a column (header) and all rows intersecting it."""
        # Iterate up the column (reverse order of cover), restoring rows
        row_node_idx = nodes[header_idx][U] # Start from node above header
        while row_node_idx != header_idx:
            # Iterate left through the row
            node_in_row_idx = nodes[row_node_idx][L]
            while node_in_row_idx != row_node_idx:
                 # Restore node to its column's vertical list
                node_u_idx, node_d_idx = nodes[node_in_row_idx][U], nodes[node_in_row_idx][D]
                node_c_hdr_idx = nodes[node_in_row_idx][CH]
                col_idx_to_increment = -nodes[node_c_hdr_idx][RID] - 1

                # Increment size of the column this node belongs to
                if 0 <= col_idx_to_increment < NUM_CONSTRAINTS:
                    col_sizes[col_idx_to_increment] += 1
                # Relink vertically
                nodes[node_u_idx][D] = node_in_row_idx
                nodes[node_d_idx][U] = node_in_row_idx

                node_in_row_idx = nodes[node_in_row_idx][L] # Move left
            row_node_idx = nodes[row_node_idx][U] # Move up the original column

        # Restore header to horizontal list
        header_l_idx = nodes[header_idx][L]
        header_r_idx = nodes[header_idx][R]
        nodes[header_l_idx][R] = header_idx
        nodes[header_r_idx][L] = header_idx

    # --- Handle Initial Grid Values ---
    # 1. Basic Sudoku Rule Check on Initial Grid
    processed_cells = set()
    processed_row_digits = set()
    processed_col_digits = set()
    processed_box_digits = set()
    initial_choice_nodes = [] # Store node indices representing initial choices

    for r in range(N):
        for c in range(N):
            digit = int(grid[r, c])
            if digit != 0:
                box_idx = (r // 3) * 3 + (c // 3)
                # Check for immediate conflicts
                if ( (r, c) in processed_cells or
                     (r, digit) in processed_row_digits or
                     (c, digit) in processed_col_digits or
                     (box_idx, digit) in processed_box_digits ):
                    print(f"Error: Initial grid violates Sudoku rules at ({r},{c}) value {digit}.", file=sys.stderr)
                    return None
                processed_cells.add((r,c))
                processed_row_digits.add((r, digit))
                processed_col_digits.add((c, digit))
                processed_box_digits.add((box_idx, digit))

                # Find the DLX row node corresponding to this initial value
                row_id = (r, c, digit)
                if row_id in row_starts:
                    initial_choice_nodes.append(row_starts[row_id])
                else:
                    print(f"Internal Error: Could not find DLX row for initial value {digit} at ({r},{c}).", file=sys.stderr)
                    return None

    # 2. Cover columns corresponding to initial values (effectively selecting these rows)
    try:
        for row_node_idx in initial_choice_nodes:
            # Cover the column header for the first node found for this choice
            cover(nodes[row_node_idx][CH])
            # Cover the column headers for all other nodes in the same row
            node_in_row_idx = nodes[row_node_idx][R]
            while node_in_row_idx != row_node_idx:
                 cover(nodes[node_in_row_idx][CH])
                 node_in_row_idx = nodes[node_in_row_idx][R]
    except Exception as e:
        print(f"Error during initial grid processing cover operation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None

    # --- Search Algorithm (Recursive Backtracking) ---
    solution_found = False
    final_solution_node_indices = [] # Stores indices of nodes in the final solution rows
    current_solution_node_indices = [] # Stores indices during recursive search

    def search():
        nonlocal solution_found, final_solution_node_indices
        if solution_found: return True # Stop searching once a solution is found

        # Base case: Header list is empty (root's right pointer points back to root)
        if nodes[ROOT_IDX][R] == ROOT_IDX:
            solution_found = True
            # Combine initial choices with recursively found choices
            final_solution_node_indices = initial_choice_nodes + list(current_solution_node_indices)
            return True

        # Choose column C with minimum size (S-heuristic)
        chosen_header_idx = -1
        min_size = float('inf')
        current_header_idx = nodes[ROOT_IDX][R] # Start from root's right
        while current_header_idx != ROOT_IDX:   # Iterate until back to root
            col_idx = -nodes[current_header_idx][RID] - 1 # Get col index (0-323)
            size = col_sizes[col_idx]
            if size < min_size:
                min_size = size
                chosen_header_idx = current_header_idx
            if min_size <= 1: break # Optimization: Cannot get smaller
            current_header_idx = nodes[current_header_idx][R] # Move to next header

        # If min_size is 0, this path is unsatisfiable -> backtrack
        if min_size == 0:
             return False
        # Should not happen if root.R != root, but as safeguard:
        if chosen_header_idx == -1:
             print("Error: No column chosen, but matrix not empty.", file=sys.stderr)
             return False

        # Cover the chosen column
        cover(chosen_header_idx)

        # Iterate through rows R intersecting the chosen column C
        row_node_idx = nodes[chosen_header_idx][D] # Down from header
        while row_node_idx != chosen_header_idx:
            # Tentatively add this row to the partial solution
            current_solution_node_indices.append(row_node_idx)

            # Cover columns J linked by other nodes in row R
            node_in_row_idx = nodes[row_node_idx][R] # Right from current node
            while node_in_row_idx != row_node_idx:
                cover(nodes[node_in_row_idx][CH]) # Cover the header of the column node_in_row is in
                node_in_row_idx = nodes[node_in_row_idx][R]

            # Recurse
            if search():
                return True # Solution found, propagate success

            # Backtrack: Remove row R and uncover columns J
            current_solution_node_indices.pop() # Remove R from partial solution
            # Uncover columns J in reverse order of covering
            node_in_row_idx = nodes[row_node_idx][L] # Left from current node
            while node_in_row_idx != row_node_idx:
                uncover(nodes[node_in_row_idx][CH])
                node_in_row_idx = nodes[node_in_row_idx][L]

            row_node_idx = nodes[row_node_idx][D] # Move down to next row in column C

        # If no row in C led to a solution, backtrack further
        uncover(chosen_header_idx)
        return False

    # --- Initiate Search and Reconstruct Solution ---
    if search():
        # Reconstruct the grid from the solution node indices
        solved_grid = np.zeros((N, N), dtype=int)
        for node_idx in final_solution_node_indices:
            row_id = nodes[node_idx][RID]
            # Ensure it's a choice node (RID is tuple)
            if isinstance(row_id, tuple) and len(row_id) == 3:
                r, c, digit = row_id
                if 0 <= r < N and 0 <= c < N and 1 <= digit <= N:
                    if solved_grid[r, c] == 0:
                        solved_grid[r, c] = digit
                    elif solved_grid[r, c] != digit:
                        # Should not happen if logic is correct and puzzle is valid
                        print(f"Error: Conflict during solution reconstruction at ({r},{c}). "
                              f"Tried to place {digit}, but found {solved_grid[r, c]}.", file=sys.stderr)
                        return None
                else:
                     print(f"Error: Invalid row_id {row_id} found in solution node {node_idx}.", file=sys.stderr)
                     return None

        # Final check: ensure all cells are filled
        if np.any(solved_grid == 0):
             # This might happen if the original puzzle had no solution,
             # but search() should have returned False in that case.
             # If search() returned True, this indicates an internal error.
             print("Internal Error: DLX search succeeded but solution grid still contains zeros.", file=sys.stderr)
             return None

        return solved_grid
    else:
        # Search failed, no solution found
        return None

def generate_writing_instructions(unsolved_board, solved_board, horizontal_cell_size, vertical_cell_size):
    instructions = []

    blank_cells = []
    for i in range(9):
        for j in range(9):
            if unsolved_board[i, j] == 0:
                blank_cells.append((i, j))

    # Snake pattern bottom to top
    sorted_cells = []
    for i in range(8, -1, -1):
        row_cells = [(i, j) for (i2, j) in blank_cells if i2 == i]
        if (8 - i) % 2 == 0:
            row_cells = sorted(row_cells, key=lambda x: -x[1])
        else:
            row_cells = sorted(row_cells, key=lambda x: x[1])
        sorted_cells.extend(row_cells)

    for (row, col) in sorted_cells:
        x_mm = float((col + 0.5) * horizontal_cell_size)
        y_mm = float(((8-row) + 0.5) * vertical_cell_size)
        digit = int(solved_board[row, col])
        instructions.append((round(x_mm, 2), round(y_mm, 2), digit))

    instructions = np.array(instructions, dtype=object)  # Convert to nice numpy array

    return instructions

def plan_robot_move(
    x1, y1, heading_deg,  # initial center position (where stamp is)
    x4, y4,               # desired center target (where stamp lands)
    vertical_cell_size,
    wheel_diameter=37,    # mm
    wheel_base=67.5,        # mm (distance between wheels)
    stamp_forward_offset=64,    # mm (center to stamp)
):
    # Constants
    wheel_circumference = math.pi * wheel_diameter
    dx = x4 - x1
    dy = y4 - y1

    if dy == 0 or math.isclose(dy, vertical_cell_size, abs_tol=0.1):  # if we are performing a "normal" move. I.e., moving only across or up

        if dy == 0:  # if we are just moving left-right
            # Translation
            if dx > 0:  # if we are just moving right
                translation_distance = dx
                translation_direction = "forward"
            else:  # if we are just moving left
                translation_distance = - dx
                translation_direction = "backward"
            translation_rotations = abs(translation_distance) / wheel_circumference

            # No initial or final rotation needed
            initial_rotation = (0, "NA", "NA", 0)
            final_rotation = (0, "NA", "NA", 0)
            translation = (translation_distance, "both", translation_direction, translation_rotations)

        elif math.isclose(dy, vertical_cell_size, abs_tol=0.1):  # If we are just moving up (will  not work if a whole row is already populated)
            # rotation right wheel to get the right wheel up to the final height
            initial_rotation_angle = - math.acos((wheel_base - vertical_cell_size)/wheel_base)  # radians (should confirm vertical cell size is correct argument here
            initial_rotation_wheel = "right"
            initial_rotation_direction = "forward"
            initial_rotation_rotations = abs(initial_rotation_angle * wheel_base) / wheel_circumference

            # rotate the left wheel to get the robot back to facing right
            final_rotation_angle = - initial_rotation_angle
            final_rotation_wheel = "left"
            final_rotation_direction = "forward"
            final_rotation_rotations = abs(final_rotation_angle * wheel_base) / wheel_circumference

            # reverse the robot to contour the drift from the rotations
            translation_distance = - (wheel_base * math.sin(final_rotation_angle)) + dx
            if translation_distance < 0:
                translation_direction = "backward"
            else:
                translation_direction = "forward"
            translation_rotations = abs(translation_distance) / wheel_circumference

            # assemble the instructions for each move
            initial_rotation = (initial_rotation_angle, initial_rotation_wheel, initial_rotation_direction, initial_rotation_rotations)
            final_rotation = (final_rotation_angle, final_rotation_wheel, final_rotation_direction, final_rotation_rotations)
            translation = (translation_distance, "both", translation_direction, translation_rotations)

        # define the order to execute the instructions
        move_sequence = [initial_rotation, final_rotation, translation]

    else:
        # 0. ROBOT GEOMETRIC PROPERTIES
        half_wheel_base = 0.5 * wheel_base
        distance_stamp_wheel = math.hypot(half_wheel_base, stamp_forward_offset)
        angle_cl_to_StampWheelLine = math.atan2(half_wheel_base, stamp_forward_offset)
        heading_rad = math.radians(heading_deg)


        # 1. INITIAL ROTATION
        # 1a. initial_rotation_angle
            # Initial right wheel position (x2, y2)
        right_wheel_angle = angle_cl_to_StampWheelLine - heading_rad
        x2 = x1 + distance_stamp_wheel * math.sin(right_wheel_angle)
        y2 = y1 - distance_stamp_wheel * math.cos(right_wheel_angle)
            # Final right wheel position (x3, y3)
        x3 = x4 - stamp_forward_offset
        y3 = y4 - half_wheel_base
            # Vector from current to target right wheel position
        dx = x3 - x2
        print(f"dx:{dx}")
        print(f"dy:{dy}")
        dy = y3 - y2
        phi = math.atan2(dx, dy)  # Rotation from vertical to final
        initial_rotation_angle = phi - heading_rad  # Rotation from initial to final
            # Normalize rotation to [-pi, pi] (is this necessary??)
        if initial_rotation_angle > math.pi:
            initial_rotation_angle -= 2 * math.pi
        elif initial_rotation_angle < - math.pi:
            initial_rotation_angle += 2 * math.pi
        initial_rotation_rotations = (wheel_base * initial_rotation_angle) / wheel_circumference

        # 2. TRANSLATION
        translation_distance = math.hypot(dx, dy)
        translation_rotations = translation_distance / wheel_circumference

        # 3. FINAL ROTATION
        final_rotation_angle = math.pi*0.5 - (initial_rotation_angle + heading_rad)
        print(f"initial rotation angle:{180/np.pi * initial_rotation_angle}")
        print(f"final rotation angle:{180/np.pi * final_rotation_angle}")
        final_rotation_rotations = (wheel_base * final_rotation_angle) / wheel_circumference

        # 4. MOVE SEQUENCE
        move_sequence = [(initial_rotation_angle, "left", "backward", -initial_rotation_rotations),
                         (translation_distance, "both", "forward", translation_rotations),
                         (final_rotation_angle, "left", "forward", final_rotation_rotations)
        ]

    return move_sequence

def print_moves(moves, i):
    # Print the move index
    print(f"\nMove {i}:")

    # Loop through each step in the move list
    for j, step in enumerate(moves):
        # Start the line for each step
        step_output = f"Step {j + 1}: ("

        # Loop through each item in the step and append it to the step_output
        for k in step:
            if isinstance(k, float):  # If the item is a float
                step_output += f"{k:.2f}, "  # Format it to 2 decimal places
            else:  # If the item is not a float, just add it as it is
                step_output += f"{k}, "

        # Remove the last comma and space, and close the tuple
        step_output = step_output.rstrip(', ') + ")"

        # Print the formatted step
        print(step_output)

def initialise_steppers():
    left_stepper_pins = [OutputDevice(5), OutputDevice(6), OutputDevice(16), OutputDevice(20)]
    right_stepper_pins = [OutputDevice(14), OutputDevice(15), OutputDevice(23), OutputDevice(24)]
    increments_per_revolution = 4096  # your motor specs
    step_sequence = [
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [1, 0, 0, 1]
    ]
    return left_stepper_pins, right_stepper_pins, step_sequence, increments_per_revolution

def set_increment(pins, increment):
    for pin, value in zip(pins, increment):
        pin.value = value

def move_stepper(pins, revolutions, direction):
    total_increments = int(abs(revolutions) * INCREMENTS_PER_REVOLUTION)
    if total_increments == 0:
        return

    # Estimate reasonable delay based on some movement speed
    delay = 0.0015  # adjust depending on your motor's capability

    sequence = STEP_SEQUENCE if direction == "forward" else STEP_SEQUENCE[::-1]

    for increment in range(total_increments):
        increment_step = sequence[increment % len(sequence)]
        set_increment(pins, increment_step)
        sleep(delay)

def move_both_steppers(left_revolutions, right_revolutions, left_direction, right_direction):
    left_thread = threading.Thread(target=move_stepper,
                                   args=(LEFT_STEPPER_PINS, left_revolutions, left_direction))
    right_thread = threading.Thread(target=move_stepper,
                                    args=(RIGHT_STEPPER_PINS, right_revolutions, right_direction))

    left_thread.start()
    right_thread.start()

    left_thread.join()
    right_thread.join()

def control_steppers(move_sequence):
    for move in move_sequence:
        distance_mm, motor, direction, revolutions = move  # ← Correct unpacking!

        if motor == "left":
            move_stepper(LEFT_STEPPER_PINS, revolutions, direction)
        elif motor == "right":
            move_stepper(RIGHT_STEPPER_PINS, revolutions, direction)
        elif motor == "both":
            move_both_steppers(revolutions, revolutions, direction, direction)

image = 'sudoku-v8.jpg'
# image = capture_image()
eroded, original, gray = preprocess_image(image)
contour = find_sudoku_contour(eroded, original)
contour_sorted = sort_corners(contour)
print(f"Contours sorted: {contour_sorted}")
print(type(contour_sorted))
world_distances_unfitted = pixel_to_world_distances(contour_sorted, debug=True)
world_distances_fitted, _ = fit_rectangle_least_squares(world_distances_unfitted)
print(f"World distances: \n{world_distances_fitted}")

robot_heading_deg, robot_position_new, horizontal_cell_size, vertical_cell_size = generate_real_world_coordinates(world_distances_fitted)
print(f"\nRobot Heading: {robot_heading_deg:.4f} degrees")
print(f"Robot Position: ({robot_position_new[0]:.2f}, {robot_position_new[1]:.2f}) mm")
print(f"Horizontal Cell Size: {horizontal_cell_size:.1f} mm\nVertical Cell Size: {vertical_cell_size:.1f} mm")
warped_image = warp_image(gray, contour_sorted, target_size=900)
# model = load_model("digit_cnn.keras")
interpreter = Interpreter(model_path="digit_cnn.tflite")
interpreter.allocate_tensors()
virtual_board, instructions = build_virtual_sudoku(warped_image)
print_virtual_board(virtual_board)
solved_board = solve_sudoku(virtual_board)
print_virtual_board(solved_board)
writing_instructions = generate_writing_instructions(virtual_board, solved_board, horizontal_cell_size, vertical_cell_size)


# Initial position
x_current = robot_position_new[0]
y_current = robot_position_new[1]
heading_deg = robot_heading_deg

# LEFT_STEPPER_PINS, RIGHT_STEPPER_PINS, STEP_SEQUENCE, INCREMENTS_PER_REVOLUTION = initialise_steppers()

# Loop through instructions
for i, instruction in enumerate(writing_instructions):
    x_target, y_target, digit = instruction

    # Call the plan_robot_move function to calculate the movement to the next target
    robot_path = plan_robot_move(x_current, y_current, heading_deg, x_target, y_target, vertical_cell_size)
    print_moves(robot_path, i)

    # CALL FUNCTION TO RUN STEPPER MOTORS
    #control_steppers(robot_path)
    sleep(0.1)
    # CALL FUNCTION TO SELECT DIGIT
    # CALL FUNCTION TO STAMP DIGIT

    # Update current position and heading for the next iteration
    x_current, y_current = x_target, y_target  # Update current position
    heading_deg = math.pi # Update heading after move
