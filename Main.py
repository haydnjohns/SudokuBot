import numpy as np
import cv2
import math
from time import sleep
from gpiozero import OutputDevice
from skimage.segmentation import clear_border
import threading  # Allows motors to run simultaneously
try:
    # Try for pi
    from ai_edge_litert.interpreter import Interpreter
except ModuleNotFoundError:
    # Fallback to full TensorFlow (for Mac / PC)
    from tensorflow.lite.python.interpreter import Interpreter

def preprocess_image(image_path, debug=False):
    image = cv2.imread(image_path)
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
    stamper_setback = 16  # mm
    focal_length = 3.744  # mm
    sensor_width = 3.6    # mm
    sensor_height = 2.7   # mm
    sensor_width_px = 2592
    sensor_height_px = 1944
    camera_height = 82.5  # mm
    camera_angle = 23.45 # degrees
    camera_angle_rad = np.radians(camera_angle)

    # --- Convert pixel coordinates to mm relative to the center of the sensor ---
    points_mm_centre = np.zeros((4, 2))
    points_mm_centre[:, 0] = ((contour_sorted[:, 0] - 0.5 * sensor_width_px) *
                              (sensor_width / sensor_width_px))  # X axis
    points_mm_centre[:, 1] = ((sensor_height_px - contour_sorted[:, 1] - 0.5 * sensor_height_px) *
                              (sensor_height / sensor_height_px))  # Y axis

    if debug:
        print(f"\n[Step 1] Pixel coordinates relative to sensor centre (mm):\n{points_mm_centre}")

    # --- Construct 3D rays in camera coordinates ---
    points_camera = np.zeros((4, 3))
    points_camera[:, 0] = points_mm_centre[:, 0]  # X
    points_camera[:, 1] = points_mm_centre[:, 1]  # Y
    points_camera[:, 2] = focal_length            # Z (depth from lens center)
    if debug:
        print(f"\n[Step 2] Vector from focal point to projected sensor (mm)"
              f"\n(left-right, sensor-up sensor-down, normal to sensor) :\n{points_camera}")

    # --- Rotate rays to world coordinates (camera tilt) ---
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(camera_angle_rad), -np.sin(camera_angle_rad)],
        [0, np.sin(camera_angle_rad),  np.cos(camera_angle_rad)]
    ])
    points_global = (rotation_matrix @ points_camera.T).T  # shape (4, 3)
    if debug:
        print(f"\n[Step 3] Vector from focal point to projected sensor in global coordinates (mm)"
              f"\n(left-right, global-height, paper-depth):\n{points_global}")

    # --- Project onto ground plane (scale vectors so Z=0) ---
    k_values = - camera_height / points_global[:, 1]  # Scaling factors to flatten to ground
    if debug:
        print(f"\n[Step 4] Scaling factor to project sensor to ground:\n{k_values}")

    ground_coords = np.zeros((4, 2))
    ground_coords[:, 0] = k_values * points_global[:, 0]  # X (left-right)
    ground_coords[:, 1] = k_values * points_global[:, 2] + stamper_setback  # Y (depth)
    if debug:
        print(f"\n[Step 5] Vector from point :\n{ground_coords}")

    return ground_coords

def generate_real_world_coordinates(world_distances):

    # Step 1: Calculate average grid width and height
    top_left, top_right, bottom_right, bottom_left = world_distances

    top_width = np.linalg.norm(top_left - top_right)
    bottom_width = np.linalg.norm(bottom_left - bottom_right)
    average_grid_width = (top_width + bottom_width) / 2

    left_height = np.linalg.norm(top_left - bottom_left)
    right_height = np.linalg.norm(top_right - bottom_right)
    average_grid_height = (left_height + right_height) / 2

    # Step 2: Assume square grid based on average size
    average_grid_size = (average_grid_width + average_grid_height) / 2

    destination_points = np.array([
        [0, average_grid_size],  # Top-left
        [average_grid_size, average_grid_size],  # Top-right
        [average_grid_size, 0],  # Bottom-right
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
    robot_original_position = np.array([0, 0])  # In robot's own frame
    robot_position_new = rotation_matrix @ robot_original_position + translation_vector

    # Step 5: Calculate robot heading
    # Extract rotation angle from rotation matrix
    heading_radians = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    robot_heading_deg = np.degrees(heading_radians)

    # Step 6: Calculate cell size
    cell_size = average_grid_size / 9


    return robot_heading_deg, robot_position_new, cell_size

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
    cell_size_buffered = target_size // 6  # ~12.5% larger for buffer

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
    eroded = cv2.erode(thresh_digit, kernel, iterations=2)
    dilated = cv2.dilate(eroded, kernel, iterations=1)
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

# def predict_digit_full_model(cell_img):
#     cell_img = cell_img.reshape(1, 28, 28, 1).astype("float32") / 255.0
#     pred = model.predict(cell_img, verbose=0)
#     return np.argmax(pred) + 1  # add 1 since our labels were 1-9

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

def solve_sudoku(board):
    def is_valid(board, row, col, num):
        # Check row
        if num in board[row]:
            return False
        # Check column
        if num in board[:, col]:
            return False
        # Check 3x3 box
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        if num in board[start_row:start_row+3, start_col:start_col+3]:
            return False
        return True

    def solve(board):
        for row in range(9):
            for col in range(9):
                if board[row, col] == 0:
                    for num in range(1, 10):
                        if is_valid(board, row, col, num):
                            board[row, col] = num
                            if solve(board):
                                return True
                            board[row, col] = 0
                    return False  # No valid number found, backtrack
        return True  # Solved!

    board_copy = board.copy()  # Avoid modifying the original board
    solve(board_copy)
    return board_copy

def generate_writing_instructions(unsolved_board, solved_board, cell_size_mm):
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
        x_mm = float((col + 0.5) * cell_size_mm)
        y_mm = float(((8-row) + 0.5) * cell_size_mm)
        digit = int(solved_board[row, col])
        instructions.append((round(x_mm, 2), round(y_mm, 2), digit))

    instructions = np.array(instructions, dtype=object)  # Convert to nice numpy array

    return instructions

def plan_robot_move(
    x1, y1, heading_deg,  # initial center position
    x4, y4,               # desired center target (where stamp lands)
    cell_size,
    wheel_diameter=37,    # mm
    wheel_base=67.5,        # mm (distance between wheels)
    stamp_forward_offset=64,    # mm (center to stamp)
):
    # Constants
    wheel_circumference = math.pi * wheel_diameter
    dx = x4 - x1
    dy = y4 - y1

    if dy == 0 or math.isclose(dy, cell_size, abs_tol=0.1):  # if we are performing a "normal" move. I.e., moving only across or up

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

        elif math.isclose(dy, cell_size, abs_tol=0.1):  # If we are just moving up (will  not work if a whole row is already populated)
            # rotation right wheel to get the right wheel up to the final height
            initial_rotation_angle = - math.acos((wheel_base - cell_size)/wheel_base)  # radians
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
        final_rotation_angle = initial_rotation_angle - heading_rad
        final_rotation_rotations = (wheel_base * final_rotation_angle) / wheel_circumference

        # 4. MOVE SEQUENCE
        move_sequence = [(initial_rotation_angle, "left", "forward", initial_rotation_rotations),
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
    # Define GPIO pins for left stepper motor
    left_stepper_pins = [OutputDevice(5), OutputDevice(6), OutputDevice(16), OutputDevice(20)]

    # Define GPIO pins for right stepper motor
    right_stepper_pins = [OutputDevice(21), OutputDevice(22), OutputDevice(23), OutputDevice(24)]

    # Total increments (half-steps) for one full shaft revolution (updated to 512)
    increments_per_revolution = 512  # Based on your motor's specs

    # Step sequence for half-stepping
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
    """Set the stepper motor pins for the current increment (small movement)."""
    for pin, value in zip(pins, increment):
        pin.value = value

def move_stepper(pins, rotations, direction, time_per_revolution):
    """
    Move the stepper motor a given number of rotations in a specified direction.

    Args:
    - pins: The GPIO pins for the stepper motor (either left or right).
    - rotations: The number of rotations (positive for forward, negative for backward).
    - direction: The direction of rotation ('forward' or 'backward').
    - time_per_revolution: The time in seconds it should take for one full revolution.
    """
    # Calculate total increments (small movements) needed for the given rotations
    total_increments = int(abs(rotations) * INCREMENTS_PER_REVOLUTION)

    # Delay per increment based on time per revolution and total increments
    delay = time_per_revolution / INCREMENTS_PER_REVOLUTION

    # Select the correct sequence based on the direction
    sequence = STEP_SEQUENCE if direction == "forward" else STEP_SEQUENCE[::-1]

    # Move the motor the required number of increments
    for increment in range(total_increments):
        increment_step = sequence[increment % len(sequence)]  # Cycle through the sequence
        set_increment(pins, increment_step)
        sleep(delay)

def move_both_steppers(left_rotations, right_rotations, left_direction, right_direction, time_per_revolution):
    """
    Move both stepper motors simultaneously (synchronized movement).

    Args:
    - left_rotations: The number of rotations for the left motor.
    - right_rotations: The number of rotations for the right motor.
    - left_direction: The direction for the left motor ('forward' or 'backward').
    - right_direction: The direction for the right motor ('forward' or 'backward').
    - time_per_revolution: The time in seconds it should take for one full revolution.
    """
    # Create threads to move both motors simultaneously
    left_thread = threading.Thread(target=move_stepper,
                                   args=(LEFT_STEPPER_PINS, left_rotations, left_direction, time_per_revolution))
    right_thread = threading.Thread(target=move_stepper,
                                    args=(RIGHT_STEPPER_PINS, right_rotations, right_direction, time_per_revolution))

    # Start both threads
    left_thread.start()
    right_thread.start()

    # Wait for both threads to finish before continuing
    left_thread.join()
    right_thread.join()

def control_steppers(move_sequence, time_per_revolution=5):  # Adjust default time per revolution here
    """
    Control the stepper motors to perform a series of movements based on the move_sequence.

    Args:
    - move_sequence: List of tuples containing (rotation_angle, motor, direction, time_for_step)
    - time_per_revolution: The time per revolution for the motor (default 5.12 seconds).
    """
    for i, move in enumerate(move_sequence):
        rotation_angle, motor, direction, time_for_step = move

        # Select the appropriate stepper motor
        if motor == "left":
            pins = LEFT_STEPPER_PINS
            move_stepper(pins, rotation_angle, direction, time_for_step)
        elif motor == "right":
            pins = RIGHT_STEPPER_PINS
            move_stepper(pins, rotation_angle, direction, time_for_step)
        elif motor == "both":  # Move both motors simultaneously
            # For both motors, we pass the same rotation angle and direction for both motors
            move_both_steppers(rotation_angle, rotation_angle, direction, direction, time_for_step)



image_path = 'sudoku.png'
eroded, original, gray = preprocess_image(image_path)
contour = find_sudoku_contour(eroded, original)
contour_sorted = sort_corners(contour)
world_distances = pixel_to_world_distances(contour_sorted)
robot_heading_deg, robot_position_new, cell_size = generate_real_world_coordinates(world_distances)
print(f"\nRobot Heading: {robot_heading_deg:.4f} degrees")
print(f"Robot Position: ({robot_position_new[0]:.2f}, {robot_position_new[1]:.2f}) mm")
print(f"Estimated Cell Size: {cell_size:.0f} mm")
warped_image = warp_image(gray, contour_sorted, target_size=900)
# model = load_model("digit_cnn.keras")
interpreter = Interpreter(model_path="digit_cnn.tflite")
interpreter.allocate_tensors()
virtual_board, instructions = build_virtual_sudoku(warped_image)
print_virtual_board(virtual_board)
solved_board = solve_sudoku(virtual_board)
print_virtual_board(solved_board)
writing_instructions = generate_writing_instructions(virtual_board, solved_board, cell_size)


# Initial position
x_current = robot_position_new[0]
y_current = robot_position_new[1]
heading_deg = robot_heading_deg

# writing_instructions = [
#     [371.11, 21.83, 8],
#     [327.45, 21.83, 7],
#     [283.79, 21.83, 4],
#     [240.13, 21.83, 1],
#     [196.47, 21.83, 3],
#     [109.15, 21.83, 9],
#     [65.49, 21.83, 5],
#     [21.83, 65.49, 1],
#     [152.81, 65.49, 8],
#     [196.47, 65.49, 9],
#     [327.45, 65.49, 6],
#     [371.11, 65.49, 2],
#     [371.11, 109.15, 5],
#     [283.79, 109.15, 1],
#     [240.13, 109.15, 2],
#     [196.47, 109.15, 4],
#     [152.81, 109.15, 7],
#     [21.83, 109.15, 6],
#     [65.49, 152.81, 7],
#     [109.15, 152.81, 1],
#     [152.81, 152.81, 4],
#     [240.13, 152.81, 6],
#     [283.79, 152.81, 2],
#     [371.11, 152.81, 3],
#     [371.11, 196.47, 4],
#     [327.45, 196.47, 1],
#     [283.79, 196.47, 9],
#     [109.15, 196.47, 6],
#     [65.49, 196.47, 3],
#     [21.83, 196.47, 5],
#     [21.83, 240.13, 8],
#     [109.15, 240.13, 4],
#     [152.81, 240.13, 9],
#     [240.13, 240.13, 3],
#     [283.79, 240.13, 6],
#     [327.45, 240.13, 5],
#     [371.11, 283.79, 9],
#     [240.13, 283.79, 7],
#     [196.47, 283.79, 6],
#     [152.81, 283.79, 3],
#     [109.15, 283.79, 5],
#     [21.83, 283.79, 4],
#     [21.83, 327.45, 3],
#     [65.49, 327.45, 6],
#     [196.47, 327.45, 8],
#     [240.13, 327.45, 9],
#     [371.11, 327.45, 1],
#     [327.45, 371.11, 3],
#     [283.79, 371.11, 5],
#     [196.47, 371.11, 2],
#     [152.81, 371.11, 1],
#     [109.15, 371.11, 8],
#     [65.49, 371.11, 9],
#     [21.83, 371.11, 7]
# ]

left_stepper_pins, right_stepper_pins, step_sequence, increments_per_revolution = initialise_steppers()
# Loop through instructions
for i, instruction in enumerate(writing_instructions):
    x_target, y_target, digit = instruction

    # Call the plan_robot_move function to calculate the movement to the next target
    robot_path = plan_robot_move(x_current, y_current, heading_deg, x_target, y_target, cell_size)
    print_moves(robot_path, i)

    # CALL FUNCTION TO RUN STEPPER MOTORS
    # control_steppers(robot_path)
    # CALL FUNCTION TO SELECT DIGIT
    # CALL FUNCTION TO STAMP DIGIT

    # Update current position and heading for the next iteration
    x_current, y_current = x_target, y_target  # Update current position
    heading_deg = math.pi # Update heading after move
