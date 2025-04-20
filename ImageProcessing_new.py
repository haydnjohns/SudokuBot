import cv2
import numpy as np
from tensorflow.keras.models import load_model
from skimage.segmentation import clear_border
import os

# --- Constants ---
GRID_SIZE = 9
RECTIFIED_GRID_SIZE = 900
FINAL_CELL_SIZE = (28, 28)

# Load trained digit classification model
model = load_model("digit_cnn.keras")

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
    original_image = image.copy()
    greyed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(greyed , (3, 3), 0)
    image = cv2.adaptiveThreshold(image, 255, 1, 1, 151, 3.2)
    image = cv2.bitwise_not(image)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.erode(image, kernel, iterations=2)

    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    biggest = None
    max_area = 0
    margin = 10
    height, width = image.shape[:2]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 10000:
            x, y, w, h = cv2.boundingRect(cnt)
            if x <= margin or y <= margin or x + w >= width - margin or y + h >= height - margin:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4 and area > max_area:
                biggest = approx
                max_area = area

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
        return warped, sorted_pts.tolist()

    print("No suitable board contour found.")
    return None, None

def extract_digit(i, j, image, target_size=RECTIFIED_GRID_SIZE):
    cell_size_actual = target_size // 9
    cell_size_buffered = target_size // 6  # ~12.5% larger for buffer

    # Center of the cell
    center_x = j * cell_size_actual + cell_size_actual // 2
    center_y = i * cell_size_actual + cell_size_actual // 2

    # Calculate crop bounds
    half_size = cell_size_buffered // 2
    x1 = max(center_x - half_size, 0)
    y1 = max(center_y - half_size, 0)
    x2 = min(center_x + half_size, image.shape[1])
    y2 = min(center_y + half_size, image.shape[0])

    digit = image[y1:y2, x1:x2]
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
    resized = cv2.resize(canvas, FINAL_CELL_SIZE, interpolation=cv2.INTER_AREA)
    # cv2.imshow('resized', resized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if np.sum(resized > 30) < 20:
        return None
    return resized

def predict_digit(cell_img):
    cell_img = cell_img.reshape(1, 28, 28, 1).astype("float32") / 255.0
    pred = model.predict(cell_img, verbose=0)
    return np.argmax(pred) + 1  # add 1 since our labels were 1-9

def build_virtual_sudoku(board_img):
    virtual_board = []
    instructions = []

    for i in range(9):
        row = []
        for j in range(9):
            digit_img = extract_digit(i, j, board_img)
            if digit_img is not None:
                digit = predict_digit(digit_img)
                row.append(digit)
                instructions.append(f"{i+1},{j+1},{digit}")
            else:
                row.append(0)
        virtual_board.append(row)

    return virtual_board, instructions


def print_virtual_board(board):
    print("\nDetected Sudoku Grid:\n")

    # Print the top border first
    print("+ - - - + - - - + - - - +")

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

# --- Main ---
if __name__ == "__main__":
    # Load the input image (adjust the path as needed)
    input_image_path = "sudoku.png"  # Replace with your actual image
    image = cv2.imread(input_image_path)

    processed_board, corners = process_board(image)
    # cv2.imshow('processed_board', processed_board)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if processed_board is not None:
        virtual_board, instructions = build_virtual_sudoku(processed_board)
        print_virtual_board(virtual_board)

        print("\nInstruction Set (for robot):")
        for inst in instructions:
            print(inst)

        print("\nCorners of board in original image:")
        for idx, pt in enumerate(corners):
            print(f"Corner {idx+1}: {pt}")

        # Optional: Save instruction set
        with open("robot_instructions.txt", "w") as f:
            for inst in instructions:
                f.write(inst + "\n")

    else:
        print("Failed to process the board.")