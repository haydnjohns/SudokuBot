import cv2
import numpy as np
from skimage.morphology import clear_border # Optional: for removing border-touching elements
import operator # Used for sorting contours

# --- Constants ---
TARGET_WARPED_SIZE = 900 # The target size of the perspective-warped Sudoku grid (square)
CELL_SIZE = TARGET_WARPED_SIZE // 9 # Size of one cell (100x100)
GRID_SIZE = 9
# Thresholds for digit detection (adjust based on testing)
MIN_CONTOUR_AREA_THRESHOLD = 50 # Minimum area for a contour to be considered a digit
MAX_CONTOUR_AREA_THRESHOLD = CELL_SIZE * CELL_SIZE * 0.8 # Max area (avoid detecting cell borders)
ASPECT_RATIO_THRESHOLD = 0.2 # Minimum aspect ratio (width/height or height/width)

# --- Placeholder for Machine Learning Model ---
def predict_digit(digit_roi):
    """
    Placeholder function for digit prediction using a trained ML model.
    Replace this with your actual model loading and prediction logic.

    Args:
        digit_roi (np.ndarray): Preprocessed image containing the isolated digit.

    Returns:
        int: The predicted digit (1-9), or 0 if prediction fails (or based on your model).
             Currently returns a placeholder value 0.
    """
    # --- ML Model Integration Point ---
    # 1. Ensure digit_roi is the correct size/format for your model (e.g., 28x28 grayscale).
    #    You might need resizing: cv2.resize(digit_roi, (28, 28))
    # 2. Preprocess if necessary (e.g., normalize pixel values).
    # 3. Pass the image to your loaded model's predict function.
    # 4. Return the predicted class label (integer 1-9).
    # --- End ML Model Integration Point ---

    print(f"    [INFO] Placeholder: Predicting digit (returning 0)")
    # Example resizing (if your model expects 28x28):
    # resized_roi = cv2.resize(digit_roi, (28, 28), interpolation=cv2.INTER_AREA)
    # In a real scenario, you would feed resized_roi (or similar) to your model
    return 0 # Placeholder return

# --- Image Processing Functions ---

def order_points(pts):
    """
    Orders the 4 points of a contour (corners of the Sudoku grid)
    in top-left, top-right, bottom-right, bottom-left order.

    Args:
        pts (np.ndarray): Array of 4 points (x, y).

    Returns:
        np.ndarray: Array of 4 points ordered correctly.
    """
    rect = np.zeros((4, 2), dtype="float32")
    pts = pts.reshape(4, 2) # Ensure shape is (4, 2)

    # Top-left point will have the smallest sum (x + y)
    # Bottom-right point will have the largest sum (x + y)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Top-right point will have the smallest difference (y - x)
    # Bottom-left point will have the largest difference (y - x)
    diff = np.diff(pts, axis=1) # calculates y - x
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # Return the ordered coordinates
    return rect

def find_sudoku_grid_contour(image):
    """
    Finds the largest 4-sided polygon contour in the image, assuming it's the Sudoku grid.

    Args:
        image (np.ndarray): Input image (preferably color).

    Returns:
        np.ndarray: The contour points (4x1x2) of the detected grid, or None if not found.
    """
    print("[INFO] Finding Sudoku grid contour...")
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 3) # Kernel size 7x7, sigma=3

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
                                   # Block size 11, C=2 (constants to subtract)

    # DEBUG: Show thresholded image
    # cv2.imshow("Thresholded Grid", thresh)
    # cv2.waitKey(0)

    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area (descending) and filter for potential grids
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    sudoku_contour = None
    for c in contours:
        # Approximate the contour shape
        perimeter = cv2.arcLength(c, True)
        epsilon = 0.02 * perimeter # Epsilon for approximation accuracy
        approx = cv2.approxPolyDP(c, epsilon, True)

        # Check if the approximated contour has 4 points (a quadrilateral)
        if len(approx) == 4:
            # Check if the area is reasonably large (e.g., > 20% of image area)
            if cv2.contourArea(approx) > (image.shape[0] * image.shape[1] * 0.1):
                 sudoku_contour = approx
                 print(f"[INFO] Found potential Sudoku grid contour with area: {cv2.contourArea(approx)}")
                 break # Found the largest quadrilateral

    if sudoku_contour is None:
        print("[ERROR] Sudoku grid contour not found.")
        return None

    # DEBUG: Draw the contour
    # debug_img = image.copy()
    # cv2.drawContours(debug_img, [sudoku_contour], -1, (0, 255, 0), 3)
    # cv2.imshow("Detected Contour", debug_img)
    # cv2.waitKey(0)

    return sudoku_contour

def warp_perspective_transform(image, contour_points, output_size):
    """
    Applies perspective transformation to rectify the Sudoku grid.

    Args:
        image (np.ndarray): Original image.
        contour_points (np.ndarray): 4 corner points of the Sudoku grid.
        output_size (int): The desired side length of the output square image (e.g., 900).

    Returns:
        np.ndarray: The perspective-warped square image of the Sudoku grid.
    """
    print("[INFO] Applying perspective warp...")
    # Order the points: top-left, top-right, bottom-right, bottom-left
    ordered_pts = order_points(contour_points)

    # Define the destination points for the square image
    # Note: output_size - 1 because coordinates are 0-indexed
    dst_pts = np.array([
        [0, 0],                       # Top-left
        [output_size - 1, 0],         # Top-right
        [output_size - 1, output_size - 1], # Bottom-right
        [0, output_size - 1]          # Bottom-left
    ], dtype="float32")

    # Calculate the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(ordered_pts, dst_pts)

    # Apply the perspective warp
    warped = cv2.warpPerspective(image, matrix, (output_size, output_size))

    print("[INFO] Perspective warp applied successfully.")
    # DEBUG: Show warped image
    # cv2.imshow("Warped Sudoku Grid", warped)
    # cv2.waitKey(0)

    return warped

def preprocess_cell(cell_image):
    """
    Preprocesses an individual cell image for OCR.

    Args:
        cell_image (np.ndarray): Image of a single Sudoku cell (color or grayscale).

    Returns:
        np.ndarray: Preprocessed binary image of the cell.
    """
    # Convert to grayscale if it's not already
    if len(cell_image.shape) == 3:
        gray_cell = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_cell = cell_image

    # Apply Gaussian blur
    blurred_cell = cv2.GaussianBlur(gray_cell, (5, 5), 0) # Smaller kernel for cells

    # Apply adaptive thresholding (inverse binary)
    # THRESH_BINARY_INV makes the digit white and background black
    thresh_cell = cv2.adaptiveThreshold(blurred_cell, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 3) # Adjust C if needed

    # --- Optional Morphological Operations ---
    # Kernel for morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)) # Small kernel

    # Example: Opening (Erosion followed by Dilation) to remove small noise
    # opened_cell = cv2.morphologyEx(thresh_cell, cv2.MORPH_OPEN, kernel)

    # Example: Closing (Dilation followed by Erosion) to close gaps in digits
    # closed_cell = cv2.morphologyEx(thresh_cell, cv2.MORPH_CLOSE, kernel)

    # Use the result from thresholding directly or after morphology
    processed_cell = thresh_cell # Change this if using morphology (e.g., = opened_cell)

    # --- Optional: Clear border-touching elements ---
    # Sometimes digits touch the extracted cell border, this can help remove them
    # Ensure image is boolean or compatible type for skimage
    try:
        processed_cell = clear_border(processed_cell)
        # Convert back to uint8 if needed by subsequent OpenCV functions
        processed_cell = (processed_cell * 255).astype(np.uint8)
    except ImportError:
        print("[WARN] skimage.morphology.clear_border not used (skimage not fully installed or import failed).")
    except Exception as e:
         print(f"[WARN] Error during clear_border: {e}. Skipping.")


    # --- Optional Padding ---
    # Add padding around the edges to ensure contours aren't cut off
    padding = 5 # pixels
    processed_cell = cv2.copyMakeBorder(processed_cell, padding, padding, padding, padding,
                                        cv2.BORDER_CONSTANT, value=0) # Black padding

    return processed_cell

# --- Main Recognition Pipeline ---

def extract_and_recognize_digits(warped_grid_image):
    """
    Extracts each cell, preprocesses it, detects/recognizes digits, and returns the Sudoku board.

    Args:
        warped_grid_image (np.ndarray): The 900x900 perspective-corrected Sudoku grid image.

    Returns:
        np.ndarray: A 9x9 NumPy array representing the Sudoku board (0 for empty cells).
                    Returns None if the input image is invalid.
    """
    if warped_grid_image is None or warped_grid_image.shape != (TARGET_WARPED_SIZE, TARGET_WARPED_SIZE, 3):
         if warped_grid_image is not None:
              print(f"[ERROR] Invalid warped image shape: {warped_grid_image.shape}. Expected: ({TARGET_WARPED_SIZE}, {TARGET_WARPED_SIZE}, 3)")
         else:
              print("[ERROR] Warped image is None.")
         return None

    print(f"[INFO] Extracting cells ({GRID_SIZE}x{GRID_SIZE}) of size {CELL_SIZE}x{CELL_SIZE}...")
    sudoku_board = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    for r in range(GRID_SIZE): # Rows
        for c in range(GRID_SIZE): # Columns
            # Calculate cell boundaries
            start_row = r * CELL_SIZE
            start_col = c * CELL_SIZE
            end_row = start_row + CELL_SIZE
            end_col = start_col + CELL_SIZE

            # Extract the cell from the warped grid
            cell = warped_grid_image[start_row:end_row, start_col:end_col]

            # Preprocess the cell
            processed_cell = preprocess_cell(cell)

            # --- Digit Detection Logic ---
            # Find contours within the preprocessed cell
            contours, _ = cv2.findContours(processed_cell.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            digit_contour = None
            max_area = 0

            # Find the largest contour that meets size criteria
            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h) if h > 0 else 0
                inv_aspect_ratio = h / float(w) if w > 0 else 0

                # Filter based on area and aspect ratio to isolate digits
                if (MIN_CONTOUR_AREA_THRESHOLD < area < MAX_CONTOUR_AREA_THRESHOLD and
                        area > max_area and
                        (aspect_ratio > ASPECT_RATIO_THRESHOLD or inv_aspect_ratio > ASPECT_RATIO_THRESHOLD) and
                        w < CELL_SIZE * 0.9 and h < CELL_SIZE * 0.9): # Ensure it's not the border itself
                    max_area = area
                    digit_contour = contour

            # --- Digit Recognition ---
            if digit_contour is not None:
                # Extract the bounding box of the detected digit contour
                x, y, w, h = cv2.boundingRect(digit_contour)

                # Extract the digit Region of Interest (ROI) from the *processed* cell
                # Apply padding correction if padding was added in preprocess_cell
                # This extracts from the padded image, so coordinates are relative to it
                digit_roi = processed_cell[y:y+h, x:x+w]

                # DEBUG: Show detected digit ROI
                # cv2.imshow(f"Digit ROI Cell ({r},{c})", digit_roi)
                # cv2.waitKey(1) # Small delay

                # --- Call the placeholder prediction function ---
                # Resize/preprocess ROI as needed for the actual model here
                predicted_value = predict_digit(digit_roi)

                if predicted_value is not None and 1 <= predicted_value <= 9:
                     sudoku_board[r, c] = predicted_value
                     print(f"  [INFO] Cell ({r},{c}): Detected digit (predicted: {predicted_value})")
                else:
                     # Handle cases where prediction fails or returns invalid value
                     sudoku_board[r, c] = 0 # Mark as empty/unknown
                     print(f"  [WARN] Cell ({r},{c}): Detected contour but failed prediction.")

            else:
                # No significant contour found, assume cell is empty
                sudoku_board[r, c] = 0
                # print(f"  [INFO] Cell ({r},{c}): Empty")


    print("[INFO] Cell extraction and digit recognition complete.")
    # cv2.destroyAllWindows() # Close any debug windows
    return sudoku_board

# --- Main Execution ---
if __name__ == "__main__":
    image_path = 'sudoku_image.png' # <--- CHANGE THIS TO YOUR SUDOKU IMAGE PATH

    # 1. Input: Read the Sudoku image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not read image from path: {image_path}")
    else:
        print(f"[INFO] Successfully loaded image: {image_path}")
        # DEBUG: Show original image
        # cv2.imshow("Original Image", image)
        # cv2.waitKey(0)

        # 2. Contour Detection: Find the Sudoku grid
        grid_contour = find_sudoku_grid_contour(image)

        if grid_contour is not None:
            # 3. Perspective Transformation: Warp the grid
            warped_grid = warp_perspective_transform(image, grid_contour, TARGET_WARPED_SIZE)

            # 4-7. Cell Extraction, Preprocessing, Digit Detection & Recognition
            sudoku_board_result = extract_and_recognize_digits(warped_grid)

            # 8. Output: Print the resulting Sudoku board
            if sudoku_board_result is not None:
                print("\n--- Recognized Sudoku Board ---")
                print(sudoku_board_result)
                print("-----------------------------")
            else:
                print("[ERROR] Failed to extract digits from the warped grid.")
        else:
            print("[ERROR] Failed to find the Sudoku grid in the image.")

        cv2.destroyAllWindows() # Close any remaining OpenCV windows