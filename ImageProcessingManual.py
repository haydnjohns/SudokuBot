import cv2
import os
import numpy as np
import skimage.segmentation
import time

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

def process_board(image_path, target_size):
    unprocessed_image = cv2.imread(image_path)
    greyed_image = cv2.cvtColor(unprocessed_image, cv2.COLOR_BGR2GRAY)
    denoised_image = cv2.fastNlMeansDenoising(greyed_image, None, h=20, templateWindowSize=7, searchWindowSize=21)
    blurred_image = cv2.GaussianBlur(denoised_image, (3, 3), 100)
    thresholded_image = cv2.adaptiveThreshold(blurred_image, 255, 1, 1, 151, 3.2)
    inverted_image = cv2.bitwise_not(thresholded_image)

    kernel = np.ones((2, 2), np.uint8)
    dilated_image = cv2.dilate(inverted_image, kernel, iterations=0)
    eroded_image = cv2.erode(dilated_image, kernel, iterations=3)

    contours, hierarchy = cv2.findContours(eroded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    biggest_approx = None
    max_area = 0
    desired_points = 4

    for i in contours:
        area = cv2.contourArea(i)
        if area > 10000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.005 * peri, True)
            if len(approx) == desired_points and area > max_area:
                biggest_approx = approx
                max_area = area

    if biggest_approx is not None:
        biggest = biggest_approx.reshape((4, 2))
        biggest_sorted = rectify(biggest)
        biggest_target = np.array([[0, 0], [target_size, 0], [target_size, target_size], [0, target_size]], np.float32)
        retval = cv2.getPerspectiveTransform(biggest_sorted, biggest_target)
        warped_image = cv2.warpPerspective(greyed_image, retval, (target_size, target_size))
        return warped_image
    else:
        print(f"Warning: No suitable contour found in {image_path}.")
        return None

def extract_digit(i, j, processed_image, target_size):
    cell_size = target_size // 9
    digit = processed_image[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]
    thresh_digit = cv2.adaptiveThreshold(digit, 255, 1, 1, 99, 7)

    kernel = np.ones((2, 2), np.uint8)
    eroded_digit = cv2.erode(thresh_digit, kernel, iterations=2)
    dilated_digit = cv2.dilate(eroded_digit, kernel, iterations=1)
    cleared_digit = skimage.segmentation.clear_border(dilated_digit)

    try:
        contours_digit, _ = cv2.findContours(cleared_digit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours_digit:
            return None  # Return None instead of raising an exception
        cnt = max(contours_digit, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        cropped_digit = thresh_digit[y:y + h, x:x + w]

        # Create a 100x100 canvas and place the cropped digit in the center
        size = 100
        processed_digit = np.zeros((size, size), dtype=np.uint8)
        cx, cy = (size - w) // 2, (size - h) // 2
        processed_digit[cy:cy + h, cx:cx + w] = cropped_digit

    except ValueError:
        return None  # Return None if an error occurs

    # Resize the processed (or blank) digit to 28x28
    resized_digit = cv2.resize(processed_digit, (28, 28), interpolation=cv2.INTER_AREA)

    return resized_digit

def save_digits_to_array(image, target_size=900):
    digits_array = []  # List to store each 28x28 digit image

    if image is None:
        print("Skipping digit extraction due to processing failure.")
        return None

    for i in range(9):
        for j in range(9):
            extracted_digit = extract_digit(i, j, image, target_size)
            if extracted_digit is not None:
                digits_array.append(extracted_digit)

    # Convert to NumPy array with shape (num_digits, 28, 28)
    return np.array(digits_array)

# Set target size for processing
target_size = 900

# Define paths

start_time = time.time()

for d in range(0, 1):

    partial_start_time = time.time()

    input_dir = f"/Users/haydnjohns/Documents/Coding/Python/SudokuBot/Images/{d+1}"
    output_dir = f"/Users/haydnjohns/Python Projects/SudokuBot/Images/Digits/{d+1}"
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through Sudoku images
    all_digits = []  # To store all digits from all Sudoku images

    for d in range(0, 1):
        input_dir = f"/Users/haydnjohns/Documents/Coding/Python/SudokuBot/Images/{d + 1}"
        for filename in os.listdir(input_dir):
            if filename.endswith(".jpg"):  # Ensure we only process .jpg files
                image_path = os.path.join(input_dir, filename)

                # Process the Sudoku board
                processed_board = process_board(image_path, target_size)

                # Get digits as array and add to master list if processed successfully
                digits_array = save_digits_to_array(processed_board, target_size)
                if digits_array is not None:
                    all_digits.append(digits_array)

    # Combine all digits into a single array for training
    all_digits_np = np.concatenate(all_digits, axis=0)  # Shape will be (total_digits, 28, 28)

    # Save the array to a .npy file for future use
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(threshold=np.inf, linewidth=500)
    print(all_digits_np)
    print(all_digits_np.shape)
    np.save('all_digits.npy', all_digits_np)
    print("All digits saved to 'all_digits.npy'.")

    partial_end_time = time.time()

    partial_time = (partial_end_time - partial_start_time) / 60
    print(f"Time for digit {d+1}: {partial_time} minutes")

end_time = time.time()
print(f"Total time: {end_time - start_time:.2f} seconds")

