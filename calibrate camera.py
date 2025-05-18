import numpy as np
import cv2
import glob

contours_sorted= np.array([[312, 125], [1206, 125], [1454, 975], [45, 959]])



# --- Step 1: Set up checkerboard dimensions ---
CHECKERBOARD = (7, 7)  # number of internal corners per row and column
square_size = 13.0     # mm (or any real-world unit)

# --- Step 2: Prepare object points (3D points in real-world space) ---
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size  # scale to real-world units

# Arrays to store object points and image points from all images
objpoints = []  # 3D real-world points
imgpoints = []  # 2D points in image plane

# --- Step 3: Load images ---
images = glob.glob('/Users/haydnjohns/Documents/Coding/Python/SudokuBot/Camera Calibration/*.jpg')  # folder with checkerboard images
print(f"Found {len(images)} calibration images.")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the checkerboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # # Optional: Draw and display the corners
        # cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
        # cv2.imshow('Corners', img)
        # cv2.waitKey(1)

cv2.destroyAllWindows()

# --- Step 4: Calibrate the camera ---
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

# --- Step 5: Print results ---
print("Camera matrix:\n", camera_matrix)
print(type(camera_matrix))
print("Distortion coefficients:\n", dist_coeffs.ravel())
print(type(dist_coeffs))

# Save or print in Python-ready format
def format_numpy_array(name, array):
    flat = array.flatten() if array.ndim == 1 else array
    return f"{name} = np.array({repr(flat.tolist())}, dtype=np.float32)"

print(format_numpy_array("CAMERA_MATRIX", camera_matrix))
print(format_numpy_array("DIST_COEFFS", dist_coeffs))