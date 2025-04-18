import numpy as np

# Camera parameters
focal_length = 3.6  # mm
sensor_width = 4.8  # mm
sensor_height = 3.6  # mm
sensor_width_px = 1920  # pixels
sensor_height_px = 1440  # pixels
camera_height = 100.0  # mm
camera_angle = 45.0  # degrees

# Convert angle to radians
camera_angle_rad = np.radians(camera_angle)

# Define 4 points in pixels (top-left, top-right, bottom-left, bottom-right)
points_px = np.array([
    [307.2, 1267.8144],  # Point 1 (top-left)
    [1536, 1305.6],      # Point 2 (top-right)
    [268.8, 499.2],      # Point 3 (bottom-left)
    [1536, 415.8]        # Point 4 (bottom-right)
])

# Convert points from pixel to mm relative to sensor center
points_mm_centre = np.zeros((4, 2))
points_mm_centre[:, 0] = (points_px[:, 0] - 0.5 * sensor_width_px) * (sensor_width / sensor_width_px)
points_mm_centre[:, 1] = (points_px[:, 1] - 0.5 * sensor_height_px) * (sensor_height / sensor_height_px)

# Convert points to vectors in the sensor coordinate system (normalized by focal length)
points_vectors = np.hstack((np.ones((4, 1)), points_mm_centre / focal_length))

# Rotation matrix for the camera's tilt (pitch only)
rotation_matrix = np.array([
    [np.cos(camera_angle_rad), 0, -np.sin(camera_angle_rad)],
    [0, 1, 0],  # No rotation in the y-axis (left-right tilt)
    [np.sin(camera_angle_rad), 0, np.cos(camera_angle_rad)]
])

# Apply rotation to get points in global coordinates
points_global = rotation_matrix @ points_vectors.T

# Calculate the scaling factor for each point to project to the ground level
k_values = camera_height / points_global[2, :]  # divide by the z-component (height above ground)

# Calculate real-world distances for each point
distances = np.zeros((4, 2))  # array to store [horizontal_distance, depth_distance]
distances[:, 0] = k_values * points_global[1, :]  # Horizontal (left/right) distances
distances[:, 1] = k_values * points_global[0, :]  # Depth (forward/backward) distances

# Output results
print("\n")
for i, (horizontal, depth) in enumerate(distances):
    print(f"Point {i+1}")
    print(f"Horizontal distance from camera: {horizontal:.1f} mm")
    print(f"Depth distance from camera: {depth:.1f} mm")
    print(f"Vertical distance from camera: {camera_height:.1f} mm\n")