'''
This script is for generating data
1. Provide desired path to store images.
2. Press 'c' to capture image and display it.
3. Press any button to continue.
4. Press 'q' to quit.
'''

import cv2
import numpy as np

# Open the camera
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Check if the camera opened successfully
if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

# Define camera matrix and distortion coefficients
mtx = np.array([[893.38874436, 0, 652.46300526],
                [0, 892.40326491, 360.40764759],
                [0, 0, 1]])
dist = np.array([0.20148339, -0.99826633, 0.00147814, 0.00218007, 1.33627184])

path = r"C:\Users\Ihsan\Documents\SMRDelft\camera_calibration\mydata"
count = 0

while True:
    ret, img = camera.read()
    if not ret:
        print("Error: Could not read image.")
        break

    # Get the frame dimensions
    h, w = img.shape[:2]

    # Calculate the optimal new camera matrix and undistort the image
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(img, mtx, dist, None, new_camera_mtx)

    # Crop the undistorted image to the ROI
    x, y, w, h = roi
    undistorted_img = undistorted_img[y:y+h, x:x+w]

    # Display the undistorted image
    cv2.imshow("img", undistorted_img)

    key = cv2.waitKey(1) & 0xFF  # Use a small delay to allow for key detection

    if key == ord('c'):
        name = f"{path}\\image_{count}.jpg"  # Adjusted file naming
        cv2.imwrite(name, undistorted_img)  # Save the undistorted image
        print(f"Image saved as {name}")
        count += 1

    if key == ord('q'):
        break

# Release the camera and destroy all OpenCV windows
camera.release()
cv2.destroyAllWindows()
