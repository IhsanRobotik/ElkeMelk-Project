import cv2 as cv
import numpy as np
import torch
from ultralytics import YOLO
import time

# Ensure CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    print("CUDA is available and enabled.")

# Load the pre-trained YOLOv8 model
model = YOLO(r"C:/Users/basti/Documents/GitHub/ElkeMelk-Project/models/obbV5.pt")  # Replace with your trained model path

# Set the model to use the GPU
model.to(device)

# Define the Y-coordinate threshold
Y_THRESHOLD = 235  # Adjust this value as needed to filter objects by y-coordinate

def main():
    mtx = np.array([[893.38874436, 0, 652.46300526],
                    [0, 892.40326491, 360.40764759],
                    [0, 0, 1]])
    dist = np.array([0.20148339, -0.99826633, 0.00147814, 0.00218007, 1.33627184])
    known_width_mm = 341
    known_pixel_width = 1280

    # Calculate conversion factor from pixels to mm
    conversion_factor = known_width_mm / known_pixel_width

    # Initialize the video capture object
    cap = cv.VideoCapture(0)  # Change to 0 for the default camera
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        return -1

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Get the frame dimensions
        h, w = frame.shape[:2]

        # Undistort the frame
        new_camera_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        undistorted_frame = cv.undistort(frame, mtx, dist, None, new_camera_mtx)

        # Run YOLOv8 inference on the entire frame
        results = model(undistorted_frame, verbose=False, conf=0.80)

        # Variables to store the rightmost 'bottle_open' center within the y-threshold
        rightmost_x = float('-inf')
        rightmost_center = None

        # Draw detections on the undistorted frame
        if results and results[0].obb:
            for i, obb in enumerate(results[0].obb):
                # Get the class label
                label = results[0].names[int(obb.cls[0])]

                # Get the OBB coordinates (vertices)
                vertices = obb.xyxyxyxy[0].cpu().numpy()  # Retrieve the vertices (4 points)

                # Draw the OBB using the vertices for all detected objects
                points = vertices.astype(int)
                for j in range(len(points)):
                    cv.line(undistorted_frame, tuple(points[j]), tuple(points[(j + 1) % len(points)]), (0, 255, 0), 2)

                # Draw the class label for all detected objects
                cv.putText(undistorted_frame, label, (points[0][0], points[0][1] - 10), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                # Process only the 'bottle_open' class for center point and specific handling
                if label == 'bottle_open':
                    # Calculate the center point of the OBB
                    center_x = np.mean(vertices[:, 0])  # Average x-coordinates
                    center_y = np.mean(vertices[:, 1])  # Average y-coordinates

                    # Filter out detections above the Y_THRESHOLD
                    if center_y > Y_THRESHOLD and center_x > 350:
                        # Draw a circle at the center point for 'bottle_open'
                        cv.circle(undistorted_frame, (int(center_x), int(center_y)), 5, (255, 0, 0), -1)
                    
                        # Display the on-screen coordinates in the window
                        on_screen_text = f"Coords: ({int(center_x)}, {int(center_y)})"
                        cv.putText(undistorted_frame, on_screen_text, (int(center_x) + 10, int(center_y) - 10), 
                                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                        # Check if this object is the rightmost one within the y-threshold
                        if center_x > rightmost_x:
                            rightmost_x = center_x
                            rightmost_center = (center_x, center_y)

        # Output the rightmost 'bottle_open' object center below the Y_THRESHOLD
        if rightmost_center:
            realX = rightmost_center[0] * conversion_factor
            realY = rightmost_center[1] * conversion_factor
            print("Rightmost 'bottle_open' Object Center (mm):", realX, realY)

        # Display the undistorted frame with annotations
        cv.imshow("Detected Bottles", undistorted_frame)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
