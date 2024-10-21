import cv2 as cv
import numpy as np
import torch
from ultralytics import YOLO
import socket
import ast
import time
camera = 1

# Ensure CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    print("CUDA is available and enabled.")

# Load the pre-trained YOLOv8 OBB model
model = YOLO(r"C:/Users/basti/Documents/GitHub/ElkeMelk-Project/models/obbV5.pt")  # Update with your OBB model path

# Part II variables
bottlecoordsX = 149.59208498001098                       #Bottle coordinates X via cameraview
bottlecoordsY = 90.73866143226624                         #Bottle coordinates Y via cameraview

robotcoordsX = 335.18                              #Robot coordinates X, real world
robotcoordsY = -489.44                              #Robot coordinates Y, real world

offsetX = (robotcoordsX) + (bottlecoordsY)
offsetY = (robotcoordsY) + (bottlecoordsX)

firstposX = 485.64                                   #First position of the robot
firstposY = -502.46                                  #First position of the robot

array = [firstposX, firstposY]

counter = 0

def main():
    global array
    mtx = np.array([[893.38874436, 0, 652.46300526],
                    [0, 892.40326491, 360.40764759],
                    [0, 0, 1]])
    dist = np.array([0.20148339, -0.99826633, 0.00147814, 0.00218007, 1.33627184])
    known_width_mm = 341
    known_pixel_width = 1280

    # Calculate conversion factor from pixels to mm
    conversion_factor = known_width_mm / known_pixel_width

    # Initialize the video capture object
    cap = cv.VideoCapture(camera)  # Change to 0 for the default camera
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Camera not accessible.")
        return -1

    # Define the ROI coordinates (adjust as necessary)
    roi_x, roi_y, roi_w, roi_h = 0, 235, 1280, 250  # Green border ROI dimensions

    while True:
        time.sleep(1)
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame.")
            break

        # Get the frame dimensions
        h, w = frame.shape[:2]

        # Undistort the frame
        new_camera_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), -1, (w, h))
        undistorted_frame = cv.undistort(frame, mtx, dist, None, new_camera_mtx)

        # Crop the undistorted frame to the ROI
        roi_frame = undistorted_frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        # Run YOLOv8 OBB inference on the cropped ROI frame
        results = model(roi_frame, verbose=False, conf=0.45)  # Lower confidence threshold

        # Copy ROI frame for annotations
        annotated_frame = undistorted_frame.copy()

        # Initialize variables to track the leftmost object
        leftmost_x = float('-inf')  # Initialize to infinity, to ensure any value of center_x will be smaller.
        leftmost_center = None

        detected_classes = []  # To log detected classes

        # Iterate over the results and find the leftmost detected object
        if results and results[0].obb:
            # print("Detected objects:")  # Debugging info
            for i, obb in enumerate(results[0].obb):
                # Get the class label
                label = results[0].names[int(obb.cls[0])]
                detected_classes.append(label)  # Log detected class labels
                # print(f"Object {i}: {label}")  # Debug: Print object label

                # Get the OBB coordinates (vertices)
                vertices = obb.xyxyxyxy[0].cpu().numpy()  # Retrieve the vertices (4 points)

                # Adjust the coordinates to the original frame by adding the ROI offset
                vertices[:, 0] += roi_x  # Adjust x-coordinates
                vertices[:, 1] += roi_y  # Adjust y-coordinates

                # Draw the OBB using the vertices for all detected objects
                points = vertices.astype(int)
                for j in range(len(points)):
                    cv.line(annotated_frame, tuple(points[j]), tuple(points[(j + 1) % len(points)]), (0, 255, 0), 2)

                # Draw the class label for all detected objects
                cv.putText(annotated_frame, label, (points[0][0], points[0][1] - 10), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                # Process only the 'bottle_open' class for center point and specific handling
                if label == 'bottle_open':
                    # Calculate the center point of the OBB
                    center_x = np.mean(vertices[:, 0])  # Average x-coordinates
                    center_y = np.mean(vertices[:, 1])  # Average y-coordinates

                    # Draw a circle at the center point for 'bottle_open'
                    cv.circle(annotated_frame, (int(center_x), int(center_y)), 5, (255, 0, 0), -1)

                    # Display the on-screen coordinates in the window
                    on_screen_text = f"Coords: ({int(center_x)}, {int(center_y)})"
                    cv.putText(annotated_frame, on_screen_text, (int(center_x) + 10, int(center_y) - 10), 
                               cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    print(f"Leftmost object center: ({center_x}, {center_y})")

                    # Check if this object is the leftmost one
                    if center_x > leftmost_x:
                        leftmost_x = center_x
                        leftmost_center = (center_x, center_y)

        # Output the leftmost object's center and calculate real-world coordinates
        if leftmost_center:
            realX = leftmost_center[0] * conversion_factor
            realY = leftmost_center[1] * conversion_factor

            deltaY = (-1) * realX
            deltaX = (-1) * realY  # Reversing the coordinates here

            pickupX = deltaX + offsetX + ((firstposX - array[0]) * (-1))
            pickupY = deltaY + offsetY + ((firstposY - array[1]) * (-1))

            print(f"mm coords: {realX}, {realY}")
            print(f"robot coords: {pickupX}, {pickupY}")
        else:
            # print("No 'bottle_open' object detected in the current frame.")
            print(f"Detected classes: {detected_classes}")  # Print all detected classes

        # Draw the green border around the ROI (as per your request)
        cv.rectangle(annotated_frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)

        # Display the result for the detected objects with the green border
        cv.imshow("Detected OBB Objects", annotated_frame)  # Show annotated frame with detections

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
