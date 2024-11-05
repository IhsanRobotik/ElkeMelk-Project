import cv2 as cv
import numpy as np
import torch
from ultralytics import YOLO

# Load the pre-trained YOLOv8 OBB model
model = YOLO(r"C:/Users/gabri/Downloads/obbV5.pt")  # Update with your OBB model path

# Ensure CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Initialize the video capture object
cap = cv.VideoCapture(0)  # Change to 0 for the default camera

if not cap.isOpened():
    print("Camera not accessible.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    # Run YOLOv8 inference on the captured frame
    results = model(frame, verbose=False, conf=0.45)

    # Initialize variables to track the leftmost object (bottle_neck)
    leftmost_x = float('inf')
    leftmost_center = None
    bottle_angle = None

    # Iterate over the results and process 'bottle_neck' class
    if results and results[0].obb:
        for obb in results[0].obb:
            label = results[0].names[int(obb.cls[0])]

            if label == 'bottle_neck':
                # Get the OBB vertices and extract angle from xywhr
                vertices = obb.xyxyxyxy[0].cpu().numpy()
                xywhr = obb.xywhr[0].cpu().numpy()  # Get the xywhr format
                
                bottle_angle = np.degrees(xywhr[4])  # The angle is in radians, convert to degrees

                # Draw the OBB using the vertices (4 points)
                for i in range(len(vertices)):
                    pt1 = tuple(vertices[i].astype(int))
                    pt2 = tuple(vertices[(i + 1) % len(vertices)].astype(int))
                    cv.line(frame, pt1, pt2, (0, 255, 0), 2)

                # Calculate the center point of the OBB
                center_x = np.mean(vertices[:, 0])  # Average x-coordinates
                center_y = np.mean(vertices[:, 1])  # Average y-coordinates

                # Draw a circle at the center point
                cv.circle(frame, (int(center_x), int(center_y)), 5, (255, 0, 0), -1)

                # Display the class name, coordinates, and angle on the screen
                on_screen_text = f"{label} Coords: ({int(center_x)}, {int(center_y)})"
                cv.putText(frame, on_screen_text, (int(center_x) + 10, int(center_y) - 10), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                angle_text = f"Angle: {bottle_angle:.2f} degrees"
                cv.putText(frame, angle_text, (int(center_x) + 10, int(center_y) + 20),
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # Check if this object is the leftmost one
                if center_x < leftmost_x:
                    leftmost_x = center_x
                    leftmost_center = (center_x, center_y)

    # Output the leftmost object's center and its angle
    if leftmost_center and bottle_angle is not None:
        print(f"Leftmost bottle_neck center: {leftmost_center}, Angle: {bottle_angle:.2f} degrees")

    # Display the annotated frame with OBB, center, class name, and angle
    cv.imshow("Bottle Detection with OBB", frame)

    # Break the loop if 'q' is pressed
    if cv.waitKey(1) == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv.destroyAllWindows()
