import cv2 as cv
import numpy as np
import torch
from ultralytics import YOLO

camera = 0  # Set to 0 for your default camera

# Ensure CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    print("CUDA is available and enabled.")

# Load the pre-trained YOLOv8 model
model = YOLO(r'C:\Users\Ihsan\Documents\GitHub\ElkeMelk-Project\models\rimV2.pt')   # Update with your model path

def main():
    # Initialize the video capture object
    cap = cv.VideoCapture(camera)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Camera not accessible.")
        return -1

    # Define the ROI coordinates
    roi_x, roi_y, roi_w, roi_h = 0, 260, 1280, 200  # Adjust these values as needed

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        # Draw the ROI rectangle on the frame
        cv.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)  # Green ROI rectangle

        # Run YOLOv8 inference on the frame
        results = model(frame, verbose=False, conf=0.75)

        # Check if results are available and handle them
        if results and results[0].obb:
            for i, obb in enumerate(results[0].obb):
                # Get the class label
                label = results[0].names[int(obb.cls[0])]

                # Process only the 'bottle_open' class
                if label == 'bottle_open':
                    # Get the OBB coordinates
                    vertices = obb.xyxyxyxy[0].cpu().numpy()  # Retrieve the vertices (4 points)

                    # Calculate the center point of the OBB
                    center_x = np.mean(vertices[:, 0])  # Average x-coordinates
                    center_y = np.mean(vertices[:, 1])  # Average y-coordinates

                    # Draw the OBB using the vertices
                    points = vertices.astype(int)
                    for j in range(len(points)):
                        cv.line(frame, tuple(points[j]), tuple(points[(j + 1) % len(points)]), (0, 255, 0), 2)

                    # Draw a circle at the center point
                    cv.circle(frame, (int(center_x), int(center_y)), 5, (255, 0, 0), -1)

                    # Print debug information
                    print(f"OBB {i}: Vertices = {vertices}, Center = ({center_x}, {center_y})")

        # Display the annotated frame
        cv.imshow("YOLOv8 Detection with ROI", frame)

        # Exit on 'q' key press
        if cv.waitKey(1) == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv.destroyAllWindows()

if _name_ == "_main_":
    main()