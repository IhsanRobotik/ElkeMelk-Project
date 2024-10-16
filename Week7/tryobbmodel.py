import cv2
import torch
import numpy as np

# Load the OBB YOLO model (use the path to your trained model)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/Ihsan/Documents/GitHub/ElkeMelk-Project/models/elkemelk.pt')  # Replace with your OBB model

# Open the camera
camera = cv2.VideoCapture(0)  # Change to 1 if you have multiple cameras

# Set desired width and height for the camera feed
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Check if the camera opened successfully
if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = camera.read()  # Capture frame from the camera
    if not ret:
        print("Error: Could not read frame.")
        break

    # Run YOLO inference on the frame
    results = model(frame)

    # Process YOLO OBB results
    # `results.xywhn` contains bounding box coordinates in (x_center, y_center, width, height, angle)
    obb_data = results.pandas().xyxy[0]  # Get OBB data as a DataFrame

    # Draw oriented bounding boxes on the frame
    for index, row in obb_data.iterrows():
        x_center = row['xcenter']  # Extracting coordinates of center
        y_center = row['ycenter']
        width = row['width']
        height = row['height']
        angle = row['angle']  # Angle for the oriented bounding box

        # You need to calculate the oriented box corners based on the angle
        box_points = cv2.boxPoints(((x_center, y_center), (width, height), angle))
        box_points = np.int0(box_points)

        # Draw the Oriented Bounding Box on the frame
        cv2.drawContours(frame, [box_points], 0, (0, 255, 0), 2)

        # Optionally, add class name and confidence score on the box
        class_name = row['name']  # Assuming the result has a class name
        confidence = row['confidence']
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(frame, label, (int(x_center), int(y_center)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Show the live frame with OBBs
    cv2.imshow('Live YOLO OBB Inference', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Press 'q' to quit the live feed
        break

# Release the camera and close all windows
camera.release()
cv2.destroyAllWindows()