# In terminal, travel to the file location where this file is saved, and open terminal, travel to that location using "cd" in there, and then type "python live_feed_yolo.py"
# It should open a window afterwards

import cv2
from ultralytics import YOLO

# Load your YOLO model
model = YOLO("C:/Users/basti/Documents/GitHub/ElkeMelk-Project/YOLOv8_Training_2/runs/detect/train/weights/best.pt")  # Replace with the correct path to the model

# Change number inside () if you have more than 1 cameras, 0 is for default
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        break

    # Run YOLO model inference on the captured frame
    results = model(frame)

    # Annotate the frame with the detection results
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow('YOLOv8 Inference', annotated_frame)

    # Wait for 1ms and check if the user pressed the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
