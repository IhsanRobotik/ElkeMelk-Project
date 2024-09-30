# This was programmed for my MacBook, please change for Windows

from ultralytics import YOLO
import cv2
import numpy as np

# Load your YOLOv8 model
model = YOLO('/Users/kukumai/Downloads/runs/weights/best.pt')

# Access the webcam
cap = cv2.VideoCapture(0)  # Use 0 for the webcam

# Define a function for circle detection within the bounding box for class 17
def detect_single_circle_in_roi(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)   # Apply Gaussian blur to reduce noise
    
    # Apply Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=100,  # Minimum distance between detected circles
        param1=50,    # Higher threshold for Canny edge detector
        param2=30,    # Accumulator threshold for circle detection
        minRadius=20,  # Minimum radius of detected circles
        maxRadius=100  # Maximum radius of detected circles
    )
    
    # If circles are detected, return the largest one
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        largest_circle = max(circles, key=lambda c: c[2])  # Sort by the radius
        return largest_circle
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference on the frame
    results = model(frame)

    # Convert results to bounding boxes and class information
    for result in results:
        boxes = result.boxes.xyxy  # Bounding box coordinates
        confidences = result.boxes.conf  # Confidence scores
        classes = result.boxes.cls  # Class predictions
        
        # Loop through detected objects
        for i in range(len(boxes)):
            class_id = int(classes[i])
            x1, y1, x2, y2 = map(int, boxes[i])
            confidence = confidences[i]
            
            # Draw the bounding box for all detected objects
            label = f"Class {class_id} {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # If class is 'bottle opening' (class 17), perform circle detection
            if class_id == 17:
                # Extract region of interest (ROI) from the bounding box
                roi = frame[y1:y2, x1:x2]

                # Detect the most prominent circle in the ROI
                circle = detect_single_circle_in_roi(roi)
                if circle is not None:
                    # Draw the detected circle
                    (x, y, r) = circle
                    cv2.circle(roi, (x, y), r, (0, 255, 0), 4)  # Draw the outer circle
                    cv2.circle(roi, (x, y), 2, (0, 0, 255), 3)  # Draw the center of the circle

    # Show the frame with YOLO and circle detections
    cv2.imshow('YOLOv8 Inference with Circle Detection for Bottle Openings', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
