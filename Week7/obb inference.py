import cv2
from ultralytics import YOLO

# Load the model
model = YOLO(r'C:\Users\Ihsan\Documents\GitHub\ElkeMelk-Project\models\obbV5.pt')

# Open a connection to the webcam (or video feed)
cap = cv2.VideoCapture(0)  # 0 is the ID for the default webcam. Change if using another source.

if not cap.isOpened():
    print("Error: Could not open video feed.")
    exit()

while True:
    # Capture frame-by-frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Run YOLO inference on the current frame
    results = model(frame)

    # Get the first image/frame results
    result = results[0]

    # Loop through the detected boxes and print each class and its bounding box coordinates
    for box in result.boxes:
        # Get the class index (integer) and the corresponding class name
        class_id = int(box.cls)
        class_name = result.names[class_id]

        # Get the bounding box coordinates (x1, y1, x2, y2)
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Print the class name and corresponding coordinates
        print(f"Class: {class_name}, Coordinates: (x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2})")

        # Draw the bounding box and class label on the frame
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{class_name}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the frame with bounding boxes and labels
    cv2.imshow('YOLO Live Detection', frame)

    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
