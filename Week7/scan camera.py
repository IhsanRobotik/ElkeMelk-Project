import cv2

# Specify the camera index (0 for default camera, 1 for the second camera, and so on)
camera_index = 0  # Change this value to 1 or 2 for other cameras

# Initialize the video capture object
cap = cv2.VideoCapture(camera_index)

# Check if the camera is opened correctly
if not cap.isOpened():
    print(f"Error: Could not open camera with index {camera_index}")
    exit()

# Read frames from the camera in a loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from camera")
        break
    
    # Display the frame
    cv2.imshow('Camera Feed', frame)

    # Press 'q' to exit the loop and close the camera feed window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
