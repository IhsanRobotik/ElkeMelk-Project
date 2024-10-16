import cv2
from ultralytics import YOLO

# Load the model
model = YOLO('C:/Users/ihsan/Documents/GitHub/ElkeMelk-Project/models/elkemelk.pt')  # Replace with your model path

# Open a live feed source (0 is usually the default camera)
while True:
    # Run inference on the live video feed
    results = model(source=0, show=True, conf=0.25, iou=0.45, save=True)
    
    # Wait for 1 ms to check if "q" is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting program.")
        break

# Release resources and close windows
cv2.destroyAllWindows()