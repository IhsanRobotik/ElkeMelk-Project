import cv2
import torch
import os
from ultralytics import YOLO

# Ensure CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    print("CUDA is available and enabled.")

# Load the pre-trained YOLOv8 model
model = YOLO('DetectBottle.pt')  # Replace 'best.pt' with your trained model

# Set the model to use the GPU
model.to(device)

# Path to the folder containing the images
image_folder = r'C:\Users\Ihsan\Documents\SMRDelft\camera_calibration'

# Get a list of all .jpg files in the folder
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

# Loop through each image in the folder
for image_file in image_files:
    # Construct the full image path
    image_path = os.path.join(image_folder, image_file)
    
    # Read the image
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"Error: Could not read image {image_file}")
        continue
    
    # Run YOLOv8 inference on the image
    results = model(frame)
    
    # Get predictions for the frame
    predictions = results[0]  # Results object for the first (and only) image
    
    # Annotate the image with detected objects
    annotated_frame = predictions.plot()  # Draw bounding boxes and labels on the image
    
    # Display the resulting frame
    cv2.imshow('YOLOv8 Inference', annotated_frame)
    
    print(f"Showing {image_file}. Press Enter to continue to the next image.")
    
    # Wait for Enter key press (13 is the ASCII code for Enter)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 13:  # Enter key
            break
    
# Close OpenCV windows when done
cv2.destroyAllWindows()
