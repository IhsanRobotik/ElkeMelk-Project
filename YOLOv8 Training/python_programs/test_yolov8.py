# I used this to test whether I had YOLO, as I had some issues throughout

from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Load a pre-trained model
model.info()  # Print model information
