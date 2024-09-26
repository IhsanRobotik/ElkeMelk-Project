from ultralytics import YOLO

# Load the YOLOv8 model (use 'yolov8n.pt' for the nano version, or 'yolov8s.pt' for the small version)
model = YOLO('yolov8n.pt')

# Train the model using the dataset.yaml
model.train(data='C:/Users/gabri/Desktop/dataset/dataset.yaml', epochs=100, imgsz=640)

# After training, the best weights will be saved in the 'runs/train/exp/weights/best.pt' file
