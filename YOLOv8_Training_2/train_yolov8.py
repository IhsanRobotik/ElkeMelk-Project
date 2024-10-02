from ultralytics import YOLO


model = YOLO('yolov8n.pt')

# Train the model using the dataset.yaml with high-resolution images, for 100 epochs
model.train(data='C:/Users/kukumav/Desktop/To-Train/dataset/dataset.yaml', epochs=100, imgsz=1024)