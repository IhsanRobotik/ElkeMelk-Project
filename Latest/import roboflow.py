import roboflow

# Initialize Roboflow with your API key
rf = roboflow.Roboflow(api_key="sbPpuvnmzLLlfiImywnB")

# Get the project
project = rf.workspace().project("bottle-rim-detection")

# Can specify weights_filename, default is "weights/best.pt"
version = project.version(3)

# Use raw string notation (r"your_path") to avoid issues with backslashes in the path
version.deploy("yolov8", r"C:\Users\Ihsan\Documents\GitHub\ElkeMelk-Project", "ah.pt")
