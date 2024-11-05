import cv2 as cv
import numpy as np
import torch
from ultralytics import YOLO
import socket
import ast

# Ensure CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    print("CUDA is available and enabled.")

# Load the pre-trained YOLOv8 model
model = YOLO(r'C:\Users\Ihsan\Documents\GitHub\ElkeMelk-Project\models\rimV2.pt') 

# Set the model to use the GPU
model.to(device)

# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def socket():
    # Bind the socket to the address and port
    s.bind(("192.168.0.1", 5005))
    print("listening for connection")
    # Listen for incoming connections
    s.listen(5)
    # Accept a connection from a client
    clientsocket, address = s.accept()
    print(f"Connection from {address} has been established!")
# socket()

bottlecoordsX = 152.16
bottlecoordsY = 94.07

robotcoordsX = 133.90
robotcoordsY = -908.72

offsetX = (robotcoordsX) + (bottlecoordsY)
offsetY = (robotcoordsY) + (bottlecoordsX)

firstposX = 275.18
firstposY = -913.11

array = [firstposX, firstposY]

counter = 0

def main():
    global array
    # Camera matrix and distortion coefficients
    mtx = np.array([[893.38874436, 0, 652.46300526],
                    [0, 892.40326491, 360.40764759],
                    [0, 0, 1]])

    dist = np.array([0.20148339, -0.99826633, 0.00147814, 0.00218007, 1.33627184])

    known_width_mm = 329
    known_pixel_width = 1280

    # Calculate conversion factor from pixels to mm
    conversion_factor = known_width_mm / known_pixel_width

    # Initialize the video capture object
    cap = cv.VideoCapture(1)  # Change to 0 for the default camera
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        return -1

    roi_x, roi_y, roi_w, roi_h = 0, 260, 1280, 200  # Define the ROI coordinates

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        # Get the frame dimensions
        h, w = frame.shape[:2]

        # Undistort the frame
        new_camera_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        undistorted_frame = cv.undistort(frame, mtx, dist, None, new_camera_mtx)

        # Crop the undistorted frame to the ROI
        roi_frame = undistorted_frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        # Run YOLOv8 inference on the cropped ROI
        results = model(roi_frame, verbose=False)

        # Convert YOLOv8 results back into an OpenCV-friendly format for display
        annotated_roi_frame = results[0].plot()

        # Overlay the annotated ROI back onto the original annotated frame
        annotated_frame = undistorted_frame.copy()
        annotated_frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = annotated_roi_frame

        # Draw the ROI rectangle on the annotated frame
        cv.rectangle(annotated_frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)

        # Initialize variables to track the leftmost object
        leftmost_x = float('inf')
        leftmost_center = None

        # Iterate over the results and find the leftmost detected object
        if results and results[0].boxes:
            for box in results[0].boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extracting coordinates and converting to integers

                # Calculate the center point
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2 + 260

                # Check if this object is the leftmost one
                if center_x < leftmost_x:
                    leftmost_x = center_x
                    leftmost_center = (center_x, center_y)

                # Optionally draw the center point on the frame
                cv.circle(annotated_frame, (int(center_x), int(center_y)), 5, (255, 0, 0), -1)

        # Print the center point of the leftmost detected object
        if leftmost_center is not None:
            # print(f"Leftmost detected object center: {leftmost_center}")
            realX = center_x * conversion_factor
            realY = center_y * conversion_factor

            deltaY = (-1) * realX
            deltaX = (-1) * realY #i might make a mistake here

            pickupX = deltaX + offsetX + ((firstposX - array[0]) * (-1))
            pickupY = deltaY + offsetY + ((firstposY - array[1]) * (-1))

            print(pickupX,pickupY)
        #     msg = clientsocket.recv(1024)
            
        #     if not msg:  # If no message is received, break the loop
        #         break
        #     msg = (msg.decode("utf-8"))
        #     print(msg)

        #     if "trig" in msg:
        #         formatted_string = "({0}, {1})".format(pickupX, pickupY)
        #         message_to_send = formatted_string  # Coordinates to send
        #         clientsocket.send(bytes(message_to_send, "ascii"))
        #         print("Robot Pick-Up Coordinate:", pickupX, pickupY)                  

        #     elif "p" in msg:
        #         cleaned_msg = msg.replace("p", "")
        #         cleaned_msg = cleaned_msg.replace("trig", "")
        #         print("this is cleaned msg", cleaned_msg)
        #         array = ast.literal_eval(cleaned_msg)
        #         array[0] = array[0] * 1000
        #         array[1] = array[1] * 1000
        #         global counter 
        #         counter = 0

        #     else: 
        #         break

        # else:
        #     counter = counter + 1
        #     if counter > 25:
        #         print("send 69")
        #         clientsocket.send(bytes("(69)", "ascii")) 

        # Display the result for the detected circles
        cv.imshow("Detected Circle 1", annotated_frame)

        # Press 'q' to quit
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()