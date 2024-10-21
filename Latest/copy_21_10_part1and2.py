import cv2 as cv
import numpy as np
import torch
from ultralytics import YOLO
import socket
import ast
import time
camera = 1

# Ensure CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    print("CUDA is available and enabled.")

# Load the pre-trained YOLOv8 OBB model
model = YOLO(r"C:/Users/basti/Documents/GitHub/ElkeMelk-Project/models/obbV5.pt")  # Update with your OBB model path

# Set the model to use the GPU
model.to(device)

# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the address and port
s.bind(("192.168.0.45", 5005))
print("listening for connection")
# Listen for incoming connections
s.listen(5)
# Accept a connection from a client
clientsocket, address = s.accept()
print(f"Connection from {address} has been established!")

# Part I variables
bottlecoordsX1 = 171.565625
bottlecoordsY1 = 114.021875

robotcoordsX1 = 320.00
robotcoordsY1 = 770.29

offsetX1 = (robotcoordsX1) + (bottlecoordsX1)
offsetY1 = (robotcoordsY1) - (bottlecoordsY1)

firstposX1 = 317.68                                 #First position of the robot
firstposY1 = 607.10                                 #First position of the robot

array1 = [firstposX1, firstposY1]


# Part II variables
bottlecoordsX = 149.59208498001098                       #Bottle coordinates X via cameraview
bottlecoordsY = 90.73866143226624                         #Bottle coordinates Y via cameraview

robotcoordsX = 335.18                              #Robot coordinates X, real world
robotcoordsY = -489.44                              #Robot coordinates Y, real world

offsetX = (robotcoordsX) + (bottlecoordsY)
offsetY = (robotcoordsY) + (bottlecoordsX)

firstposX = 485.64                                   #First position of the robot
firstposY = -502.46                                  #First position of the robot

array = [firstposX, firstposY]

counter = 0

def main():
    global array
    mtx = np.array([[893.38874436, 0, 652.46300526],
                    [0, 892.40326491, 360.40764759],
                    [0, 0, 1]])
    dist = np.array([0.20148339, -0.99826633, 0.00147814, 0.00218007, 1.33627184])
    known_width_mm = 341
    known_pixel_width = 1280

    # Calculate conversion factor from pixels to mm
    conversion_factor = known_width_mm / known_pixel_width

    # Initialize the video capture object
    cap = cv.VideoCapture(camera)  # Change to 0 for the default camera
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Camera not accessible.")
        return -1
    roi_x, roi_y, roi_w, roi_h = 515, 0, 250, 720

    firstposX = 485.64                                   #First position of the robot
    firstposY = -502.46                                  #First position of the robot

    array = [firstposX, firstposY]

    while True:
        msg = clientsocket.recv(1024)

        if not msg:
            break
        msg = (msg.decode("utf-8"))
        print(msg)

        if "trig" in msg:
            counterb = 0
            if array1[0] > 557:
                print("Going to next row.")
                clientsocket.send(bytes("(69)", "ascii"))

            elif array1[1] < 602:                                   #certain y border   #-484 correct position
                print("Going to part II")
                clientsocket.send(bytes("(25)", "ascii"))
                cap.release()
                cv.destroyAllWindows()
                break
            else:
                while True:	
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
                    results = model(roi_frame, verbose=False, conf=0.75)

                    # Convert YOLOv8 results back into an OpenCV-friendly format for display
                    annotated_roi_frame = results[0].plot()

                    # Overlay the annotated ROI back onto the original annotated frame
                    annotated_frame = undistorted_frame.copy()
                    annotated_frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = annotated_roi_frame

                    # Draw the ROI rectangle on the annotated frame
                    cv.rectangle(annotated_frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)

                    # Initialize variables to track the leftmost object
                    leftmost_x = float('-inf')  # Initialize to infinity, to ensure any value of center_x will be smaller.
                    leftmost_center = None

                    detected_classes = []  # To log detected classes

                    # Iterate over the results and find the leftmost detected object
                    if results and results[0].obb:
                    # print("Detected objects:")  # Debugging info
                        for i, obb in enumerate(results[0].obb):
                        # Get the class label
                            label = results[0].names[int(obb.cls[0])]
                            detected_classes.append(label)  # Log detected class labels
                            # print(f"Object {i}: {label}")  # Debug: Print object label

                            # Get the OBB coordinates (vertices)
                            vertices = obb.xyxyxyxy[0].cpu().numpy()  # Retrieve the vertices (4 points)

                            # Adjust the coordinates to the original frame by adding the ROI offset
                            vertices[:, 0] += roi_x  # Adjust x-coordinates
                            vertices[:, 1] += roi_y  # Adjust y-coordinates

                            # Draw the OBB using the vertices for all detected objects
                            points = vertices.astype(int)
                            for j in range(len(points)):
                                cv.line(annotated_frame, tuple(points[j]), tuple(points[(j + 1) % len(points)]), (0, 255, 0), 2)

                            # Draw the class label for all detected objects
                            cv.putText(annotated_frame, label, (points[0][0], points[0][1] - 10), 
                                    cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                            # Process only the 'bottle_open' class for center point and specific handling
                            if label == 'bottle_open':
                                # Calculate the center point of the OBB
                                center_x = np.mean(vertices[:, 0])  # Average x-coordinates
                                center_y = np.mean(vertices[:, 1])  # Average y-coordinates

                                # Draw a circle at the center point for 'bottle_open'
                                cv.circle(annotated_frame, (int(center_x), int(center_y)), 5, (255, 0, 0), -1)

                                # Display the on-screen coordinates in the window
                                on_screen_text = f"Coords: ({int(center_x)}, {int(center_y)})"
                                cv.putText(annotated_frame, on_screen_text, (int(center_x) + 10, int(center_y) - 10), 
                                        cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                                # Check if this object is the leftmost one
                                if center_y > leftmost_x:
                                    leftmost_x = center_y
                                    leftmost_center = (center_x, center_y)

                    # Output the leftmost object's center
                    if leftmost_center:
                        print(f"Leftmost object center: {leftmost_center}")
                        # time.sleep(1)

                        # print(f"Leftmost detected object center: {leftmost_center}")
                        realX = leftmost_center[0] * conversion_factor
                        realY = leftmost_center[1] * conversion_factor

                        deltaX = realX
                        deltaY = realY #i might make a mistake here

                        pickupX = deltaX + offsetX1 + (array1[0] - firstposX1)
                        pickupY = deltaY + offsetY1 + (array1[1] - firstposY1)
                        
                        #print(f"array: {array1[0]},{array1[1]}")
                        print(f"robot coords:{realX},{realY}")

                        formatted_string = "({0}, {1})".format(pickupX, pickupY)
                        message_to_send = formatted_string  # Coordinates to send
                        clientsocket.send(bytes(message_to_send, "ascii"))
                        print("Robot Pick-Up Coordinate:", pickupX, pickupY)
                        counterb = 0
                        break

                    # else:
                    #     # print("No 'bottle_open' object detected in the current frame.")
                    #     print(f"Detected classes: {detected_classes}")  # Print all detected classes

                    else:
                        counterb = counterb + 1
                        print(f"{counterb}")
                        if counterb > 4:
                            print("Going to move up a little because no bottles")
                            clientsocket.send(bytes("(55)", "ascii"))
                            break

        elif "p" in msg:
            cleaned_msg = msg.replace("p", "")
            cleaned_msg = cleaned_msg.replace("trig", "")
            print("this is cleaned msg", cleaned_msg)
            array = ast.literal_eval(cleaned_msg)
            array1[0] = array[0] * 1000
            array1[1] = array[1] * 1000
        
        else: 
            break

        # Display the result for the detected circles
        x,y,w,h = roi
        annotated1_frame = annotated_frame[y:y+h, x:x+w] #crop to roi
        cv.imshow("Detected Circle 1", annotated1_frame) #show the cropped imgge

        if cv.waitKey(1) == ord('w'):
            break

    cap.release()
    cv.destroyAllWindows()


    firstposX = 485.64                                   #First position of the robot
    firstposY = -502.46                                  #First position of the robot

    array = [firstposX, firstposY]

    # Initialize the video capture object
    cap = cv.VideoCapture(camera)  # Change to 0 for the default camera
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        return -1
    roi_x, roi_y, roi_w, roi_h = 0, 240, 1280, 210  # Define the ROI coordinates

    while True:
        msg = clientsocket.recv(1024)
        
        if not msg:  # If no message is received, break the loop
            break
        msg = (msg.decode("utf-8"))
        print(msg)

        if "trig" in msg:
            
            if array[1] > 620:
                print("Going to next row")
                clientsocket.send(bytes("(69)", "ascii"))
            else:
                while True:
                    # time.sleep(1)
                    # Capture frame-by-frame
                    ret, frame = cap.read()

                    if not ret:
                        print("Failed to capture frame.")
                        break

                    # Get the frame dimensions
                    h, w = frame.shape[:2]

                    # Undistort the frame
                    new_camera_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), -1, (w, h))
                    undistorted_frame = cv.undistort(frame, mtx, dist, None, new_camera_mtx)

                    # Crop the undistorted frame to the ROI
                    roi_frame = undistorted_frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

                    # Run YOLOv8 OBB inference on the cropped ROI frame
                    results = model(roi_frame, verbose=False, conf=0.45)  # Lower confidence threshold

                    # Copy ROI frame for annotations
                    annotated_frame = undistorted_frame.copy()

                    # Draw the ROI rectangle on the annotated frame
                    cv.rectangle(annotated_frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)

                    # Initialize variables to track the leftmost object
                    leftmost_x = float('-inf')  # Initialize to infinity, to ensure any value of center_x will be smaller.
                    leftmost_center = None

                    detected_classes = []  # To log detected classes

                    # Iterate over the results and find the leftmost detected object
                    if results and results[0].obb:
                        # print("Detected objects:")  # Debugging info
                        for i, obb in enumerate(results[0].obb):
                            # Get the class label
                            label = results[0].names[int(obb.cls[0])]
                            detected_classes.append(label)  # Log detected class labels
                            # print(f"Object {i}: {label}")  # Debug: Print object label

                            # Get the OBB coordinates (vertices)
                            vertices = obb.xyxyxyxy[0].cpu().numpy()  # Retrieve the vertices (4 points)

                            # Adjust the coordinates to the original frame by adding the ROI offset
                            vertices[:, 0] += roi_x  # Adjust x-coordinates
                            vertices[:, 1] += roi_y  # Adjust y-coordinates

                            # Draw the OBB using the vertices for all detected objects
                            points = vertices.astype(int)
                            for j in range(len(points)):
                                cv.line(annotated_frame, tuple(points[j]), tuple(points[(j + 1) % len(points)]), (0, 255, 0), 2)

                            # Draw the class label for all detected objects
                            cv.putText(annotated_frame, label, (points[0][0], points[0][1] - 10), 
                                    cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                            # Process only the 'bottle_open' class for center point and specific handling
                            if label == 'bottle_open':
                                # Calculate the center point of the OBB
                                center_x = np.mean(vertices[:, 0])  # Average x-coordinates
                                center_y = np.mean(vertices[:, 1])  # Average y-coordinates

                                # Draw a circle at the center point for 'bottle_open'
                                cv.circle(annotated_frame, (int(center_x), int(center_y)), 5, (255, 0, 0), -1)

                                # Display the on-screen coordinates in the window
                                on_screen_text = f"Coords: ({int(center_x)}, {int(center_y)})"
                                cv.putText(annotated_frame, on_screen_text, (int(center_x) + 10, int(center_y) - 10), 
                                        cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                                print(f"Leftmost object center: ({center_x}, {center_y})")

                                # Check if this object is the leftmost one
                                if center_x > leftmost_x:
                                    leftmost_x = center_x
                                    leftmost_center = (center_x, center_y)

                    # Output the leftmost object's center and calculate real-world coordinates
                    if leftmost_center:
                        realX = leftmost_center[0] * conversion_factor
                        realY = leftmost_center[1] * conversion_factor

                        deltaY = (-1) * realX
                        deltaX = (-1) * realY  # Reversing the coordinates here

                        pickupX = deltaX + offsetX + ((firstposX - array[0]) * (-1))
                        pickupY = deltaY + offsetY + ((firstposY - array[1]) * (-1))

                        # print(f"mm coords:{realX},{realY}")
                        # print(f"robot coords:{offsetX},{offsetY}")
                        # print(f"robot coords:{pickupX},{pickupY}")
                        # print(f"array: {array[0]},{array[1]}")
                        # print(f"firstpos: {firstposX},{firstposY}")

                        formatted_string = "({0}, {1})".format(pickupX, pickupY)
                        message_to_send = formatted_string  # Coordinates to send
                        clientsocket.send(bytes(message_to_send, "ascii"))
                        print("Robot Pick-Up Coordinate:", pickupX, pickupY)
                        
                        global counter 
                        counter = 0  
                        break 

                    else:
                        counter = counter + 1
                        print(f"counter: {counter}")
                        if counter > 5:
                            print("Move up a little because no bottles")
                            clientsocket.send(bytes("(55)", "ascii"))
                            array[1] = array[1] + 170.5
                            break
                          
        elif "p" in msg:
            cleaned_msg = msg.replace("p", "")
            cleaned_msg = cleaned_msg.replace("trig", "")
            print("this is cleaned msg", cleaned_msg)
            array = ast.literal_eval(cleaned_msg)
            array[0] = array[0] * 1000
            array[1] = array[1] * 1000

        else: 
            break 
        
        # Display the result for the detected circles
        x,y,w,h = roi
        annotated_frame = annotated_frame[y:y+h, x:x+w] #crop to roi
        cv.imshow("Detected Circle 1", annotated_frame) #show the cropped imgge


        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()