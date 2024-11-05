import cv2 as cv
import numpy as np
import torch
from ultralytics import YOLO
import socket
import ast
import time
camera = 0

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
bottlecoordsX1 = 165.8643946647644
bottlecoordsY1 = 151.17653875350953

robotcoordsX1 = 343.52
robotcoordsY1 = 803.89

offsetX1 = (robotcoordsX1) + (bottlecoordsX1)
offsetY1 = (robotcoordsY1) - (bottlecoordsY1)

firstposX1 = 346.82                                 #First position of the robot
firstposY1 = 607.10                                 #First position of the robot

array1 = [firstposX1, firstposY1]


# Part II variables
bottlecoordsX = 182.1722815513611                   #Bottle coordinates X via cameraview      
bottlecoordsY = 92.53994426727296                   #Bottle coordinates Y via cameraview

robotcoordsX = 342.77                              #Robot coordinates X, real world
robotcoordsY = -525.86                             #Robot coordinates Y, real world

offsetX = (robotcoordsX) + (bottlecoordsY)
offsetY = (robotcoordsY) + (bottlecoordsX)

firstposX = 485.64                                   #First position of the robot
firstposY = -502.46                                  #First position of the robot

array = [firstposX, firstposY]

counter = 0

# Define the Y-coordinate threshold
X_THRESHOLD = 450  # Adjust this value as needed to filter objects by x-coordinate
Y_THRESHOLD = 235  # Adjust this value as needed to filter objects by y-coordinate

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

    firstposX = 485.64                                   #First position of the robot
    firstposY = -502.46                                  #First position of the robot

    array = [firstposX, firstposY]

    array1	= [firstposX1, firstposY1]
    while True:
        
        msg = clientsocket.recv(1024)

        if not msg:
            break
        msg = (msg.decode("utf-8"))
        print(msg)

        if "trig" in msg:
            counterb = 0
            if array1[0] > 557:
                clientsocket.send(bytes("(69)", "ascii"))

            elif array1[1] < 276:                                   #certain y border   #-484 correct position    602 to test   demo:276
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

                    # Run YOLOv8 inference on the cropped ROI
                    results = model(undistorted_frame, verbose=False, conf=0.75)

                    # Initialize variables to track the leftmost object
                    bottommost_y = float('-inf')  # Initialize to infinity, to ensure any value of center_x will be smaller.
                    bottommost_center = None

                    # Iterate over the results and find the leftmost detected object
                    if results and results[0].obb:
                    # print("Detected objects:")  # Debugging info
                        for i, obb in enumerate(results[0].obb):
                            # Get the class label
                            label = results[0].names[int(obb.cls[0])]
                    
                            # Get the OBB coordinates (vertices)
                            vertices = obb.xyxyxyxy[0].cpu().numpy()  # Retrieve the vertices (4 points)

                            # Draw the OBB using the vertices for all detected objects
                            points = vertices.astype(int)
                            for j in range(len(points)):
                                cv.line(undistorted_frame, tuple(points[j]), tuple(points[(j + 1) % len(points)]), (0, 255, 0), 2)

                            # Draw the class label for all detected objects
                            cv.putText(undistorted_frame, label, (points[0][0], points[0][1] - 10), 
                                    cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                            # Process only the 'bottle_open' class for center point and specific handling
                            if label == 'bottle_open':
                                # Calculate the center point of the OBB
                                center_x = np.mean(vertices[:, 0])  # Average x-coordinates
                                center_y = np.mean(vertices[:, 1])  # Average y-coordinates
                                
                                if center_x > X_THRESHOLD and center_y > 250:
                                    # Draw a circle at the center point for 'bottle_open'
                                    cv.circle(undistorted_frame, (int(center_x), int(center_y)), 5, (255, 0, 0), -1)

                                    # Display the on-screen coordinates in the window
                                    on_screen_text = f"Coords: ({int(center_x)}, {int(center_y)})"
                                    cv.putText(undistorted_frame, on_screen_text, (int(center_x) + 10, int(center_y) - 10), 
                                            cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                                    # Check if this object is the leftmost one
                                    if center_y > bottommost_y:
                                        bottommost_y = center_y
                                        bottommost_center = (center_x, center_y)

                    # Output the leftmost object's center
                    if bottommost_center:
                        print(f"Leftmost object center: {bottommost_center}")
                        # time.sleep(1)

                        # print(f"Leftmost detected object center: {leftmost_center}")
                        realX = bottommost_center[0] * conversion_factor
                        realY = bottommost_center[1] * conversion_factor

                        deltaX = realX
                        deltaY = realY #i might make a mistake here

                        pickupX = -deltaX + offsetX1 + (array1[0] - firstposX1)
                        pickupY = deltaY + offsetY1 + (array1[1] - firstposY1)
                        
                        #print(f"array: {array1[0]},{array1[1]}")
                        # print(f"robot coords:{realX},{realY}")
                        
                        # print(f"mm coords:{realX},{realY}")
                        # print(f"robot coords:{offsetX1},{offsetY1}")
                        # print(f"robot coords:{pickupX},{pickupY}")
                        # print(f"array: {array1[0]},{array1[1]}")
                        # print(f"firstpos: {firstposX1},{firstposY1}")

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
        cv.imshow("Detected Circle 1", undistorted_frame) #show the cropped imgge

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
    
    rij = 0
    
    while True:
        msg = clientsocket.recv(1024)
        
        if not msg:  # If no message is received, break the loop
            break
        msg = (msg.decode("utf-8"))
        print(msg)

        if "trig" in msg:
            
            if array[1] > 580:
                print("Going to next row")
                clientsocket.send(bytes("(69)", "ascii"))
                rij = rij + 1
                print("rij:", rij)

            elif rij > 7 and array[1] > 540:                                      #rij moet 7 zijn 2 voor testen
                print("End of code")
                clientsocket.send(bytes("(33)", "ascii"))
                break
            else:
                while True:
                    
                    #time.sleep(1)
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

                    # Run YOLOv8 OBB inference on the cropped ROI frame
                    results = model(undistorted_frame, verbose=False, conf=0.90)  # Lower confidence threshold

                    # Initialize variables to track the leftmost object
                    rightmost_x = float('-inf')  # Initialize to infinity, to ensure any value of center_x will be smaller.
                    rightmost_center = None


                    # Iterate over the results and find the leftmost detected object
                    if results and results[0].obb:
                        # print("Detected objects:")  # Debugging info
                        for i, obb in enumerate(results[0].obb):
                            # Get the class label
                            label = results[0].names[int(obb.cls[0])]

                            # Get the OBB coordinates (vertices)
                            vertices = obb.xyxyxyxy[0].cpu().numpy()  # Retrieve the vertices (4 points)

                            # Draw the OBB using the vertices for all detected objects
                            points = vertices.astype(int)
                            for j in range(len(points)):
                                cv.line(undistorted_frame, tuple(points[j]), tuple(points[(j + 1) % len(points)]), (0, 255, 0), 2)

                            # Draw the class label for all detected objects
                            cv.putText(undistorted_frame, label, (points[0][0], points[0][1] - 10), 
                                    cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                            # Process only the 'bottle_open' class for center point and specific handling
                            if label == 'bottle_open':
                                # Calculate the center point of the OBB
                                center_x = np.mean(vertices[:, 0])  # Average x-coordinates
                                center_y = np.mean(vertices[:, 1])  # Average y-coordinates

                                if center_y > Y_THRESHOLD and center_x > 350:
                                    # Draw a circle at the center point for 'bottle_open'
                                    cv.circle(undistorted_frame, (int(center_x), int(center_y)), 5, (255, 0, 0), -1)

                                    # Display the on-screen coordinates in the window
                                    on_screen_text = f"Coords: ({int(center_x)}, {int(center_y)})"
                                    cv.putText(undistorted_frame, on_screen_text, (int(center_x) + 10, int(center_y) - 10), 
                                            cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                                    # Check if this object is the leftmost one
                                    if center_x > rightmost_x:
                                        rightmost_x = center_x
                                        rightmost_center = (center_x, center_y)
                                        print(f"Rightmost object center: {rightmost_center}")

                    # Output the leftmost object's center and calculate real-world coordinates
                    if rightmost_center:
                        realX = rightmost_center[0] * conversion_factor
                        realY = rightmost_center[1] * conversion_factor

                        deltaY = (-1) * realX
                        deltaX = (-1) * realY  # Reversing the coordinates here

                        pickupX = deltaX + offsetX + ((firstposX - array[0]) * (-1))
                        pickupY = deltaY + offsetY + ((firstposY - array[1]) * (-1))
                        print(f"mm coords:{realX},{realY}")


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
                            array[1] = array[1] + 75                          #was 170.5
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
        cv.imshow("Detected Circle 1", undistorted_frame) #show the cropped imgge


        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()