import cv2 as cv
import numpy as np
import socket
import time
import ast

# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the address and port
s.bind(("192.168.0.45", 5005))

# Listen for incoming connections
s.listen(5)
print("Server is listening for connections...")

# Accept a connection from a client
clientsocket, address = s.accept()
print(f"Connection from {address} has been established!")

# Set a timeout for receiving data to prevent blocking
clientsocket.settimeout(5.0)

pickupX = 0
pickupY = 0

bottleXRef = 212.31
bottleYRef = 222.36

robotXRef = 185.29
robotYRef = -500.52

# Camera matrix and distortion coefficients
mtx = np.array([[663.81055373, 0, 320.9241384],
                 [0, 663.2383916, 241.48871247],
                 [0, 0, 1]])

dist = np.array([2.75825516e-01, -1.88952738e+00, 2.27429839e-03,
                 -2.77074731e-03, 4.02307100e+00])

known_width_mm = 402 
known_pixel_width = 640 

# Calculate conversion factor from pixels to mm
conversion_factor = known_width_mm / known_pixel_width

def main():
    # Initialize video capture
    cap = cv.VideoCapture(0)  # Change to 0 for the default camera
    
    if not cap.isOpened():
        print("Cannot open camera")
        return -1

    print("Camera opened successfully. Entering main loop...")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Cannot receive frame")
            break

        h, w = frame.shape[:2]

        # Undistort the frame
        new_camera_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        undistorted_frame = cv.undistort(frame, mtx, dist, None, new_camera_mtx)

        # Convert to grayscale
        gray = cv.cvtColor(undistorted_frame, cv.COLOR_BGR2GRAY)
        gray = cv.medianBlur(gray, 5)

        # Detect circles (bottles)
        rows = gray.shape[0]
        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                                   param1=40, param2=25,
                                   minRadius=25, maxRadius=33)

        if circles is not None:
            circles = np.uint16(np.around(circles))

            # Sort circles by their y-coordinates
            circles = sorted(circles[0, :], key=lambda x: x[1])

            for index, i in enumerate(circles):
                center_pixels = (i[0], i[1])

                # Convert pixel coordinates to mm for the bottle
                center_mm = (center_pixels[0] * conversion_factor, center_pixels[1] * conversion_factor)

                bottleXRef = 212.31
                bottleYRef = 222.36

                robotXRef = 185.29
                robotYRef = -500.52

                # Calculate deltas between the bottle and the gripper position
                deltaX = center_mm[0] - bottleXRef
                deltaY = center_mm[1] - bottleYRef

                pickupX = robotXRef - deltaY
                pickupY = robotYRef - deltaX
                        
                # Send the new coordinates to the robot
                formatted_string = "({0}, {1})".format(pickupX, pickupY)
                clientsocket.send(bytes(formatted_string, "ascii"))
                print(f"Sent coordinates to robot: {formatted_string}")

                # Wait for the message from the robot that it is at the correct position
                robot_at_position = False
                while not robot_at_position:
                    try:
                        robot_msg = clientsocket.recv(1024)
                        decoded_msg = robot_msg.decode("utf-8")
                        print(f"Received from robot: {decoded_msg}")  # Debugging statement
                        
                        if decoded_msg == "I am there":
                            # Robot has reached the position, save the new coordinates
                            robotXRef = pickupX
                            robotYRef = pickupY
                            bottleXRef = center_mm[0]
                            bottleYRef = center_mm[1]

                            print("Coordinates saved successfully!")

                            # Send a message back to the robot
                            clientsocket.send("saved it".encode("utf-8"))
                            print("Sent confirmation to the robot: 'saved it'. Robot can proceed.")
                            robot_at_position = True
                    except socket.timeout:
                        print("Waiting for response...")
                    except Exception as e:
                        print(f"Error occurred: {e}")
                        break

                # Draw the circle and place text
                cv.circle(undistorted_frame, center_pixels, 1, (0, 100, 100), 3)
                radius = i[2]
                cv.circle(undistorted_frame, center_pixels, radius, (255, 0, 255), 3)

                text = f"{index + 1}: ({center_mm[0]:.2f}, {center_mm[1]:.2f}) mm"
                cv.putText(undistorted_frame, text, (center_pixels[0] - 30, center_pixels[1]), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Show the undistorted frame
        cv.imshow("Detected Circles", undistorted_frame)

        # Press 'q' to stop
        if cv.waitKey(1) == ord('q'):
            break

    # End the capture and close all windows
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
