import sys
import cv2 as cv
import numpy as np
import socket
import time

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

pickupX = 0
pickupY = 0

robotXRef = 228.60
robotYRef = -482.63
# Flip the bottleXRef
bottleXRef = 186.71
bottleYRef = 171.15

def empty(val):
    pass

def create_trackbars():
    # Create a window named 'TrackBars' to hold the trackbars
    cv.namedWindow("TrackBars")
    cv.resizeWindow("TrackBars", 640, 240)

    # Create trackbars for adjusting HSV filter values
    cv.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
    cv.createTrackbar("Hue Max", "TrackBars", 179, 179, empty)
    cv.createTrackbar("Sat Min", "TrackBars", 10, 255, empty)
    cv.createTrackbar("Sat Max", "TrackBars", 98, 255, empty)
    cv.createTrackbar("Val Min", "TrackBars", 0, 255, empty)
    cv.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

def get_trackbar_values():
    # Read the current values of the trackbars
    h_min = cv.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv.getTrackbarPos("Val Max", "TrackBars")
    return h_min, h_max, s_min, s_max, v_min, v_max

def main():
    # Camera matrix and distortion coefficients
    mtx = np.array([[663.81055373, 0, 320.9241384],
                [0, 663.2383916, 241.48871247],
                [0, 0, 1]])

    dist = np.array([2.75825516e-01, -1.88952738e+00, 2.27429839e-03,
                 -2.77074731e-03, 4.02307100e+00])

    known_width_mm = 383 
    known_pixel_width = 640 

    # Calculate conversion factor from pixels to mm
    conversion_factor = known_width_mm / known_pixel_width

    # Initialize the video capture object
    cap = cv.VideoCapture(0)  # Change to 0 for the default camera
    
    if not cap.isOpened():
        print("Cannot open camera")
        return -1

    # Create the HSV trackbars
    create_trackbars()

    while True:
        start_time = time.time()

        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Cannot receive frame")
            break

        # Get the frame dimensions
        h, w = frame.shape[:2]

        # Undistort the frame
        new_camera_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        undistorted_frame = cv.undistort(frame, mtx, dist, None, new_camera_mtx)

        # Get current HSV filter values from the trackbars
        h_min, h_max, s_min, s_max, v_min, v_max = get_trackbar_values()

        # Convert the frame to HSV color space
        imgHSV = cv.cvtColor(undistorted_frame, cv.COLOR_BGR2HSV)

        # Create lower and upper bounds for the mask
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        
        # Create a mask based on the HSV bounds
        mask = cv.inRange(imgHSV, lower, upper)
        
        # Apply the mask to the undistorted frame
        imgResult = cv.bitwise_and(undistorted_frame, undistorted_frame, mask=mask)

        # ** Now applying the circle detection on the mask result, not the undistorted frame **
        gray = cv.cvtColor(imgResult, cv.COLOR_BGR2GRAY)
        gray = cv.medianBlur(gray, 5)

        # Detect circles
        rows = gray.shape[0]
        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                                   param1=20, param2=20,
                                   minRadius=25, maxRadius=33)

        # Draw the circles on the mask result and label them
        if circles is not None:
            circles = np.uint16(np.around(circles))

            # Sort circles based on their y-coordinates
            circles = sorted(circles[0, :], key=lambda x: x[1])

            for index, i in enumerate(circles):
                center_pixels = (i[0], i[1])
                
                # Convert pixel coordinates to mm
                center_mm = (center_pixels[0] * conversion_factor, center_pixels[1] * conversion_factor)

                # Circle center
                cv.circle(imgResult, center_pixels, 1, (0, 100, 100), 3)
                # Circle outline
                radius = i[2]
                cv.circle(imgResult, center_pixels, radius, (255, 0, 255), 3)

                # Put text with circle number and center in mm
                text = f"{index + 1}: ({center_mm[0]:.2f}, {center_mm[1]:.2f}) mm"
                cv.putText(imgResult, text, (center_pixels[0] - 30, center_pixels[1]), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                deltaX = bottleXRef - center_mm[0]
                deltaY = bottleYRef - center_mm[1]
                
                pickupX = robotXRef + deltaY
                pickupY = robotYRef + deltaX

                formatted_string = "({0}, {1})".format(pickupX, pickupY)
                message_to_send = formatted_string  # Coordinates to send
                clientsocket.send(bytes(message_to_send, "ascii"))
                
                print(f"Circle {index + 1} - Coordinate: ({center_mm[0]:.2f}, {center_mm[1]:.2f}), Robot Pick-Up Coordinate: ({pickupX:.2f}, {pickupY:.2f})")               

        # Display the undistorted frame and mask result
        cv.imshow("Detected Circles on Mask", imgResult)
        cv.imshow("Original Undistorted Frame", undistorted_frame)

        # Press 'q' to quit
        if cv.waitKey(1) == ord('q'):
            break

        elapsed_time = time.time() - start_time
        time.sleep(max(0, 0.2 - elapsed_time))

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
