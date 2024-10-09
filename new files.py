import cv2 as cv
import numpy as np
import socket
import ast
import cv2
import time

# Create a socket object
# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# # Bind the socket to the address and port
# s.bind(("192.168.0.45", 5005))

# # Listen for incoming connections
# s.listen(5)
# # Accept a connection from a client
# clientsocket, address = s.accept()
# print(f"Connection from {address} has been established!")

bottlecoordsX = 295.70
bottlecoordsY = 170.70

robotcoordsX = 247.25
robotcoordsY = -487.20

offsetX = (robotcoordsX) + (bottlecoordsY)
offsetY = (robotcoordsY) + (bottlecoordsX)

firstposX = 111.43
firstposY = -500.54

array = [111.43, -500.54]

def empty(val):
    pass

def create_trackbars():
    # Create a window named 'TrackBars' to hold the trackbars
    cv.namedWindow("TrackBars")
    cv.resizeWindow("TrackBars", 640, 240)

    # Create trackbars for adjusting HSV filter values
    cv2.createTrackbar("Hue Min", "TrackBars", 29, 179, empty)
    cv2.createTrackbar("Hue Max", "TrackBars", 142, 179, empty)
    cv2.createTrackbar("Sat Min", "TrackBars", 8, 255, empty)
    cv2.createTrackbar("Sat Max", "TrackBars", 36, 255, empty)
    cv2.createTrackbar("Val Min", "TrackBars", 0, 255, empty)
    cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

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
    global array
    # Camera matrix and distortion coefficients
    mtx = np.array([[893.38874436, 0, 652.46300526],
                [0, 892.40326491, 360.40764759],
                [0, 0, 1]])

    dist = np.array([ 0.20148339, -0.99826633,  0.00147814,  0.00218007,  1.33627184])

    known_width_mm = 597 
    known_pixel_width = 1280 

    # Calculate conversion factor from pixels to mm
    conversion_factor = known_width_mm / known_pixel_width

    # Initialize the video capture object
    cap = cv.VideoCapture(0)  # Change to 0 for the default camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        return -1

    # Create the HSV trackbars
    create_trackbars()

    while True:
        time.sleep(1)

        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        # Get the frame dimensions
        h, w = frame.shape[:2]

        # Undistort the frame
        new_camera_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        undistorted_frame = cv.undistort(frame, mtx, dist, None, new_camera_mtx)

        # x,y,w,h = roi
        # undistorted_frame = undistorted_frame[y:y+h, x:x+w]

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
                                   param1=15, param2=15,
                                   minRadius=35, maxRadius=40)

        # Draw the circles on the mask result and label them
        if circles is not None:
            circles = np.uint16(np.around(circles))

            # Sort circles based on their y-coordinates
            circles = sorted(circles[0, :], key=lambda x: x[1])

            sorted_circles = sorted(circles, key=lambda i: i[0])

            for index, i in enumerate(sorted_circles):
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
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                

                deltaY = (-1) * center_mm[0]
                deltaX = (-1) * center_mm[1]

                pickupX =  deltaX + offsetX + ((firstposX - array[0]) * (-1))
                pickupY =  deltaY + offsetY + ((firstposY - array[1]) * (-1))

                # msg = clientsocket.recv(1024)
                
                # if not msg:  # If no message is received, break the loop
                #     break
                # msg = (msg.decode("utf-8"))
                # print(msg)

                # if "trig" in msg:
                #     formatted_string = "({0}, {1})".format(pickupX, pickupY)
                #     message_to_send = formatted_string  # Coordinates to send
                #     clientsocket.send(bytes(message_to_send, "ascii"))
                #     print(f"Circle {index + 1} - Coordinate: ({center_mm[0]:.2f}, {center_mm[1]:.2f}), Robot Pick-Up Coordinate: ({pickupX:.2f}, {pickupY:.2f})")                  

                # elif "p" in msg:
                #     cleaned_msg = msg.replace("p", "")
                #     cleaned_msg = cleaned_msg.replace("trig", "")
                #     print("this is cleaned msg", cleaned_msg)
                #     array = ast.literal_eval(cleaned_msg)
                #     array[0] = array[0] * 1000
                #     array[1] = array[1] * 1000

                # else: 
                #     break

        # Display the undistorted frame and mask result
        cv.imshow("Detected Circles on Mask", imgResult)
        cv.imshow("ass", undistorted_frame)

        # Press 'q' to quit
        if cv.waitKey(1) == ord('q'):
            break


    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()