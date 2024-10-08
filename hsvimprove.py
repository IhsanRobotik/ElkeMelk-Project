import cv2
import numpy as np

def empty(a):
    pass

def getContours(img):
    """ Function to find and draw contours on an image. """
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    imgContour = np.zeros_like(frame)  # Create an empty image to draw contours

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:  # Filter out small contours based on area
            cv2.drawContours(imgContour, cnt, -1, (0, 255, 0), 2)  # Draw contours in green

    return imgContour

# Create a named window for trackbars
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 240)

# Create trackbars for HSV value adjustments
cv2.createTrackbar("Hue Min", "TrackBars", 60, 179, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 112, 179, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 7, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 36, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 25, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 155, 255, empty)

# Camera matrix and distortion coefficients
mtx = np.array([[893.38874436, 0, 652.46300526],
                [0, 892.40326491, 360.40764759],
                [0, 0, 1]])

dist = np.array([ 0.20148339, -0.99826633,  0.00147814,  0.00218007,  1.33627184])

# Initialize the video capture object
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

known_width_mm = 597  
known_pixel_width = 1280 

# Calculate conversion factor from pixels to mm
conversion_factor = known_width_mm / known_pixel_width

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Cannot receive frame")
        break

    # Get the frame dimensions
    h, w = frame.shape[:2]

    # Undistort the frame
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted_frame = cv2.undistort(frame, mtx, dist, None, new_camera_mtx)

    # Convert frame to HSV color space
    imgHSV = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2HSV)

    # Get the current positions of the trackbars
    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")

    # Create lower and upper bounds for the mask
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    
    # Create a mask based on the HSV bounds
    mask = cv2.inRange(imgHSV, lower, upper)
    
    # Apply the mask to the undistorted frame
    imgResult = cv2.bitwise_and(undistorted_frame, undistorted_frame, mask=mask)

    # Convert the masked result to grayscale for circle detection
    gray = cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY)

    # Detect circles
    rows = imgResult.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                                 param1=25, param2=20,
                                 minRadius=35, maxRadius=40)

    # Draw the circles on the masked result
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for index, i in enumerate(circles[0, :]):
            center_pixels = (i[0], i[1])
            
            # Convert pixel coordinates to mm
            center_mm = (center_pixels[0] * conversion_factor, center_pixels[1] * conversion_factor)

            # Circle center
            cv2.circle(imgResult, center_pixels, 1, (0, 100, 100), 3)
            # Circle outline
            radius = i[2]
            cv2.circle(imgResult, center_pixels, radius, (255, 0, 255), 3)

            # Put text with circle number and center in mm
            text = f"bottle {index + 1}: ({center_mm[0]:.2f}, {center_mm[1]:.2f}) mm"
            cv2.putText(imgResult, text, (center_pixels[0] - 30, center_pixels[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Display the original and processed frames
    cv2.imshow("Original Video", frame)
    cv2.imshow("HSV Filtered Result", imgResult)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()