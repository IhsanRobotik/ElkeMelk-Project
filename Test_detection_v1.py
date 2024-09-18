import cv2
import numpy as np

# Load the image (replace 'image.jpg' with the path to your image)
image = cv2.imread("C:/Users/basti/Downloads/20240916_105228.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a Gaussian blur to reduce noise and improve circle detection
blurred = cv2.GaussianBlur(gray, (9, 9), 2)

# Apply Hough Circle Transform to detect circles (bottles)
circles = cv2.HoughCircles(blurred, 
                           cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                           param1=71, param2=35, minRadius=70, maxRadius=90)

# If some circles are detected
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")

    # Loop over the detected circles
    for (x, y, r) in circles:
        # Draw the circle in the output image
        cv2.circle(image, (x, y), r, (0, 255, 0), 4)
        # Draw a rectangle at the center of the circle
        cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

# Create a named window and resize it to a smaller size
cv2.namedWindow("Bottles Detected", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Bottles Detected", 800, 600)  # Resize window to 800x600

# Show the output image with detected circles
cv2.imshow("Bottles Detected", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

