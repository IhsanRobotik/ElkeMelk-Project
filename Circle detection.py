import sys
import cv2 as cv
import numpy as np

def main(argv):
    default_file = 'sample.jpg'
    filename = argv[0] if len(argv) > 0 else default_file

    # Load the images
    
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)
    src = cv.resize(src, (640, 480))
    # Check if image is loaded fine
    if src is None:
        print('Error opening image!')
        print('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
        return -1

    # Convert to gray scale
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)

    # Detect circles
    rows = gray.shape[0]
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                               param1=95, param2=20,
                               minRadius=10, maxRadius=50)

    # Draw the circles
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for index, i in enumerate(circles[0, :]):
            center = (i[0], i[1])
            # Circle center
            cv.circle(src, center, 1, (0, 100, 100), 3)
            # Circle outline
            radius = i[2]
            cv.circle(src, center, radius, (255, 0, 255), 3)
            # Put text with circle number
            text = f"bottle {index + 1}"
            cv.putText(src, text, (center[0] - 30, center[1]), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the resulting image
    cv.imshow("Detected Circles", src)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return 0

if __name__ == "__main__":
    main(sys.argv[1:])