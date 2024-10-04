import sys
import cv2 as cv
import numpy as np
import pyrealsense2 as rs

def main():
    # Camera matrix and distortion coefficients
    mtx = np.array([[680.43205516, 0, 325.70663487],
                    [0, 680.82113726, 224.79035512],
                    [0, 0, 1]])  

    dist = np.array([2.86943840e-01, -1.81250009e+00, -9.85433600e-03,
                     2.34191361e-03, 4.19774044e+00]) 

    known_width_mm = 387  
    known_pixel_width = 640 

    # Calculate conversion factor from pixels to mm
    conversion_factor = known_width_mm / known_pixel_width

    # Initialize the Intel RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert the frame to a numpy array
            frame = np.asanyarray(color_frame.get_data())

            # Get the frame dimensions
            h, w = frame.shape[:2]

            # Undistort the frame
            new_camera_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
            undistorted_frame = cv.undistort(frame, mtx, dist, None, new_camera_mtx)

            # Convert to grayscale
            gray = cv.cvtColor(undistorted_frame, cv.COLOR_BGR2GRAY)
            gray = cv.medianBlur(gray, 5)

            # Detect circles
            rows = gray.shape[0]
            circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                                       param1=20, param2=30,
                                       minRadius=30, maxRadius=40)

            # Draw the circles on the undistorted frame and print centers in mm
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for index, i in enumerate(circles[0, :]):
                    center_pixels = (i[0], i[1])

                    # Convert pixel coordinates to mm
                    center_mm = (center_pixels[0] * conversion_factor, center_pixels[1] * conversion_factor)

                    # Circle center
                    cv.circle(undistorted_frame, center_pixels, 1, (0, 100, 100), 3)
                    # Circle outline
                    radius = i[2]
                    cv.circle(undistorted_frame, center_pixels, radius, (255, 0, 255), 3)

                    # Put text with circle number and center in mm
                    text = f"bottle {index + 1}: ({center_mm[0]:.2f}, {center_mm[1]:.2f}) mm"
                    print(text)
                    cv.putText(undistorted_frame, text, (center_pixels[0] - 30, center_pixels[1]), 
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Display the undistorted frame
            cv.imshow("Detected Circles", undistorted_frame)

            # Press 'q' to quit
            if cv.waitKey(1) == ord('q'):
                break

    finally:
        # Stop streaming
        pipeline.stop()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()