import sys
import cv2 as cv
import numpy as np

# Bekende positie van de gripper op de grond (XY-coördinaten)
gripperXRef = 285.25  # Dit is de bekende X-positie op de grond
gripperYRef = 348.25  # Dit is de bekende Y-positie op de grond

def main():
    # Camera matrix en distortiecoëfficiënten
    mtx = np.array([[680.43205516, 0, 325.70663487],
                    [0, 680.82113726, 224.79035512],
                    [0, 0, 1]])  

    dist = np.array([2.86943840e-01, -1.81250009e+00, -9.85433600e-03,
                     2.34191361e-03, 4.19774044e+00]) 

    known_width_mm = 560  
    known_pixel_width = 640 

    # Bereken conversiefactor van pixels naar mm
    conversion_factor = known_width_mm / known_pixel_width

    # Initialiseer video capture object
    cap = cv.VideoCapture(0)  # Change to 0 for the default camera
    
    if not cap.isOpened():
        print("Cannot open camera")
        return -1

    while True:
        # Frame vastleggen
        ret, frame = cap.read()

        if not ret:
            print("Cannot receive frame")
            break

        # Verkrijg de frame afmetingen
        h, w = frame.shape[:2]

        # Ontdistor de frame
        new_camera_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        undistorted_frame = cv.undistort(frame, mtx, dist, None, new_camera_mtx)

        # Converteer naar grijswaarden
        gray = cv.cvtColor(undistorted_frame, cv.COLOR_BGR2GRAY)
        gray = cv.medianBlur(gray, 5)

        # Detecteer cirkels (flessen)
        rows = gray.shape[0]
        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                                   param1=18, param2=20,
                                   minRadius=19, maxRadius=25)

        # Teken de cirkels en bereken het verschil met de gripper
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for index, i in enumerate(circles[0, :]):
                center_pixels = (i[0], i[1])
                
                # Converteer pixelcoördinaten naar mm voor de fles
                center_mm = (center_pixels[0] * conversion_factor, center_pixels[1] * conversion_factor)

                # Bereken het verschil tussen fles en gripperpositie
                deltaX = gripperXRef - center_mm[0]
                deltaY = gripperYRef - center_mm[1]

                # Dit verschil kun je gebruiken om de robot te verplaatsen
                print(f"Bottle {index + 1}: Delta X = {deltaX:.2f} mm, Delta Y = {deltaY:.2f} mm")

                # Teken cirkel en center
                cv.circle(undistorted_frame, center_pixels, 1, (0, 100, 100), 3)
                cv.circle(undistorted_frame, center_pixels, i[2], (255, 0, 255), 3)

                # Plaats tekst met cirkelnummer en delta in mm
                text = f"bottle {index + 1}: dX = {deltaX:.2f}, dY = {deltaY:.2f}"
                cv.putText(undistorted_frame, text, (center_pixels[0] - 30, center_pixels[1]), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Toon de ontdistorde frame
        cv.imshow("Detected Circles", undistorted_frame)

        # Druk op 'q' om te stoppen
        if cv.waitKey(1) == ord('q'):
            break

    # Beëindig de capture en sluit alle vensters
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
