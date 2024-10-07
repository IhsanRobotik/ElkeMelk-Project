import cv2 as cv
import numpy as np
import socket
import time
import ast

# Creëer een socketobject
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind de socket aan het adres en de poort
s.bind(("192.168.0.42", 5005))

# Luister naar binnenkomende verbindingen
s.listen(5)
print("Server is listening for connections...")

# Accepteer een verbinding van een client
clientsocket, address = s.accept()
print(f"Connection from {address} has been established!")

# Stel een timeout in voor het ontvangen van gegevens om te voorkomen dat het blokkeert
clientsocket.settimeout(1.0)

pickupX = 0
pickupY = 0

# Bekende referentiepositie van de fles
bottleXRef = 285.25
bottleYRef = 348.25

# Camera matrix en distorsiecoëfficiënten
mtx = np.array([[663.81055373, 0, 320.9241384],
                [0, 663.2383916, 241.48871247],
                [0, 0, 1]])

dist = np.array([2.75825516e-01, -1.88952738e+00, 2.27429839e-03,
                 -2.77074731e-03, 4.02307100e+00])

known_width_mm = 383 
known_pixel_width = 640 

# Bereken conversiefactor van pixels naar mm
conversion_factor = known_width_mm / known_pixel_width

def main():
    # Initialiseer de videocapture
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

        # Ontdistor het frame
        new_camera_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        undistorted_frame = cv.undistort(frame, mtx, dist, None, new_camera_mtx)

        # Converteer naar grijswaarden
        gray = cv.cvtColor(undistorted_frame, cv.COLOR_BGR2GRAY)
        gray = cv.medianBlur(gray, 5)

        # Detecteer cirkels
        rows = gray.shape[0]
        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                                   param1=18, param2=20,
                                   minRadius=19, maxRadius=25)

        if circles is not None:
            circles = np.uint16(np.around(circles))

            # Sorteer cirkels op hun y-coördinaten
            circles = sorted(circles[0, :], key=lambda x: x[1])

            for index, i in enumerate(circles):
                center_pixels = (i[0], i[1])

                # Converteer pixelcoördinaten naar mm
                center_mm = (center_pixels[0] * conversion_factor, center_pixels[1] * conversion_factor)

                # Bereken de deltas
                deltaX = bottleXRef - center_mm[0]
                deltaY = bottleYRef - center_mm[1]
               
                print(f"Bottle {index + 1}: Delta X = {deltaX:.2f} mm, Delta Y = {deltaY:.2f} mm")
                # Ontvang de huidige coördinaten van de robot van de client
                try:
                    msg = clientsocket.recv(1024)

                    if msg:
                        decoded_msg = msg.decode("utf-8")
                        cleaned_msg = decoded_msg.replace("p", "")
                        array = ast.literal_eval(cleaned_msg)
                        print(array)

                        # array[0] is de huidige Y-coördinaat van de robot
                        # array[1] is de huidige X-coördinaat van de robot

                        # Bereken de nieuwe robot pick-up coördinaten
                        pickupX = (array[0]*1000) + deltaY  # Huidige X-coördinaat + deltaX
                        pickupY = (array[1]*1000) + deltaX  # Huidige Y-coördinaat + deltaY

                        # Druk de nieuwe pick-up coördinaten van de robot af
                        robotPickUpCoord = [pickupX, pickupY]
                        print(f"Robot Pick-Up Coordinates: {robotPickUpCoord}")

                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"Error occurred: {e}")
                    break

                # Teken de cirkel en plaats tekst
                cv.circle(undistorted_frame, center_pixels, 1, (0, 100, 100), 3)
                radius = i[2]
                cv.circle(undistorted_frame, center_pixels, radius, (255, 0, 255), 3)

                text = f"bottle {index + 1}: dX = {deltaX:.2f}, dY = {deltaY:.2f}"
                cv.putText(undistorted_frame, text, (center_pixels[0] - 30, center_pixels[1]), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Toon het ontdistorde frame
        cv.imshow("Detected Circles", undistorted_frame)

        # Druk op 'q' om te stoppen
        if cv.waitKey(1) == ord('q'):
            break

    # Beëindig de capture en sluit alle vensters
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
