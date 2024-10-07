import cv2 as cv
import numpy as np
import socket
import time
import ast

# Creëer een socketobject
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind de socket aan het adres en de poort
s.bind(("192.168.0.45", 5005))

# Luister naar binnenkomende verbindingen
s.listen(5)
print("Server is listening for connections...")

# Accepteer een verbinding van een client
clientsocket, address = s.accept()
print(f"Connection from {address} has been established!")

# Stel een timeout in voor het ontvangen van gegevens om te voorkomen dat het blokkeert
clientsocket.settimeout(5.0)

pickupX = 0
pickupY = 0

bottleXRef = 212.31
bottleYRef = 222.36

robotXRef = 185.29
robotYRef = -500.52


# Camera matrix en distorsiecoëfficiënten
mtx = np.array([[663.81055373, 0, 320.9241384],
                [0, 663.2383916, 241.48871247],
                [0, 0, 1]])

dist = np.array([2.75825516e-01, -1.88952738e+00, 2.27429839e-03,
                 -2.77074731e-03, 4.02307100e+00])

known_width_mm = 402 
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

        # Detecteer cirkels (flessen)
        rows = gray.shape[0]
        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                                   param1=40, param2=25,
                                   minRadius=25, maxRadius=33)

        if circles is not None:
            circles = np.uint16(np.around(circles))

            # Sorteer cirkels op hun y-coördinaten
            circles = sorted(circles[0, :], key=lambda x: x[1])

            for index, i in enumerate(circles):
                center_pixels = (i[0], i[1])

                # Converteer pixelcoördinaten naar mm voor de fles
                center_mm = (center_pixels[0] * conversion_factor, center_pixels[1] * conversion_factor)

                bottleXRef = 212.31
                bottleYRef = 222.36

                robotXRef = 185.29
                robotYRef = -500.52

                # Bereken de delta's tussen de fles en de gripperpositie
                deltaX = center_mm[0] - bottleXRef
                deltaY = center_mm[1] - bottleYRef

                pickupX = robotXRef - deltaY
                pickupY = robotYRef - deltaX
                        
                # Stuur de nieuwe coördinaten naar de robot
                formatted_string = "({0}, {1})".format(pickupX, pickupY)
                message_to_send = formatted_string  # Coordinates to send
                clientsocket.send(bytes(message_to_send, "ascii"))

                # Wacht op het bericht van de robot dat hij op de juiste positie is
                robot_at_position = False
                while not robot_at_position:
                    try:
                        robot_msg = clientsocket.recv(1024)
                        if robot_msg.decode("utf-8") == "I am there":
                            # Robot heeft de positie bereikt, sla de nieuwe coördinaten op
                            robotXRef = pickupX
                            robotYRef = pickupY
                            bottleXRef = center_mm[0]
                            bottleYRef = center_mm[1]

                            print("Coordinates saved successfully!")

                            # Stuur bericht terug naar de robot
                            clientsocket.send("saved it".encode("utf-8"))
                            robot_at_position = True
                            print("Sent confirmation to the robot: 'saved it'. Robot can proceed.")
                    except socket.timeout:
                        continue
                    except Exception as e:
                        print(f"Error occurred: {e}")
                        break
                
                # Teken de cirkel en plaats tekst
                cv.circle(undistorted_frame, center_pixels, 1, (0, 100, 100), 3)
                radius = i[2]
                cv.circle(undistorted_frame, center_pixels, radius, (255, 0, 255), 3)

                text = f"{index + 1}: ({center_mm[0]:.2f}, {center_mm[1]:.2f}) mm"
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
