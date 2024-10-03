from ultralytics import YOLO
import cv2
import numpy as np
import sys

# Laad je YOLOv8-model (gebruik het juiste pad voor je modelbestand)
model = YOLO(r"C:/Users/basti/Documents/GitHub/ElkeMelk-Project/YOLOv8_Training_2/runs/detect/train/weights/best.pt")

# Camera matrix en vervormingscoëfficiënten
mtx = np.array([[680.43205516, 0, 325.70663487],
                [0, 680.82113726, 224.79035512],
                [0, 0, 1]])

dist = np.array([2.86943840e-01, -1.81250009e+00, -9.85433600e-03,
                 2.34191361e-03, 4.19774044e+00])

# Bekende breedte van object in mm en de breedte in pixels
known_width_mm = 560  # Breedte van de fles in mm
known_pixel_width = 640  # Bekende breedte in pixels van de fles op een specifieke afstand

# Bereken de conversiefactor van pixels naar mm
conversion_factor = known_width_mm / known_pixel_width

# Functie voor cirkelherkenning binnen een bounding box (ROI)
def detect_single_circle_in_roi(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # Grijswaardenconversie
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)   # Gebruik Gaussian blur om ruis te verminderen
    
    # Pas de Hough Circle Transform toe om cirkels te detecteren
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=50,  # Minimumafstand tussen gedetecteerde cirkels
        param1=30,    # Hogere drempel voor Canny-edge detector
        param2=20,    # Accumulatiedrempel voor cirkelherkenning
        minRadius=35,  # Minimale straal van de cirkel
        maxRadius=40  # Maximale straal van de cirkel
    )
    
    # Als er cirkels zijn gevonden, retourneer de grootste
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        largest_circle = max(circles, key=lambda c: c[2])  # Sorteer op basis van straal (radius)
        return largest_circle
    return None

# Start webcam stream
cap = cv2.VideoCapture(0)  # Gebruik 0 voor de standaard webcam

if not cap.isOpened():
    print("Kan de camera niet openen")
    sys.exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kan het frame niet ophalen")
        break

    # Voer YOLO-inferentie uit op het frame
    results = model(frame)

    # Verwerk de resultaten (bounding boxes en klasse-informatie)
    for result in results:
        boxes = result.boxes.xyxy  # Bounding box coördinaten
        confidences = result.boxes.conf  # Vertrouwensscores
        classes = result.boxes.cls  # Klassevoorspellingen
        
        # Loop door gedetecteerde objecten
        for i in range(len(boxes)):
            class_id = int(classes[i])
            x1, y1, x2, y2 = map(int, boxes[i])
            confidence = confidences[i]

            # Teken de bounding box voor alle gedetecteerde objecten
            label = f"Class {class_id} {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Als klasse 'flesopening' (stel dat het class_id 17 is), voer dan cirkelherkenning uit
            if class_id == 17:
                # Uitsnijden van regio van interesse (ROI) uit de bounding box
                roi = frame[y1:y2, x1:x2]

                # Detecteer de meest prominente cirkel in de ROI
                circle = detect_single_circle_in_roi(roi)
                if circle is not None:
                    # Teken de gedetecteerde cirkel
                    (x, y, r) = circle
                    cv2.circle(roi, (x, y), r, (0, 255, 0), 4)  # Buitenste cirkel tekenen
                    cv2.circle(roi, (x, y), 2, (0, 0, 255), 3)  # Centrum van de cirkel

                    # Bereken de positie van de cirkel in millimeters
                    center_pixels = (x + x1, y + y1)
                    center_mm = (center_pixels[0] * conversion_factor, center_pixels[1] * conversion_factor)

                    # Tekst met de positie van de flesopening in mm
                    text = f"Flesopening: ({center_mm[0]:.2f}, {center_mm[1]:.2f}) mm"
                    cv2.putText(frame, text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Toon het frame met YOLO en cirkelherkenning
    cv2.imshow('YOLOv8 met cirkelherkenning voor flesopeningen', frame)

    # Druk op 'q' om te stoppen
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release de video capture en sluit alle vensters
cap.release()
cv2.destroyAllWindows()
