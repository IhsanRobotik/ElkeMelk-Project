def main():
    global array
    mtx = np.array([[893.38874436, 0, 652.46300526],
                    [0, 892.40326491, 360.40764759],
                    [0, 0, 1]])
    dist = np.array([0.20148339, -0.99826633, 0.00147814, 0.00218007, 1.33627184])
    known_width_mm = 329
    known_pixel_width = 1280

    # Calculate conversion factor from pixels to mm
    conversion_factor = known_width_mm / known_pixel_width

    # Initialize the video capture object
    cap = cv.VideoCapture(camera)  # Change to 0 for the default camera
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        return -1
    roi_x, roi_y, roi_w, roi_h = 0, 240, 1280, 210  # Define the ROI coordinates

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Undistort the frame
        h, w = frame.shape[:2]
        new_camera_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        undistorted_frame = cv.undistort(frame, mtx, dist, None, new_camera_mtx)

        # Crop the undistorted frame to the ROI
        roi_frame = undistorted_frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        # Wait for the "trig" message before processing YOLO detection
        msg = clientsocket.recv(1024)
        if not msg:  # If no message is received, break the loop
            break
        msg = msg.decode("utf-8")
        print(msg)

        if "trig" in msg:
            print("Received 'trig', starting circle detection.")

            # Run YOLOv8 inference on the cropped ROI
            results = model(roi_frame, verbose=False, conf=0.85)

            # Convert YOLOv8 results back into an OpenCV-friendly format for display
            annotated_roi_frame = results[0].plot()

            # Overlay the annotated ROI back onto the original annotated frame
            annotated_frame = undistorted_frame.copy()
            annotated_frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = annotated_roi_frame

            # Draw the ROI rectangle on the annotated frame
            cv.rectangle(annotated_frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)

            # Initialize variables to track the leftmost object
            leftmost_x = float('-inf')
            leftmost_center = None

            # Iterate over the results and find the leftmost detected object
            if results and results[0].boxes:
                for i, box in enumerate(results[0].boxes):
                    # Get bounding box coordinates
                    try:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extracting coordinates and converting to integers
                    except Exception as e:
                        continue  # Skip this box if something is wrong
                    
                    # Calculate the center point
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2 + 240  # Adjust y-coordinate as needed
                    cv.circle(annotated_frame, (int(center_x), int(center_y)), 5, (255, 0, 0), -1)

                    # Check if this object is the leftmost one
                    if center_x > leftmost_x:
                        leftmost_x = center_x
                        leftmost_center = (center_x, center_y)

            # Output the leftmost object's center
            if leftmost_center:
                print(f"Leftmost object center: {leftmost_center}")

                # Calculate real-world coordinates based on detection
                realX = leftmost_center[0] * conversion_factor
                realY = leftmost_center[1] * conversion_factor

                deltaY = (-1) * realX
                deltaX = (-1) * realY

                pickupX = deltaX + offsetX + ((firstposX - array[0]) * (-1))
                pickupY = deltaY + offsetY + ((firstposY - array[1]) * (-1))

                print(f"Robot coords: {pickupX}, {pickupY}")
            # Send the coordinates to the client
            formatted_string = "({0}, {1})".format(pickupX, pickupY)
            clientsocket.send(bytes(formatted_string, "ascii"))
            print("Robot Pick-Up Coordinate:", pickupX, pickupY)

        else:
            counter += 1
            if counter > 25:
                print("Send 69")
                clientsocket.send(bytes("(69)", "ascii"))

            

        elif "p" in msg:
            # Process "p" message to update array
            cleaned_msg = msg.replace("p", "").replace("trig", "")
            print("this is cleaned msg", cleaned_msg)
            array = ast.literal_eval(cleaned_msg)
            array[0] = array[0] * 1000
            array[1] = array[1] * 1000
            counter = 0

        
        # Display the annotated frame with the detections
        cv.imshow("Detected Objects", annotated_frame)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
