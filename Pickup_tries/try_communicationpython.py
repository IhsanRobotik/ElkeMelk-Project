import socket
import time

# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the address and port
s.bind(("192.168.0.45", 5005))

# Listen for incoming connections
s.listen(5)
print("Server is listening for connections...")

# Accept a connection from a client
clientsocket, address = s.accept()
print(f"Connection from {address} has been established!")

# Set a timeout for receiving data to prevent blocking
clientsocket.settimeout(5.0)

# Send the initial message to the robot
pickupX = 100.0  # Example coordinates
pickupY = 200.0
formatted_string = "({0}, {1})".format(pickupX, pickupY)
message_to_send = formatted_string
clientsocket.send(bytes(message_to_send, "ascii"))
print(f"Sent: {message_to_send}")

# Wait for confirmation from the robot
robot_at_position = False
while not robot_at_position:
    try:
        # Receive message from the robot
        print("Waiting...")
        robot_msg = clientsocket.recv(1024)
        decoded_msg = robot_msg.decode("utf-8")
        print(f"Received message from robot: {decoded_msg}")  # Debugging statement
        
        if decoded_msg == "I am there":
            # Robot has reached the position, respond with "saved it"
            clientsocket.send("saved it".encode("utf-8"))
            print("Received 'I am there' and sent 'saved it'")
            robot_at_position = True
    except socket.timeout:
        print("Waiting for response...")
    except Exception as e:
        print(f"Error occurred: {e}")
        break

# Close the socket
clientsocket.close()
s.close()
