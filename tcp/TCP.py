import socket
import time

# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the address and port
s.bind(("192.168.78.1", 1244))

# Listen for incoming connections
s.listen(5)
print("Server is listening for connections...")

while True:
    # Accept a connection from a client
    clientsocket, address = s.accept()
    print(f"Connection from {address} has been established!")

    while True:
        # Receive a message from the client
        msg = clientsocket.recv(1024)
        if not msg:  # If no message is received, break the loop
            break
        print(msg.decode("utf-8"))

        # Sending a message back to the client
        message_to_send = "(1.2,1.3,1.4)"  # Let's say this is our response
        clientsocket.send(bytes(message_to_send, "ascii"))

        time.sleep(1)

    # Close the client socket
    clientsocket.close()