import socket
import time

# Function to send a script to the robot
def send_script_to_robot(script_text, host, port):
    # Create a socket connection to the robot
    mySocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    mySocket.connect((host, port))
    
    # Send the script to the robot
    mySocket.send((script_text + "\n").encode())
    
    # Close the socket connection
    mySocket.close()

# Change the robot IP address and port here
host = '192.168.0.43'
port = 30001

# Define the scripts
script_1 = """def test_move1():
   a = get_actual_tcp_pose()
   global P_start_p=p[0.6106, -0.15197, 0.300609, -2.2919, 2.1463, -0.0555]
   global P_mid_p=p[.6206, -.1497, .3721, 2.2919, -2.1463, -.0555]
   global P_end_p=p[.6206, -.1497, .4658, 2.2919, -2.1463, -.0555]
   movel(P_start_p, a=0.5, v=0.20)
   movel(a)
end"""
 
script_2 = """def test_move2():
   global P_mid_p=p[.6306, -.1497, .3721, 2.2919, -2.1463, -.0555]
   movel(P_mid_p, a=0.5, v=0.20)
end"""

# Send the first script
send_script_to_robot(script_1, host, port)

# Wait for the first script to complete (adjust time based on the robot's action duration)
time.sleep(5)  # Wait for 5 seconds

# Send the second script
send_script_to_robot(script_2, host, port)
