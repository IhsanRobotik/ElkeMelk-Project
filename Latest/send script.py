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

host = '192.168.0.43'
port = 30001

script_text="def test_move():\n" \
            "   a = get_actual_tcp_pose()"\
            "   global P_start_p=p[0.6106, -0.15197, 0.300609, -2.2919, 2.1463, -0.0555]\n" \
            "   global P_mid_p=p[.6206, -.1497, .3721, 2.2919, -2.1463, -.0555]\n" \
            "   global P_end_p=p[.6206, -.1497, .4658, 2.2919, -2.1463, -.0555]\n" \
            "   movel(P_start_p, a=0.5, v=0.20)\n" \
            "   movel(a)\n" \
            "end\n"


script="def tes_move():\n" \
            "   a = get_actual_tcp_pose()"\
            "   global P_start_p=p[0.6106, -0.35197, 0.300609, -2.2919, 2.1463, -0.0555]\n" \
            "   global P_mid_p=p[.6206, -.1497, .3721, 2.2919, -2.1463, -.0555]\n" \
            "   global P_end_p=p[.6206, -.1497, .4658, 2.2919, -2.1463, -.0555]\n" \
            "   movel(P_start_p, a=0.5, v=0.20)\n" \
            "   movel(a)\n" \
        "end\n"

send_script_to_robot(script_text, host, port)

# Wait for the first script to complete (adjust time based on the robot's action duration)
time.sleep(2)  # Wait for 2 seconds

# Send the second script
send_script_to_robot(script, host, port)