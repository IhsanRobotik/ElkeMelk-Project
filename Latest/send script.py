import socket

mySocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# change the robot IP address here
host = '192.168.0.43'
port = 30001

mySocket.connect((host, port))

script_text="def test_move():\n" \
            "   port = 50005" \
            "   ip = "192.168.0.1" "\
            "   socket_name = "fadil" "\
            "   socket = socket_open(ip, port, socket_name)"\
            "   movel (p[-0.13482, -0.500521, 0.562753, -2.26491, 2.14446, -0.01273])"\
            "   sleep(0.1)"\
            "   set_digital_out(0,True)"\
            "   while(True):"\
            "   socket_send_string("trig", socket_name)"\
            "   textmsg("trig send")"\
            "   a = get_actual_tcp_pose()"\
            
            
            "   global P_start_p=p[0.6106, -0.15197, 0.300609, -2.2919, 2.1463, -0.0555]\n" \
            "   global P_mid_p=p[.6206, -.1497, .3721, 2.2919, -2.1463, -.0555]\n" \
            "   global P_end_p=p[.6206, -.1497, .4658, 2.2919, -2.1463, -.0555]\n" \
            "   movel(P_start_p, a=0.5, v=0.20)\n" \
            "   movel(a)\n" \
            "end\n"




mySocket.send((script_text + "\n").encode())

script_text="def test_move():\n" \
            "   a = get_actual_tcp_pose()"\
            "   global P_start_p=p[0.6106, -0.15197, 0.300609, -2.2919, 2.1463, -0.0555]\n" \
            "   global P_mid_p=p[.6206, -.1497, .3721, 2.2919, -2.1463, -.0555]\n" \
            "   global P_end_p=p[.6206, -.1497, .4658, 2.2919, -2.1463, -.0555]\n" \
            "   movel(P_start_p, a=0.5, v=0.20)\n" \
            "   movel(a)\n" \
            "end\n"
            
mySocket.close()