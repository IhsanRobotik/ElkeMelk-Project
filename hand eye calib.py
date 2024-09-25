import cv2 
import numpy as np 

# my robot coord
hand_coords = np.array([[-0.230, -0.351, 1], [-0.203, -0.500, 1], 
                         [-0.293, -0.520, 1], [-0.317, -0.410, 1]])

eye_coords = np.array([[61, 54, 1], [218, 59, 1], 
                        [219, 154, 1], [104, 155, 1]])

# rotation matrix between the target and camera 
R_target2cam = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [ 
                        0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]) 
  
# translation vector between the target and camera 
t_target2cam = np.array([0.0, 0.0, 0.0, 0.0]) 

T, _ = cv2.calibrateHandEye(hand_coords, eye_coords, R_target2cam, t_target2cam)
print (T)

#test some pixel coordinate
c = np.array([143, 110, 1]) 

transformed_coords = T @ c

print(transformed_coords)