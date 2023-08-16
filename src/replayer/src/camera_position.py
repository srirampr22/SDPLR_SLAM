#!/usr/bin/env python3
import numpy as np
import math
import rospy
from tf2_msgs.msg import TFMessage

def tf_callback(msg):
    for transform in msg.transforms:
        if transform.child_frame_id == "camera_link":
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            print("Camera Translation:", translation)
            print("Camera Rotation:", rotation)

# rospy.init_node("tf_listener")
# rospy.Subscriber("/tf", TFMessage, tf_callback)
# rospy.spin()

def find_midpoint(point1, point2):
    midpoint = (point1 + point2) / 2
    return midpoint

# Example usage
point1 = np.array([ 0.59775488, -0.04577845,  0.00398079])
point2 = np.array([0.59763428, -0.0466758, 0.00602693])

midpoint = find_midpoint(point1, point2)
print("Midpoint:", midpoint)



# 'mean': array([ 0.59775488, -0.04577845,  0.00398079])
# Coordinates
# mean_x = 0.59763428
# mean_y = -0.0466758
# mean_z = 0.00602693
mean_x = 0.59775488
mean_y = -0.04577845
mean_z = 0.00398079
camera_x = 0.59763428
camera_y = -0.0466758
camera_z = 0.00602693

# Calculate distance
distance = math.sqrt((mean_x - camera_x)**2 + (mean_y - camera_y)**2 + (mean_z - camera_z)**2)

# Mean and covariance of the cluster
mean = np.array([ 0.59775488, -0.04577845,  0.00398079])  # Replace with your mean values
covariance = np.array([[ 1.73872415e-05, -1.04065274e-05, -3.85501391e-05],
       [-1.04065274e-05,  1.45186300e-04,  2.04395405e-05],
       [-3.85501391e-05,  2.04395405e-05,  1.30197429e-04]])  # Replace with your covariance matrix

# Calculate eigenvalues and eigenvectors of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(covariance)

# Find the maximum eigenvalue
max_eigenvalue = np.max(eigenvalues)

# Estimate the radius as the square root of the maximum eigenvalue
radius = (np.sqrt(max_eigenvalue)) * 100

print("Estimated Radius:", radius)



# Convert distance to centimeters
distance_cm = (distance/2) * 100

bias = distance_cm / radius
print("bias:", bias)

# Print the result
print("Distance:", distance_cm, "cm")