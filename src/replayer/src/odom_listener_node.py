#!/usr/bin/env python3


import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
import math

class CameraMotion:
    def __init__(self):
        rospy.init_node('camera_motion_detector', anonymous=True)
        rospy.Subscriber("/odom", Odometry, self.callback)
        self.prev_point = None

    def callback(self, data):
        if self.prev_point is None:
            self.prev_point = data.pose.pose.position

        current_point = data.pose.pose.position
        distance_moved = self.calculate_distance(self.prev_point, current_point)

        print(distance_moved)
        if distance_moved > 0.05:  # Threshold for considering the camera as moving.
            print("Moving")
        else:
            print("Stationary")

        self.prev_point = current_point

    @staticmethod
    def calculate_distance(point1, point2):
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2 + (point1.z - point2.z)**2)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    detector = CameraMotion()
    detector.run()


