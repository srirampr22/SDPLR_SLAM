#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry

def pointcloud_callback(data):
    print("Received PointCloud2 message:\n", data)

def odom_callback(data):
    print("Received Odometry message:\n", data)

def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('/camera/depth/color/target_points', PointCloud2, pointcloud_callback)
    rospy.Subscriber('/odom', Odometry, odom_callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
