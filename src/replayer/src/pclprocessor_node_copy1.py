import pdb
import rospy
import roslib
import math
import tf
import numpy as np
# import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from ros_numpy.point_cloud2 import pointcloud2_to_array, array_to_pointcloud2, split_rgb_field, merge_rgb_fields

# for RGBD camera frustum visualization
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Quaternion, Pose, Point, Vector3
from std_msgs.msg import Header, ColorRGBA

import cv2
import pyrealsense2 as realsense

lastpcl_data = ""
started = False
pclpublisher = rospy.Publisher('/map2', PointCloud2, latch=True, queue_size=1000)
marker_publisher = rospy.Publisher('visualization_marker', Marker)

def frustum_visualize():
    pass


def pcl_callback(pcl2data):
    print("New message received", "\n")
    global started, lastpcl_data
    pcl_data = pcl2data # <class 'sensor_msgs.msg._PointCloud2.PointCloud2'>
    pcl2_array = pointcloud2_to_array(pcl_data)
    pcl2_array = split_rgb_field(pcl2_array)
    print('pcl2 array shape BEFORE color separation: ', pcl2_array.shape)

    # maskint1 = (pcl2_array['r'] <= 190)
    # maskint2 = (pcl2_array['g'] <= 190)
    # maskint3 = (pcl2_array['b'] >=140)
    maskint1 = (pcl2_array['r'] <= 210)
    maskint2 = (pcl2_array['g'] <= 210)
    maskint3 = (pcl2_array['b'] >=130)
    pcl2_array = pcl2_array[np.logical_not(np.logical_and(maskint1,maskint2,maskint3))]
    
    maskint4 = (pcl2_array['z'] >= 0.35)
    pcl2_array = pcl2_array[np.logical_not(maskint4)]

    pcl2_array = merge_rgb_fields(pcl2_array)
    print('pcl2 array shape AFTER color separation: ', pcl2_array.shape)
    lastpcl_data = array_to_pointcloud2(pcl2_array, stamp=pcl2data.header.stamp, frame_id=pcl2data.header.frame_id) # <class 'sensor_msgs.msg._PointCloud2.PointCloud2'>
    
    
    print('CHECKPOINT ****1')
    if (not started):
        started = True

def timer_callback(event):
    global started, pclpublisher, lastpcl_data
    if (started):
        pclpublisher.publish(lastpcl_data)
        # print("Last message published", "\n")

def pcl_processor():
    rospy.init_node('pclprocessor')
    rospy.Subscriber('/map', PointCloud2, pcl_callback)
    timer = rospy.Timer(rospy.Duration(0.5), timer_callback)
    
    rospy.spin()
    timer.shutdown()

if __name__ == '__main__':
    # print("Running")
    pcl_processor()

# rosbag file names for ease of entering in replay_recording.launch
# single_d1_0p1res.bag
# NOTE: Remember to set map topic to /map2 to visualize processed pointcloud2 in RViz!