#!/usr/bin/env python
import rospy
import numpy as np
import pdb

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
import std_msgs
import ros_numpy
# from ros_numpy.point_cloud2 import pointcloud2_to_array, array_to_pointcloud2, split_rgb_field, merge_rgb_fields
import message_filters
import tf.transformations as tr
import pdb
import ctypes
import struct
import hdbscan

class PointcloudProcessor():
    def __init__(self):
        rospy.init_node('pcl_processor', anonymous=True)
        # rospy.Subscriber('/map', PointCloud2, self.callback)
        # rospy.Subscriber('/velodyne_points_filtered', PointCloud2, self.pcl_filter)
        # rospy.Subscriber('/camera/depth/color/target_points', PointCloud2, self.pcl_transformer)

        pcl_sub = message_filters.Subscriber('/camera/depth/color/target_points', PointCloud2)
        odom_sub = message_filters.Subscriber('/odom', Odometry)
        ts = message_filters.TimeSynchronizer([pcl_sub, odom_sub], 10)
        ts.registerCallback(self.pcl_transformer)

        self.pclpublisher = rospy.Publisher('transformed_Ypcl', PointCloud2, latch=True, queue_size=100)

    def pose_to_pq(self, msg):
        """Convert a C{nav_msgs/Odometry} into position/quaternion np arrays

        @param msg: ROS message to be converted
        @return:
        - p: position as a np.array
        - q: quaternion as a numpy array (order = [x,y,z,w])
        """
        p = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
        q = np.array([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        return p, q

    def msg_to_se3(self, msg):
        """Conversion from geometric ROS messages into SE(3)

        @param msg: Message to transform. Acceptable type - C{nav_msgs/Odometry}
        @return: a 4x4 SE(3) matrix as a numpy array
        @note: Throws TypeError if we receive an incorrect type.
        """
        if isinstance(msg, Odometry):
            p, q = self.pose_to_pq(msg)
        else:
            raise TypeError("Invalid type for conversion to SE(3)")

        norm = np.linalg.norm(q)
        if np.abs(norm - 1.0) > 1e-3:
            raise ValueError(
                "Received un-normalized quaternion (q = {0:s} ||q|| = {1:3.6f})".format(str(q), np.linalg.norm(q)))
        elif np.abs(norm - 1.0) > 1e-6:
            q = q / norm

        g = tr.quaternion_matrix(q)
        g[0:3, -1] = p
        return g

    def gmm_EM(self):
        pass
    

    def pcl_mat_generator(self, ros_point_cloud):
        xyz = np.array([[0,0,0]])
        rgb = np.array([[0,0,0]])
        #self.lock.acquire()
        gen = pc2.read_points(ros_point_cloud, skip_nans=True)
        int_data = list(gen)

        for x in int_data:
            test = x[3] 
            # cast float32 to int so that bitwise operations are possible
            s = struct.pack('>f' ,test)
            i = struct.unpack('>l',s)[0]
            # you can get back the float value by the inverse operations
            pack = ctypes.c_uint32(i).value
            r = (pack & 0x00FF0000)>> 16
            g = (pack & 0x0000FF00)>> 8
            b = (pack & 0x000000FF)
            # prints r,g,b values in the 0-255 range
                        # x,y,z can be retrieved from the x[0],x[1],x[2]
            xyz = np.append(xyz,[[x[0],x[1],x[2]]], axis = 0)
            rgb = np.append(rgb,[[r,g,b]], axis = 0)

        return [xyz, rgb]

    def pcl_matrix_converter(self, point_cloud_msg):
        # print('CHECKPOINT1\n')
        pc = ros_numpy.numpify(point_cloud_msg)
        # print('CHECKPOINT2\n')
        points=np.zeros((pc.shape[0],4))
        points[:,0]=pc['x']
        points[:,1]=pc['y']
        points[:,2]=pc['z']
        points[:,3]=pc['rgb']  # fix yellow color; shows up as purple
        return points

    def pcl_transformer(self, pcl_msg, odom_msg):
        pose_se3 = self.msg_to_se3(odom_msg)
        pose_se3_inv = np.linalg.inv(pose_se3)
        pcl_mat = self.pcl_matrix_converter(pcl_msg)

        # testing HDBSCAN - Hierarchical Density-Based Spatial 
        # Clustering of Applications with Noise
        xyzrgb = pcl_mat[:,:4]
        print("CHECKPOINT 1")

        # Cluster the points using HDBSCAN
        # clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
        # gen_min_span_tree=False, leaf_size=40, metric='euclidean', min_cluster_size=5, 
        # min_samples=None, p=None)
        # clusterer = hdbscan.HDBSCAN(min_samples=5, min_cluster_size=8, 
        #                             cluster_selection_epsilon=0.3)
        clusterer = hdbscan.HDBSCAN(min_samples=5, min_cluster_size=100)                      

        print("CHECKPOINT 2")
        cluster_labels = clusterer.fit_predict(xyzrgb[:,:3])
        
        print("CHECKPOINT 3")
        # pdb.set_trace()
        # Visualize the clusters
        colors = cluster_labels.astype(float) / np.max(cluster_labels)
        plt.figure()
        plt.scatter(xyzrgb[:,0], xyzrgb[:,1], c=colors)
        plt.show()

        pcl_xyz = pcl_mat[:,:3]
        ones_column = np.ones([pcl_xyz.shape[0], 1])
        transformed_pcl = np.hstack((pcl_xyz, ones_column))
        # transformed_pcl_cloud = np.matmul(pose_se3, np.transpose(transformed_pcl))
        transformed_pcl_cloud = np.matmul(pose_se3_inv, np.transpose(transformed_pcl))
        transformed_pcl_cloud[-1, :] = pcl_mat[:, -1]
        transformed_pcl_cloud = np.transpose(transformed_pcl_cloud)

        pc_array = np.zeros(len(transformed_pcl_cloud), dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            # ('intensity', np.float32),
            ('rgb', np.float32),
        ])
        pc_array['x'] = transformed_pcl_cloud[:, 0]
        pc_array['y'] = transformed_pcl_cloud[:, 1]
        pc_array['z'] = transformed_pcl_cloud[:, 2]
        pc_array['rgb'] = transformed_pcl_cloud[:, 3]

        # Define the PointCloud2 message header
        pc_msg = ros_numpy.msgify(PointCloud2, pc_array, stamp=pcl_msg.header.stamp, frame_id=pcl_msg.header.frame_id)
        self.pclpublisher.publish(pc_msg)


if __name__ == '__main__':
    pcl_node = PointcloudProcessor()
    rospy.spin()

