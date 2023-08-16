#!/usr/bin/env python3
import rospy
import numpy as np
from tqdm import tqdm
import random
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
import std_msgs
import ros_numpy
import message_filters
import tf.transformations as tr
import pdb
import ctypes
import struct
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
import time
from tf2_msgs.msg import TFMessage
from ros_numpy.point_cloud2 import pointcloud2_to_array, array_to_pointcloud2, split_rgb_field, merge_rgb_fields
# import pyntcloud
import seaborn as sns
import open3d as o3d
from datetime import datetime
from scipy.spatial.distance import euclidean, cdist
import pyntcloud


class PointcloudProcessor():
    def __init__(self):
        rospy.init_node('pcl_processor', anonymous=True)
        # rospy.Subscriber('/velodyne_points_filtered', PointCloud2, self.pcl_cluster_meancov)
        rospy.Subscriber('/map', PointCloud2, self.pcl_cluster_meancov)

        self.pclpublisher = rospy.Publisher(
            'filtered_clusters', PointCloud2, latch=True, queue_size=100)
        self.pclpublisher2 = rospy.Publisher(
            'dandelion_map', PointCloud2, latch=True, queue_size=100)

        self.ref_dandelion_cov = np.array([[1.57329671e-05, -4.12004470e-06, -1.80839649e-05],
                                           [-4.12004470e-06,  2.32158495e-04,
                                               2.14231478e-06],
                                           [-1.80839649e-05,  2.14231478e-06,  1.58321861e-04]])
        
        self.prev_pcl_mat = None
        self.trans_init = None


    def pose_to_pq(self, msg):
        """Convert a C{nav_msgs/Odometry} into position/quaternion np arrays


        @param msg: ROS message to be converted
        @return:
        - p: position as a np.array
        - q: quaternion as a numpy array (order = [x,y,z,w])
        """
        p = np.array([msg.pose.pose.position.x,
                     msg.pose.pose.position.y, msg.pose.pose.position.z])
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

    def pcl_mat_to_XYZRGB(self, pcl_mat):
        # Extract color components
        red = pcl_mat['r']
        green = pcl_mat['g']
        blue = pcl_mat['b']

        # Create a new point cloud matrix with XYZRGB values
        xyzrgb = np.zeros((pcl_mat.shape[0], 6))
        xyzrgb[:, 0] = pcl_mat['x']
        xyzrgb[:, 1] = pcl_mat['y']
        xyzrgb[:, 2] = pcl_mat['z']
        xyzrgb[:, 3] = red
        xyzrgb[:, 4] = green
        xyzrgb[:, 5] = blue

        return xyzrgb

    def XYZRGB_to_pcl_mat(self, xyzrgb):
        # Extract XYZRGB components
        x = xyzrgb[:, 0]
        y = xyzrgb[:, 1]
        z = xyzrgb[:, 2]
        red = xyzrgb[:, 3]
        green = xyzrgb[:, 4]
        blue = xyzrgb[:, 5]

        # Create a new point cloud matrix with separate components
        pcl_mat = np.zeros((xyzrgb.shape[0],),
                           dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32),
                                  ('r', np.uint8), ('g', np.uint8), ('b', np.uint8)])
        pcl_mat['x'] = x
        pcl_mat['y'] = y
        pcl_mat['z'] = z
        pcl_mat['r'] = red.astype(np.uint8)
        pcl_mat['g'] = green.astype(np.uint8)
        pcl_mat['b'] = blue.astype(np.uint8)

        return pcl_mat

    def compute_cluster_stats(self, xyz_cl):
        cluster_labels = np.unique(xyz_cl[:, -1])
        cluster_stats = {}

        # print("debug checkpoint cluster_labels", len(cluster_labels))

        for label in cluster_labels:
            if label == -1:
                continue  # Skip noise points

            cluster_points = xyz_cl[xyz_cl[:, -1] == label][:, :3]
            cluster_mean = np.mean(cluster_points, axis=0)
            cluster_cov = np.cov(cluster_points, rowvar=False)

            cluster_stats[label] = {
                'mean': cluster_mean, 'covariance': cluster_cov}

        return cluster_stats

    def compare_covariances_SVD(self, dandelion_dict):

        similar_clusters = []  # List to store similar clusters

        for cluster_num, dandelion in dandelion_dict.items():
            covariance = dandelion['covariance']

            if covariance.ndim < 2:
                print("Invalid covariance matrix for cluster {}. Skipping comparison.".format(
                    cluster_num))
                continue

            # Perform SVD on the reference covariance matrix
            U_reference, S_reference, V_reference = np.linalg.svd(
                self.ref_dandelion_cov)

            # Perform SVD on the specimen covariance matrix
            U_specimen, S_specimen, V_specimen = np.linalg.svd(covariance)

            # Compare the eigenvalues
            # lesser then tolereance then more tighter the threshold is
            eigenvalue_similarity = np.allclose(
                S_reference, S_specimen, atol=0.001, rtol=0.0001)

            if eigenvalue_similarity:
                # Append cluster number to similar_clusters list
                similar_clusters.append(cluster_num)

        return similar_clusters  # Return the list of similar clusters
    
    def ICP_alignment(self, source_pcl, target_pcl, max_iterations=100, tolerance=1e-6):

        source_cloud = o3d.geometry.PointCloud()
        target_cloud = o3d.geometry.PointCloud()
        source_cloud.points = o3d.utility.Vector3dVector(source_pcl[:, :3])
        target_cloud.points = o3d.utility.Vector3dVector(target_pcl[:, :3])

        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations, 
                                                          relative_fitness=tolerance, 
                                                          relative_rmse=tolerance)

        # Perform ICP alignment
        transformation = o3d.pipelines.registration.registration_icp(source_cloud, target_cloud, 
                                                           max_correspondence_distance=0.05, 
                                                           criteria=criteria).transformation

        # Transform the source point cloud using the obtained transformation matrix
        aligned_source_pcl = np.hstack((source_pcl[:, :3], np.ones((source_pcl.shape[0], 1))))
        aligned_source_pcl = np.dot(aligned_source_pcl, transformation.T)
        aligned_source_pcl = np.hstack((aligned_source_pcl[:, :3], source_pcl[:, 3:]))

        return aligned_source_pcl
    

    def pcl_cluster_meancov(self, pcl_msg):

        pcl_xyzcol = pointcloud2_to_array(pcl_msg)
        pcl_mat = split_rgb_field(pcl_xyzcol)
        print("CHECKPOINT 1 pcl_mat shape: ", pcl_mat.shape)

        # maskint1 = ((pcl_mat['z'] < 0.03) | (pcl_mat['z'] > 0.30))

        # pcl_mat = pcl_mat[np.logical_not(maskint1)]

        # SSG edit
        curr_pcl_mat = self.pcl_mat_to_XYZRGB(pcl_mat)

        print("current_pcl_mat shape:", curr_pcl_mat.shape)

        # ICP alignment between previous point cloud and current point cloud
        if self.prev_pcl_mat is not None:
            aligned_pcl_mat = self.ICP_alignment(self.prev_pcl_mat, curr_pcl_mat)

        else:
            aligned_pcl_mat = curr_pcl_mat

        self.prev_pcl_mat = curr_pcl_mat

        print("aligned_pcl_mat shape:", aligned_pcl_mat.shape)
            
        pcl_unfiltered = self.XYZRGB_to_pcl_mat(
            aligned_pcl_mat[:, :6])

        pc_array_unfiltered = merge_rgb_fields(pcl_unfiltered)
        pc_msg_unfiltered = ros_numpy.msgify(
            PointCloud2, pc_array_unfiltered, stamp=pcl_msg.header.stamp, frame_id=pcl_msg.header.frame_id)
        self.pclpublisher.publish(pc_msg_unfiltered)


if __name__ == '__main__':
    pcl_node = PointcloudProcessor()
    rospy.spin()
