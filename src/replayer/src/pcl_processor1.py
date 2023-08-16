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


class PointcloudProcessor():
    def __init__(self):
        rospy.init_node('pcl_processor', anonymous=True)
        # rospy.Subscriber('/velodyne_points_filtered', PointCloud2, self.pcl_cluster_meancov)
        rospy.Subscriber('/map', PointCloud2, self.pcl_cluster_meancov)

        self.pclpublisher = rospy.Publisher(
            'map2', PointCloud2, latch=True, queue_size=100)
        self.pclpublisher2 = rospy.Publisher(
            'similar_clusters', PointCloud2, latch=True, queue_size=100)
        
        self.ref_dandelion_cov = np.array([[ 1.57329671e-05, -4.12004470e-06, -1.80839649e-05],
            [-4.12004470e-06,  2.32158495e-04,  2.14231478e-06],
            [-1.80839649e-05,  2.14231478e-06,  1.58321861e-04]])
        # [1.57719516e-04, 9.82519920e-07, 7.33407198e-06],
        # [9.82519920e-07, 1.51163665e-04, 3.57459912e-05],
        # [7.33407198e-06, 3.57459912e-05, 2.60784083e-05]])

        self.pcl_buffer = {}

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

    def pcl_mat_generator(self, ros_point_cloud):
        xyz = np.array([[0, 0, 0]])
        rgb = np.array([[0, 0, 0]])
        # self.lock.acquire()
        gen = pc2.read_points(ros_point_cloud, skip_nans=True)
        int_data = list(gen)

        for x in int_data:
            test = x[3]
            # cast float32 to int so that bitwise operations are possible
            s = struct.pack('>f', test)
            i = struct.unpack('>l', s)[0]
            # you can get back the float value by the inverse operations
            pack = ctypes.c_uint32(i).value
            r = (pack & 0x00FF0000) >> 16
            g = (pack & 0x0000FF00) >> 8
            b = (pack & 0x000000FF)
            # prints r,g,b values in the 0-255 range
            # x,y,z can be retrieved from the x[0],x[1],x[2]
            xyz = np.append(xyz, [[x[0], x[1], x[2]]], axis=0)
            rgb = np.append(rgb, [[r, g, b]], axis=0)

        return [xyz, rgb]

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

        print("debug checkpoint cluster_labels", len(cluster_labels))

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
                S_reference, S_specimen, rtol=0.0001, atol=0.0001)

            if eigenvalue_similarity:
                # Append cluster number to similar_clusters list
                similar_clusters.append(cluster_num)

        return similar_clusters  # Return the list of similar clusters

    def downsample_voxel_grid(self, pcl_mat_xyzrgb):
        # Convert the XYZRGB point cloud to Open3D format
        pcl_cloud = o3d.geometry.PointCloud()
        pcl_cloud.points = o3d.utility.Vector3dVector(pcl_mat_xyzrgb[:, :3])
        pcl_cloud.colors = o3d.utility.Vector3dVector(
            pcl_mat_xyzrgb[:, 3:6] / 255.0)

        # Create a voxel grid filter object
        voxel_grid = pcl_cloud.voxel_down_sample(
            voxel_size=0.001)  # Adjust the voxel size as needed

        # Perform downsampling
        downsampled_cloud = voxel_grid

        # Convert the downsampled point cloud back to numpy array format
        downsampled_pcl_mat_xyzrgb = np.concatenate([np.asarray(downsampled_cloud.points),
                                                    np.asarray(downsampled_cloud.colors) * 255], axis=1)

        return downsampled_pcl_mat_xyzrgb

    def pcl_cluster_meancov(self, pcl_msg):
        # pcl_mat = self.pcl_matrix_converter(pcl_msg)
        pcl_xyzcol = pointcloud2_to_array(pcl_msg)
        pcl_mat = split_rgb_field(pcl_xyzcol)
        print("CHECKPOINT 1 pcl_mat shape: ", pcl_mat.shape)

        # maskint1 = ((pcl_mat['z'] < -0.1) | (pcl_mat['z'] > 0.15))
        # maskint1 = ((pcl_mat['z'] < 0.14) | (pcl_mat['z'] > 0.38))
        # pcl_mat = pcl_mat[np.logical_not(maskint1)]
        # print("CHECKPOINT 2 pcl_mat shape: ", pcl_mat.shape)

        # SSG edit
        maskintcol1 = (pcl_mat['r'] <= 210)
        maskintcol2 = (pcl_mat['g'] <= 210)
        maskintcol3 = (pcl_mat['b'] >= 110)
        pcl_mat = pcl_mat[np.logical_not(np.logical_and(
            maskintcol1, maskintcol2, maskintcol3))]
        # print("CHECKPOINT 3 pcl_mat shape: ", pcl_mat.shape)

        # SSG edit
        pcl_mat_xyzrgb = self.pcl_mat_to_XYZRGB(pcl_mat)
        # print("CHECKPOINT 3.2pcl_mat_xyzrgb shape: ", pcl_mat_xyzrgb.shape)

        # Downsampling pcl_mat_xyzrgb
        pcl_mat_xyzrgb = self.downsample_voxel_grid(pcl_mat_xyzrgb)

        # print("CHECKPOINT 3.3 pcl_mat_xyzrgb downsampled shape: ",       pcl_mat_xyzrgb.shape)

        t0_dbscan = time.time()

        db = DBSCAN(eps=0.025, min_samples=18, n_jobs=-1)
        for _ in tqdm(range(10), desc='Clustering'):
            db.fit(pcl_mat_xyzrgb[:, :3])

        # print("CHECKPOINT 4 pcl_mat_xyzrgb shape: ", pcl_mat_xyzrgb.shape)

        # print("DBSCAN computation time:", time.time() - t0_dbscan)

        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        # print('Estimated number of clusters: %d' % n_clusters)
        xyz_cl = np.c_[pcl_mat_xyzrgb, labels]  # will be n x 7 cols array
        # print("CHECKPOINT 5 xyz_cl shape: ", xyz_cl.shape)

        # de-noises data
        maskint2 = (xyz_cl[:, -1] == -1)
        xyz_cl = xyz_cl[np.logical_not(maskint2)]

        # Count the number of occurrences of each unique label
        lbls, counts = np.unique(xyz_cl[:, -1], return_counts=True)
        mask = np.isin(xyz_cl[:, -1], lbls[counts > 100])
        # xyz_cl = xyz_cl[~mask]
        # print("CHECKPOINT 6 xyz_cl shape: ", xyz_cl.shape)

        label_components = np.unique(xyz_cl[:, -1])
        # print("Number of clusters after removing noise:", len(label_components))
        t0 = time.time()
        cluster_gaussian_stats = self.compute_cluster_stats(xyz_cl)
        # print("Completed Gaussian statistic computation on clusters in time",time.time() - t0)

        # print("CHECKPOINT 7 cluster_gaussian_stats : ",len(cluster_gaussian_stats))

        # Use SVD to compare covariance structure of clusters with reference dandelion
        # similar_clusters = self.compare_covariances_SVD(cluster_gaussian_stats)
        # similar_clusters_np = np.array(similar_clusters)

        # print("CHECKPOINT 8 similar_clusters : ", similar_clusters_np)

        # # # Assign a unique color to each cluster
        # # num_clusters = len(label_components)

        # # de noise the clustered point cloud
        # # save this pointcloud in a buffer
        # # next pcl_msg I receive, I will compare it with the pointcloud in the buffer
        # pc_array = merge_rgb_fields(self.XYZRGB_to_pcl_mat(xyz_cl[:, :6]))

        # # Define the PointCloud2 message header
        # pc_msg = ros_numpy.msgify(
        #     PointCloud2, pc_array, stamp=pcl_msg.header.stamp, frame_id=pcl_msg.header.frame_id)

        # self.pclpublisher.publish(pc_msg)

        # # # Filter points for cluster label 2 and publish
        # cluster_labels = similar_clusters_np
        # selected_points = []

        # for label in cluster_labels:
        #     points = xyz_cl[xyz_cl[:, -1] == label]
        #     selected_points.extend(points)

        # selected_points = np.array(selected_points)
        # print("CHECKPOINT 9 similar_clusters shape : ", selected_points.shape)

        # pc_array_cluster2 = merge_rgb_fields(
        #     self.XYZRGB_to_pcl_mat(selected_points[:, :6]))
        # pc_msg_cluster2 = ros_numpy.msgify(
        #     PointCloud2, pc_array_cluster2, stamp=pcl_msg.header.stamp, frame_id=pcl_msg.header.frame_id)
        # self.pclpublisher2.publish(pc_msg_cluster2)


        

        #################################################################

        similar_clusters = self.compare_covariances_SVD(cluster_gaussian_stats)
        similar_clusters_np = np.array(similar_clusters)

        print("CHECK POINT 2 similar_clusters_np:", similar_clusters_np)

        cluster_labels = similar_clusters_np
        selected_points = []

        for label in cluster_labels:
            points = xyz_cl[xyz_cl[:, -1] == label]
            selected_points.extend(points)

        selected_points = np.array(selected_points)
        # print("CHECKPOINT 9 similar_clusters shape : ", selected_points.shape)

        pc_array_cluster2 = merge_rgb_fields(
            self.XYZRGB_to_pcl_mat(selected_points[:, :6]))
        pc_msg_cluster2 = ros_numpy.msgify(
            PointCloud2, pc_array_cluster2, stamp=pcl_msg.header.stamp, frame_id=pcl_msg.header.frame_id)
        self.pclpublisher2.publish(pc_msg_cluster2)

        #############################################################

        # similar_cluster_means = []
        # for label in similar_clusters_np:
        #     if label in cluster_gaussian_stats:
        #         cluster_mean = cluster_gaussian_stats[label]['mean']
        #         similar_cluster_means.append((label, cluster_mean))

        # # Print the labels and means of similar clusters
        # for label, mean in similar_cluster_means:
        #     print("Similar Cluster Label: ", label)
        #     print("Cluster Mean: ", mean)

        buffer_entry = {}
        buffer_entry['labels'] = similar_clusters_np.tolist()
        buffer_entry['means'] = []
        buffer_entry['points'] = []

        for label in similar_clusters_np:
            if label in cluster_gaussian_stats:
                cluster_mean = cluster_gaussian_stats[label]['mean']
                cluster_points = xyz_cl[xyz_cl[:, -1] == label][:, :3]

                buffer_entry['means'].append(cluster_mean.tolist())
                buffer_entry['points'].append(cluster_points.tolist())

        # Store the buffer entry with timestamp
        self.pcl_buffer[pcl_msg.header.stamp] = buffer_entry

        timestamp = pcl_msg.header.stamp.to_sec()

        current_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')
        print("CHECK POINT 3.1 current time stamp",current_time)
        # print("CHECK POINT 3.2 self.pcl_buffer",self.pcl_buffer)

            # Find the most recent previous timestamp
        prev_time_stamp = None
        for timestamp in self.pcl_buffer.keys():
            if timestamp < pcl_msg.header.stamp:
                if prev_time_stamp is None or timestamp > prev_time_stamp:
                    prev_time_stamp = timestamp

        if prev_time_stamp is not None:
            prev_buffer_entry = self.pcl_buffer[prev_time_stamp]
            prev_time = datetime.fromtimestamp(prev_time_stamp.to_sec()).strftime('%Y-%m-%d %H:%M:%S.%f')
            print("CHECK POINT 4 prev time stamp", prev_time)

        # # Compare the current clusters with the clusters from the previous time step
        # if len(self.pcl_buffer) > 1:
        #     prev_time_stamp = pcl_msg.header.stamp - rospy.Duration(1)
        #     # print("prev_time_stamp",prev_time_stamp)
            
        #     if prev_time_stamp in self.pcl_buffer:
        #         prev_clusters = self.pcl_buffer[prev_time_stamp]  # Get the clusters from the previous time step
                
        #         similar_clusters_prev = np.array(prev_clusters)
        #         # print("t-1 prev_clusters",similar_clusters_prev)

        #         # Convert the current clusters to a different color
        #         unique_labels = np.unique(similar_clusters_np)
        #         # print("t similar_clusters_np:", similar_clusters_np)
        #         for label in unique_labels:
        #             mask = similar_clusters_np == label
        #             pcl_mat_xyzrgb[mask, 3:6] = [255, 0, 0]  # Set color to red (RGB: 255, 0, 0)

        #         # Convert the clusters from the previous time step to a different color
        #         unique_labels_prev = np.unique(similar_clusters_prev)
        #         for label in unique_labels_prev:
        #             mask = similar_clusters_prev == label
        #             pcl_mat_xyzrgb[mask, 3:6] = [0, 0, 255]  # Set color to blue (RGB: 0, 0, 255)

        #         # Publish the point cloud with different colored clusters for the previous time step
        #         pc_array_prev = merge_rgb_fields(self.XYZRGB_to_pcl_mat(pcl_mat_xyzrgb))
        #         pc_msg_prev = ros_numpy.msgify(PointCloud2, pc_array_prev, stamp=pcl_msg.header.stamp, frame_id=pcl_msg.header.frame_id)
        #         self.pclpublisher.publish(pc_msg_prev)

        # # Publish the point cloud with different colored clusters for the current time step
        # pc_array_curr = merge_rgb_fields(self.XYZRGB_to_pcl_mat(pcl_mat_xyzrgb))
        # pc_msg_curr = ros_numpy.msgify(PointCloud2, pc_array_curr, stamp=pcl_msg.header.stamp, frame_id=pcl_msg.header.frame_id)
        # self.pclpublisher2.publish(pc_msg_curr)

if __name__ == '__main__':
    pcl_node = PointcloudProcessor()
    rospy.spin()
