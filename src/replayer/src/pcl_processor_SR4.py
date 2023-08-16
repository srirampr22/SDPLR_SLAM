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
# import seaborn as sns
# import open3d as o3d
from datetime import datetime
from scipy.spatial.distance import euclidean, cdist
import colorsys


class PointcloudProcessor():
    def __init__(self):
        rospy.init_node('pcl_processor', anonymous=True)
        # rospy.Subscriber('/velodyne_points_filtered', PointCloud2, self.pcl_cluster_meancov)
        rospy.Subscriber('/map', PointCloud2, self.pcl_cluster_meancov)

        self.pclpublisher = rospy.Publisher(
            'filtered_clusters', PointCloud2, latch=True, queue_size=100)
        self.pclpublisher2 = rospy.Publisher(
            'dandelion_map', PointCloud2, latch=True, queue_size=100)

        self.ref_dandelion_cov = np.array([[ 3.50720184e-06, -5.56239452e-06, -1.57297399e-05],
 [-5.56239452e-06,  1.15584851e-04, -3.63744337e-06],
 [-1.57297399e-05, -3.63744337e-06,  9.87829456e-05]])
        # [1.57719516e-04, 9.82519920e-07, 7.33407198e-06],
        # [9.82519920e-07, 1.51163665e-04, 3.57459912e-05],
        # [7.33407198e-06, 3.57459912e-05, 2.60784083e-05]])

        self.pcl_buffer = {}
        self.true_buffer = {}
        self.accumulated_points = []
        self.accumulated_label = []
        self.correspondence_dict = {}

        self.assigned_labels = []


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
        pcl_mat['r'] = red
        pcl_mat['g'] = green
        pcl_mat['b'] = blue

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
                S_reference, S_specimen, atol=0.00005, rtol=0.1)

            if eigenvalue_similarity:
                # Append cluster number to similar_clusters list
                similar_clusters.append(cluster_num)

        return similar_clusters  # Return the list of similar clusters
    
    def pcl_cluster_meancov(self, pcl_msg):
        # pcl_mat = self.pcl_matrix_converter(pcl_msg)
        pcl_xyzcol = pointcloud2_to_array(pcl_msg)
        pcl_mat = split_rgb_field(pcl_xyzcol)
        print("CHECKPOINT 1 pcl_mat shape: ", pcl_mat.shape)



        # SSG edit
        maskintcol1 = (pcl_mat['r'] == 255)
        maskintcol2 = (pcl_mat['g'] == 0)
        maskintcol3 = (pcl_mat['b'] == 0)

        mask_c = np.logical_and(np.logical_and(maskintcol1, maskintcol2), maskintcol3)
        pcl_mat = pcl_mat[mask_c]
        print("CHECKPOINT 1.2 color filter pcl_mat shape: ", pcl_mat.shape)

        # SSG edit
        pcl_mat_xyzrgb = self.pcl_mat_to_XYZRGB(pcl_mat)


        # print("CHECKPOINT 1.3 pcl_mat_xyzrgb shape: ", pcl_mat_xyzrgb.shape)

        # Distance filter SR edit
        # distances = np.linalg.norm(pcl_mat_xyzrgb[:, :3], axis=1)
        # max_distance = 3
        # maskd = distances <= max_distance

        filtered_pcl_mat = pcl_mat_xyzrgb

        # print("CHECKPOINT 1.4 distance filter pcl_mat_xyzrgb shape: ", filtered_pcl_mat.shape)

        t0_dbscan = time.time()

        db = DBSCAN(eps=0.085, min_samples=5, n_jobs=-1)
        for _ in tqdm(range(10), desc='Clustering'):
            db.fit(filtered_pcl_mat[:, :3])

        print("DBSCAN computation time:", time.time() - t0_dbscan)

        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print('CHECKPOINT 2.1 Estimated number of clusters after DBSCAN: %d' % n_clusters)
        xyz_cl = np.c_[filtered_pcl_mat, labels]  # will be n x 7 cols array
        # print("CHECKPOINT 5 xyz_cl shape: ", xyz_cl.shape)

        # de-noises data
        maskint2 = (xyz_cl[:, -1] == -1)
        xyz_cl = xyz_cl[np.logical_not(maskint2)]

        # Count the number of occurrences of each unique label
        lbls, counts = np.unique(xyz_cl[:, -1], return_counts=True)
        mask = np.isin(xyz_cl[:, -1], lbls[counts > 100])
        # xyz_cl = xyz_cl[~mask]
        # print("CHECKPOINT 6 xyz_cl shape: ", xyz_cl.shape)

        DBSCAN_labels = np.unique(xyz_cl[:, -1])
        print("DBSCAN labels:", len(DBSCAN_labels))


        # # for label in DBSCAN_labels:
        # #     new_label = random.randint(0, 30)
        # #     while new_label in self.assigned_labels:
        # #         new_label = random.randint(0, 30)

        # #     xyz_cl[xyz_cl[:, -1] == label, -1] = new_label
        # #     self.assigned_labels.append(new_label)

        # # new_labels = np.unique(xyz_cl[:,-1])

        # # print("Newly assgined labels:", new_labels)

        # t0 = time.time()
        # cluster_gaussian_stats = self.compute_cluster_stats(xyz_cl)
        # print("Completed Gaussian statistic computation on clusters in time",
        #       time.time() - t0)

        # print("CHECKPOINT 2.2 cluster_gaussian_stats : ",
        #       len(cluster_gaussian_stats))

        # # Use SVD to compare covariance structure of clusters with reference dandelion
        # similar_clusters = self.compare_covariances_SVD(cluster_gaussian_stats)
        # similar_clusters_labels = np.array(similar_clusters)

        # print("CHECKPOINT 3 similar_clusters : ", similar_clusters_labels, "number:", len(similar_clusters_labels))

        # # ##############################################################################################

        # cluster_labels = similar_clusters_labels

        # selected_points = np.concatenate([xyz_cl[xyz_cl[:, -1] == label] for label in cluster_labels], axis=0)

        # mean_values = {}
        # for label in similar_clusters_labels:
        #     if label in cluster_gaussian_stats:
        #         mean_values[label] = cluster_gaussian_stats[label]['mean']

        # mean_points = np.array(list(mean_values.values()))

        # print(mean_points.shape)

        pcl_unfiltered = self.XYZRGB_to_pcl_mat(
            xyz_cl)

        # pcl_unfiltered['r'] = 255
        # pcl_unfiltered['g'] = 0
        # pcl_unfiltered['b'] = 0

        pc_array_unfiltered = merge_rgb_fields(pcl_unfiltered)
        pc_msg_unfiltered = ros_numpy.msgify(
            PointCloud2, pc_array_unfiltered, stamp=pcl_msg.header.stamp, frame_id=pcl_msg.header.frame_id)
        self.pclpublisher.publish(pc_msg_unfiltered)


if __name__ == '__main__':
    pcl_node = PointcloudProcessor()
    rospy.spin()
