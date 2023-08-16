#!/usr/bin/env python3
import rospy
import numpy as np
from scipy.linalg import logm
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from pyransac3d import Sphere
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MeanShift
import pcl

import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
import std_msgs
import ros_numpy
import message_filters
import tf.transformations as tr
import ctypes
import struct
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
import time
import pickle
from ros_numpy.point_cloud2 import pointcloud2_to_array, array_to_pointcloud2, split_rgb_field, merge_rgb_fields
from scipy.spatial import KDTree
from sklearn.decomposition import PCA


class PointcloudProcessor():
    def __init__(self):

        rospy.init_node('pcl_refrence_feature', anonymous=True)

        # pcl_sub = rospy.Subscriber(
        #     '/camera/depth/color/points', PointCloud2, self.pcl_transformer)
        # self.pcl_publisher = rospy.Publisher(
        #     'cluster_map', PointCloud2, latch=True, queue_size=100)
        pcl_sub_velodyne = rospy.Subscriber('/velodyne_points_filtered', PointCloud2, self.pcl_transformer)
        self.velodyne_pcl_publisher = rospy.Publisher('velodyne_filtered_topic', PointCloud2, latch=True, queue_size=100)

    def pcl_callback(self, msg):
        print("Received a PointCloud2 message at time: ", msg.header.stamp)

    def fit_gmm_EM(self, pcl_mat, n_components):

        gmm = GaussianMixture(n_components=n_components,
                              covariance_type='full', random_state=42)
        gmm.fit(pcl_mat[:, :3])
        means = gmm.means_
        covariances = gmm.covariances_

        return means, covariances

    def pcl_matrix_converter(self, point_cloud_msg):

        pc = ros_numpy.numpify(point_cloud_msg)
        points = np.zeros((pc.shape[0], 4))
        points[:, 0] = pc['x']
        points[:, 1] = pc['y']
        points[:, 2] = pc['z']
        points[:, 3] = pc['rgb']

        return points

    def split_rgb(self, pcl_mat):
        num_points = pcl_mat.shape[0]
        rgb_values = np.zeros((num_points, 3), dtype=np.uint8)

        rgb_packed = pcl_mat[:, 3].copy().view(np.int32)
        for i in range(num_points):
            rgb_values[i, 0] = (rgb_packed[i] >> 16) & 255  # Red channel
            rgb_values[i, 1] = (rgb_packed[i] >> 8) & 255  # Green channel
            rgb_values[i, 2] = rgb_packed[i] & 255  # Blue channel

        return rgb_values

    def compute_cluster_stats(self, xyz_cl):
        cluster_labels = np.unique(xyz_cl[:, 4])
        cluster_stats = {}

        for label in cluster_labels:
            if label == -1:
                continue  # Skip noise points

            cluster_points = xyz_cl[xyz_cl[:, 4] == label][:, :3]
            cluster_mean = np.mean(cluster_points, axis=0)
            cluster_cov = np.cov(cluster_points, rowvar=False)

            cluster_stats[label] = {
                'mean': cluster_mean, 'covariance': cluster_cov}

        return cluster_stats

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

    def compute_features(self, pcl):
        # This function accepts a list of point clouds and returns a matrix where each row is the feature vector of a point cloud.
        feature_list = []

        for point_cloud in pcl:
            # Calculate centroid
            centroid = np.mean(point_cloud, axis=0)

            # Calculate normalized variance
            variance = np.var(point_cloud, axis=0)
            normalized_variance = variance / np.sum(variance)

            # Combine features
            features = np.hstack([centroid, normalized_variance])
            feature_list.append(features)

        return np.array(feature_list)
    
    def compute_normals_and_curvature(self, pcl_mat):
        """
        Compute normals and curvature for each point in the point cloud.
        """
        normals = []
        curvatures = []

        # Build a KD-Tree for efficient nearest neighbors search
        tree = KDTree(pcl_mat[:, :3])

        # Iterate over all points
        for i in range(pcl_mat.shape[0]):
            # Find the nearest neighbors (e.g., 10 nearest neighbors)
            indices = tree.query(pcl_mat[i, :3], k=5)[1]

            # Fit a plane to the nearest neighbors
            plane = PCA(n_components=3).fit(pcl_mat[indices, :3])

            # The normal of the plane is the eigenvector corresponding to the smallest eigenvalue
            normal = plane.components_[-1]
            normals.append(normal)

            # The curvature is the ratio of the smallest eigenvalue to the sum of eigenvalues
            curvature = plane.explained_variance_[-1] / np.sum(plane.explained_variance_)
            curvatures.append(curvature)

        normals = np.array(normals)
        curvatures = np.array(curvatures)

        return normals, curvatures

    def segment_by_curvature(self, pcl_mat):
        """
        Segment point cloud by curvature.
        """
        _, curvatures = self.compute_normals_and_curvature(pcl_mat)

        # Define curvature threshold (e.g., mean curvature)
        threshold = np.mean(curvatures)

        # Segment points with curvature greater than the threshold
        pcl_segmented = pcl_mat[curvatures > threshold]

        return pcl_segmented
    
    # def apply_sor_filter(self, pcl_mat, mean_k=50, std_dev=1.0):
    #     """
    #     Apply the Statistical Outlier Removal (SOR) filter to the input point cloud.
    #     mean_k: number of neighboring points to use for mean distance estimation.
    #     std_dev: standard deviation threshold. 
    #     """

    #     # Create a cloud object and populate it with the point cloud data.
    #     cloud = pcl.PointCloud()
    #     cloud.from_array(pcl_mat.astype(np.float32))

    #     # Create a SOR filter object.
    #     sor = cloud.make_statistical_outlier_filter()

    #     # Set the number of neighboring points to use for mean distance estimation.
    #     sor.set_mean_k(mean_k)

    #     # Set the standard deviation multiplier threshold. All points whose mean distances are X standard deviation away from the mean will be classified as outliers.
    #     sor.set_std_dev_mul_thresh(std_dev)

    #     # Call the filter function.
    #     cloud_filtered = sor.filter()

    #     # Convert the filtered cloud to a numpy array and return it.
    #     filtered_pcl_mat = cloud_filtered.to_array()
    
    #     return filtered_pcl_mat

    

    def pcl_transformer(self, pcl_msg):

        print("Successfully received pcl_msg at time: ", pcl_msg.header.stamp)
        # convert pointcloud to array and RGB split SSG
        pcl_xyzcol = pointcloud2_to_array(pcl_msg)
        pcl_arr = split_rgb_field(pcl_xyzcol)

        print("CHECKPOINT 1 pcl_mat shape: ", pcl_arr.shape)
        # pcl_mat = self.pcl_matrix_converter(pcl_msg)
        # maskint1 = ((pcl_arr[:, 2] < 0.14) | (pcl_arr[:, 2] > 0.38))
        # Heigth Mask
        maskint1 = ((pcl_arr['z'] < -0.005) | (pcl_arr['z'] > 0.30))
        pcl_arr = pcl_arr[np.logical_not(maskint1)]

        print("CHECKPOINT 2 pcl_mat shape: ", pcl_arr.shape)

        # Color Mask SSG
        maskintcol1 = (pcl_arr['r'] <= 210)
        maskintcol2 = (pcl_arr['g'] <= 210)
        maskintcol3 = (pcl_arr['b'] >= 110)
        pcl_arr = pcl_arr[np.logical_not(np.logical_and(
            maskintcol1, maskintcol2, maskintcol3))]
        print("CHECKPOINT color pcl_mat shape: ", pcl_arr.shape)

        # Convert to XYZRGB (N,6) SSG
        pcl_mat = self.pcl_mat_to_XYZRGB(pcl_arr)
        

        # pcl_mat[:, :3] = pcl_mat[:, :3].astype(int)

        # # Assuming that the color values in pcl_mat are in the range [0, max_val]
        max_val = pcl_mat[:, 3:].max()

        pcl_mat[:, 3:] = (pcl_mat[:, 3:] / max_val) * 255

        # RANSAC Sphere fitting
        sphere = Sphere()
        center, radius, _ = sphere.fit(
            pcl_mat[:, :3], thresh=0.005)  # Adjust this value

        threshold = 5  # Adjust this value

        distances = np.linalg.norm(pcl_mat[:, :3] - center, axis=1) - radius
        pcl_mat = pcl_mat[np.abs(distances) < threshold]
        print("SPHERE POINTS SHPAE", pcl_mat.shape)

        # scaler = StandardScaler()
        # pcl_cluster = scaler.fit_transform(pcl_mat)

        print("CHECKPOINT 3 pcl_cluster: ", pcl_mat.shape,pcl_mat.shape)

        # DBSCAN clustering
        color_mat = pcl_mat[:, :3]
        db = DBSCAN(eps=0.025, min_samples=18, n_jobs=-1)  # Adjust this value
        for _ in tqdm(range(10), desc='Clustering'):
            db.fit(color_mat)

        # mean_shift = MeanShift(bandwidth=5,bin_seeding=True,min_bin_freq=10,cluster_all=False,n_jobs=-1)  # you might need to adjust the bandwidth
        # for _ in tqdm(range(10), desc='Clustering'):
        #     mean_shift.fit(pcl_mat[:, 3:])

        labels = db.labels_
        print("CHECKPOINT 3.2 pcl_mat: ",pcl_mat.shape)

        # Cluster denoising
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print('Estimated number of clusters after mean_shift filtering: %d' % n_clusters)
        xyz_cl = np.c_[pcl_mat, labels]

        print("CHECKPOINT 4 xyz_cl shape: ", xyz_cl.shape)

        # Count the number of occurrences of each unique label before Mask
        lbls, counts = np.unique(xyz_cl[:, 6], return_counts=True)
        # print("unique label before point mask: ", lbls)
        # print("unique label count before point mask: ", counts)

        # Pointcloud count filtering
        mask = np.isin(xyz_cl[:, 6], lbls[counts < 1000])
        xyz_cl = xyz_cl[~mask]
        print("CHECKPOINT 6 xyz_cl shape: ", xyz_cl.shape)

        # Count the number of occurrences of each unique label after Mask
        lbls, counts = np.unique(xyz_cl[:, 6], return_counts=True)
        print("unique label after point mask: ", lbls)
        # print("unique label count after point mask: ", counts)

        # cluster_label = 1
        # xyz_cl_cluster2 = xyz_cl[xyz_cl[:, 6] == cluster_label]
        # pc_array_cluster2 = merge_rgb_fields(self.XYZRGB_to_pcl_mat(xyz_cl_cluster2[:,:6]))

        cluster_labels = np.unique(xyz_cl[:, 6])

        # Feature comparision using refrence model for each cluster

        # with open('reference_model.pkl', 'rb') as f:
        #     ref_model = pickle.load(f)

        # _mat = xyz_cl[:, :3]
        # db = DBSCAN(eps=0.025, min_samples=18, n_jobs=-1)  # Adjust this value
        # for _ in tqdm(range(10), desc='Clustering'):
        #     db.fit(xyz_cl[:, :3])
        
        # normals, curvatures = self.compute_normals_and_curvature(xyz_cl)
        # pcl_segmented = self.segment_by_curvature(xyz_cl)
        # print("CHECKPOINT 7 xyz_cl shape: ", pcl_segmented.shape)
        filtered_clusters = []

        for label in cluster_labels:
            if label == -1:
                continue

            label_int = label.astype(int)
            cluster_data = xyz_cl[xyz_cl[:, 6] == label_int]
            # maskintcolR = (cluster_data[:,3]  <= 255) & (cluster_data[:,3] >= 144)
            # maskintcolG = (cluster_data[:,4]  <= 255) & (cluster_data[:,4] >= 121)
            # maskintcolB = (cluster_data[:,5]  <= 239) & (cluster_data[:,5] >= 0)
            # # maskintcolZ = (cluster_data[:,1]  <= 0.05)

            # combined_mask = maskintcolR & maskintcolG & maskintcolB 
            cluster_data  = self.apply_sor_filter(cluster_data , mean_k=50, std_dev=1.0)
            print("CHECKPOINT 9 final_pcl shape", cluster_data.shape)


            # filtered_cluster_data = cluster_data[combined_mask]

            filtered_clusters.append(cluster_data )
            # maskintcolG = (pcl_arr['g'] <= 255)
            # maskintcolB = (pcl_arr['b'] >= 110)

        # for label in cluster_labels:
        #     if label == -1:
        #         continue

        #     label_int = label.astype(int)
        #     cluster_data = xyz_cl[xyz_cl[:, 6] == label_int]
        #     sphere = Sphere()
        #     center, radius, _ = sphere.fit(
        #         cluster_data[:, :3], thresh=0.002, maxIteration=100)  # Adjust this value

        #     threshold = 0.009  # Adjust this value

        #     distances = np.linalg.norm(cluster_data[:, :3] - center, axis=1) - radius
        #     cluster_data = cluster_data[np.abs(distances) < threshold]


        #     filtered_clusters.append(cluster_data)
        #     # maskintcolG = (pcl_arr['g'] <= 255)
        #     # maskintcolB = (pcl_arr['b'] >= 110)

        # final_pcl = np.concatenate(filtered_clusters, axis=0)
        final_pcl = xyz_cl
        # print("Green max:", np.max(final_pcl[:,4]) ,"Green min value:", np.min(final_pcl[:,4]) )
        # print("Z max:", np.max(final_pcl[:,1]) ,"Z min value:", np.min(final_pcl[:,1]) )

        print("CHECKPOINT 10 final_pcl shape", final_pcl.shape)

        # Convert matrix to message SSG
        pc_array = merge_rgb_fields(self.XYZRGB_to_pcl_mat(final_pcl[:, :6]))
        # PointCloud2 message header
        pc_msg = ros_numpy.msgify(
            PointCloud2, pc_array, stamp=pcl_msg.header.stamp, frame_id=pcl_msg.header.frame_id)

        self.velodyne_pcl_publisher.publish(pc_msg)


if __name__ == '__main__':
    pcl_node = PointcloudProcessor()
    rospy.spin()
