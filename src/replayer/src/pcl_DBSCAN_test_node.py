#!/usr/bin/env python3
import rospy
import numpy as np
from scipy.linalg import logm
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from pyransac3d import Sphere

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
import open3d as o3d
from ros_numpy.point_cloud2 import pointcloud2_to_array, array_to_pointcloud2, split_rgb_field, merge_rgb_fields


class PointcloudProcessor():
    def __init__(self):

        rospy.init_node('pcl_DBSCAN_test_processor', anonymous=True)

        pcl_sub = rospy.Subscriber(
            '/map', PointCloud2, self.pcl_transformer)
        # pcl_sub_velodyne = rospy.Subscriber('/velodyne_points_filtered', PointCloud2, self.pcl_transformer)
        self.pclpublisher = rospy.Publisher(
            'transformed_Ypcl', PointCloud2, latch=True, queue_size=100)
        self.downsample_pcl_publisher = rospy.Publisher(
            'downsampled_topic', PointCloud2, latch=True, queue_size=100)

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
        # xyzrgb = np.zeros((pc.shape[0], 6))
        # xyzrgb[:, 0] = pc['x']
        # xyzrgb[:, 1] = pc['y']
        # xyzrgb[:, 2] = pc['z']
        # xyzrgb[:, 3] = pc['r']
        # xyzrgb[:, 4] = pc['g']
        # xyzrgb[:, 5] = pc['b']

        return points

    def generate_sphere(self, radius, center, n_points):
        # Generate random angles
        theta = 2.0 * np.pi * np.random.rand(n_points)  # polar angle
        phi = np.arccos(2.0 * np.random.rand(n_points) -
                        1.0)  # azimuthal angle

        # Convert the spherical coordinates to Cartesian coordinates
        x = radius * np.sin(phi) * np.cos(theta) + center[0]
        y = radius * np.sin(phi) * np.sin(theta) + center[1]
        z = radius * np.cos(phi) + center[2]

        return x, y, z

    def split_rgb(self, pcl_mat):
        num_points = pcl_mat.shape[0]
        rgb_values = np.zeros((num_points, 3), dtype=np.uint8)

        rgb_packed = pcl_mat[:, 3].copy().view(np.int32)
        for i in range(num_points):
            rgb_values[i, 0] = (rgb_packed[i] >> 16) & 255  # Red channel
            rgb_values[i, 1] = (rgb_packed[i] >> 8) & 255  # Green channel
            rgb_values[i, 2] = rgb_packed[i] & 255  # Blue channel

        return rgb_values

    def get_yellow_mask(self, rgb_values):
        reds = rgb_values[:, 0]
        greens = rgb_values[:, 1]
        blues = rgb_values[:, 2]

        # yellow_mask = ((reds >= 181) & (reds <= 255)) & ((greens >= 166) & (greens <= 255)) & (blues <= 240)
        # yellow_mask = (reds <= 255 ) & (greens <= 255) & (blues <= 150)
        yellow_mask = (reds <= 210) & (greens <= 210) & (blues >= 130)

        return yellow_mask

    def compute_cluster_stats(self, xyz_cl):
        # cluster_labels = np.unique(xyz_cl[:, 4])
        # cluster_stats = {}

        # for label in cluster_labels:
        #     if label == -1:
        #         continue  # Skip noise points

        cluster_points = xyz_cl[:, :3]
        cluster_mean = np.mean(cluster_points, axis=0)
        cluster_cov = np.cov(cluster_points, rowvar=False)

        cluster_stats = {'mean': cluster_mean, 'covariance': cluster_cov}

        return cluster_stats
    
    def compute_frobenius_norm(self, covariance_matrix):
        return np.linalg.norm(covariance_matrix, ord='fro')
    
    def log_euclidean_distance(self, matrix1, matrix2):
        diff_matrix = logm(matrix1) - logm(matrix2)
        return np.linalg.norm(diff_matrix, ord='fro')

    def downsample_voxel_grid(self, pcl_mat_xyzrgb):
        # Convert the XYZRGB point cloud to Open3D format
        pcl_cloud = o3d.geometry.PointCloud()
        pcl_cloud.points = o3d.utility.Vector3dVector(pcl_mat_xyzrgb[:, :3])
        pcl_cloud.colors = o3d.utility.Vector3dVector(pcl_mat_xyzrgb[:, 3:6] / 255.0)

        # Create a voxel grid filter object
        voxel_grid = pcl_cloud.voxel_down_sample(voxel_size=0.0001)  # Adjust the voxel size as needed

        # Perform downsampling
        downsampled_cloud = voxel_grid

        # Convert the downsampled point cloud back to numpy array format
        downsampled_pcl_mat_xyzrgb = np.concatenate([np.asarray(downsampled_cloud.points),
                                                    np.asarray(downsampled_cloud.colors) * 255], axis=1)

        return downsampled_pcl_mat_xyzrgb
    
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


    def pcl_transformer(self, pcl_msg):
        # try:
        # print("TEST RUN USING /map topic")
        print("Successfully received pcl_msg at time: ", pcl_msg.header.stamp)
        pcl_mat = self.pcl_matrix_converter(pcl_msg)
        # # _, yellow_indices = self.pcl_mat_generator(pcl_msg)

        # min_z = np.min(pcl_mat[:, 2])
        # max_z = np.max(pcl_mat[:, 2])

        # z_coords = pcl_mat[:, 2]


        # Save z-coordinates to a file
        # np.savetxt('z_coords.txt', z_coords)

        # maskint1 = ((pcl_mat[:, 2] < 0.14) | (pcl_mat[:, 2] > 0.38))

        # print("pcl_mat_shape_before",pcl_mat.shape)
        # pcl_mat = pcl_mat[np.logical_not(maskint1)]


        # print("pcl_mat_shape_after",pcl_mat.shape)
        t0_dbscan = time.time()
        db = DBSCAN(eps=0.01, min_samples=20, n_jobs=-1)
        for _ in tqdm(range(10), desc='Clustering'):
            db.fit(pcl_mat[:, :3])
        t1_dbscan = time.time()
        # print("DBSCAN computation:", t1_dbscan - t0_dbscan)

        labels = db.labels_
        n_clusters_before = len(set(labels)) - (1 if -1 in labels else 0)
        print('Estimated number of clusters before filtering: %d' %
              n_clusters_before)
        xyz_cl = np.c_[pcl_mat, labels]
        # self.visualize_clusters(xyz_cl)

        maskint2 = (xyz_cl[:, 4] == -1)
        xyz_cl = xyz_cl[np.logical_not(maskint2)]

        lbls, counts = np.unique(xyz_cl[:, 4], return_counts=True)
        print("uniquelabels_before_mask", lbls)
        print("counts_of_unique_labels_before_mask", counts)

        # selected_labels = lbls[np.logical_and(counts > 100, counts <= 150)]
        # selected_labels = 192 #5.1
        selected_labels = 180
        mask = np.isin(xyz_cl[:, 4], selected_labels)
        xyz_cl = xyz_cl[mask]
        lbls1, counts1 = np.unique(xyz_cl[:, 4], return_counts=True)
        print("uniquelabels_after_mask", lbls1)
        print("counts_of_unique_labels_after mask", counts1)


        cluster_gaussian_stats = self.compute_cluster_stats(xyz_cl)

        print("cluster mean and cov:", cluster_gaussian_stats)

        # ref_dandelion_mean = np.array([-0.01268642, -0.05060471,  0.27628341])

        # ref_dandelion_cov = np.array([
        #     [1.57719516e-04, 9.82519920e-07, 7.33407198e-06],
        #     [9.82519920e-07, 1.51163665e-04, 3.57459912e-05],
        #     [7.33407198e-06, 3.57459912e-05, 2.60784083e-05]
        # ])

        # valid_labels = []
        # ref_dandelion_forbnorm = self.compute_frobenius_norm(ref_dandelion_cov)

        # for label, stats in cluster_gaussian_stats.items():
        #     cluster_mean_diff = ref_dandelion_mean - stats['mean']
        #     cluster_cov_diff = ref_dandelion_cov - stats['covariance']

        #     new_cluster_cov = stats['covariance']
        #     new_cluster_formnorm = self.compute_frobenius_norm(stats['covariance'])
        #     ref_dandelion_forbnorm = self.compute_frobenius_norm(ref_dandelion_cov)
        #     # cluster_cov_diff = abs(new_cluster_formnorm - ref_dandelion_forbnorm)
        #     new_cluster_eigen = np.linalg.eigvals(new_cluster_cov)
        #     ref_cluster_eigen = np.linalg.eigvals(ref_dandelion_cov)
        #     # print("NEW_CLUSTER_COV",new_cluster_eigen)
        #     # print("REF_CLUSTER_COV",ref_cluster_eigen)
        #     cluster_cov_diff = abs(self.log_euclidean_distance(new_cluster_cov,ref_dandelion_cov))

        #     if ((cluster_cov_diff) >= 0 & (cluster_cov_diff <= 1)):
        #         valid_labels.append(label)
        #         print(f"Cluster {label} covariance difference from model: {abs(cluster_cov_diff)}")
        #     else:
        #         print("Not dandelion cluster")

        # ###########FORBENIUS NORM######################

        # forb_filtered_cl = xyz_cl[np.isin(xyz_cl[:, 4], valid_labels)]

        # print("forbenius norm filtered point shape", forb_filtered_cl.shape)

        # ############COLOR FILTERING#####################

        # xyz_cl_rgb = self.split_rgb(forb_filtered_cl)
        # yellow_mask = self.get_yellow_mask(xyz_cl_rgb)
        # yellow_points_cl = xyz_cl[yellow_mask]

        # ############RANSDAC SPHERE FITTING #####################

        # sphere = Sphere()
        # center, radius, _ = sphere.fit(xyz_cl[:, :3],thresh=0.00005, maxIteration=100)

        # threshold = 0.001  

        # distances = np.linalg.norm(xyz_cl[:, :3] - center, axis=1) - radius
        # sphere_points_cl = xyz_cl[np.abs(distances) < threshold]
        # print("SPHERE POINTS SHPAE", sphere_points_cl.shape)

        transformed_pcl = xyz_cl[:,:6]

        pc_array = np.zeros(len(transformed_pcl), dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('rgb', np.float32),
        ])
        pc_array['x'] = transformed_pcl[:, 0]
        pc_array['y'] = transformed_pcl[:, 1]
        pc_array['z'] = transformed_pcl[:, 2]
        pc_array['rgb'] = transformed_pcl[:, 3]

        pc_msg = ros_numpy.msgify(
            PointCloud2, pc_array, stamp=pcl_msg.header.stamp, frame_id=pcl_msg.header.frame_id)
        self.pclpublisher.publish(pc_msg)




        # pcl_xyzcol2 = pointcloud2_to_array(pc_msg)
        # pcl_mat2 = split_rgb_field(pcl_xyzcol2)
        # print("CHECKPOINT F1 pcl_mat shape: ", pcl_mat2.shape)
        # pcl_mat2 = self.pcl_mat_to_XYZRGB(pcl_mat2)
        # print("CHECKPOINT F3.2pcl_mat_xyzrgb shape: ", pcl_mat2.shape)

        # # dwn_pclmat2 = self.downsample_voxel_grid(pcl_mat2)
    

        # # pcl_mat2 = dwn_pclmat2
        # print("CHECKPOINT F3.3 pcl_mat_xyzrgb downsampled shape: ", pcl_mat2.shape)
        # print("REF dandelion (voxel_size = 0.0001) mean and covariance:",self.compute_cluster_stats(pcl_mat2))

        # pc_array2 = merge_rgb_fields(self.XYZRGB_to_pcl_mat(pcl_mat2))

        # # Define the PointCloud2 message header
        # pc_msg2 = ros_numpy.msgify(PointCloud2, pc_array2, stamp=pcl_msg.header.stamp, frame_id=pcl_msg.header.frame_id)

        # self.downsample_pcl_publisher.publish(pc_msg2)
        # except Exception as e:
        #     print("Error in pcl_transformer: ", e)
        #     traceback.print_exc()

        


if __name__ == '__main__':
    pcl_node = PointcloudProcessor()
    rospy.spin()
