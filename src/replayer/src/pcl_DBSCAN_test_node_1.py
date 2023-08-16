#!/usr/bin/env python3
import rospy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

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

class PointcloudProcessor():
    def __init__(self):

        rospy.init_node('pcl_DBSCAN_test_processor', anonymous=True)

        pcl_sub = rospy.Subscriber('/camera/depth/color/points', PointCloud2, self.pcl_transformer)
        # pcl_sub_velodyne = rospy.Subscriber('/velodyne_points_filtered', PointCloud2, self.pcl_transformer)
        self.pclpublisher = rospy.Publisher('transformed_Ypcl', PointCloud2, latch=True, queue_size=100)
        self.velodyne_pcl_publisher = rospy.Publisher('velodyne_filtered_topic', PointCloud2, latch=True, queue_size=100)

        
    def pcl_callback(self, msg):
        print("Received a PointCloud2 message at time: ", msg.header.stamp)
 
    
    def fit_gmm_EM(self, pcl_mat, n_components):

        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        gmm.fit(pcl_mat[:, :3])
        means = gmm.means_
        covariances = gmm.covariances_

        return means, covariances
    
    def pcl_matrix_converter(self, point_cloud_msg):

        pc = ros_numpy.numpify(point_cloud_msg)
        points=np.zeros((pc.shape[0],4))
        points[:,0]=pc['x']
        points[:,1]=pc['y']
        points[:,2]=pc['z']
        points[:,3]=pc['rgb'] 
  
        return points

    def generate_sphere(self, radius, center, n_points):
        # Generate random angles
        theta = 2.0 * np.pi * np.random.rand(n_points)  # polar angle
        phi = np.arccos(2.0 * np.random.rand(n_points) - 1.0)  # azimuthal angle

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
        yellow_mask = (reds <= 210 ) & (greens <= 210) & (blues >= 130)
        
        return yellow_mask
         
    def compute_cluster_stats(self, xyz_cl):
        cluster_labels = np.unique(xyz_cl[:, 4])
        cluster_stats = {}

        for label in cluster_labels:
            if label == -1:
                continue  # Skip noise points

            cluster_points = xyz_cl[xyz_cl[:, 4] == label][:, :3]
            cluster_mean = np.mean(cluster_points, axis=0)
            cluster_cov = np.cov(cluster_points, rowvar=False)

            cluster_stats[label] = {'mean': cluster_mean, 'covariance': cluster_cov}

        return cluster_stats
    
    # def rgb_to_hsv(self,rgb_values):
    #     # Normalize RGB values to the [0,1] range
    #     rgb_values_normalized = rgb_values / 255.0

    #     # Convert RGB to HSV
    #     hsv_values = mcolors.rgb_to_hsv(rgb_values_normalized)

    #     return hsv_values
    
    # def get_yellow_mask(self, rgb_values):
    #     hsv_values = self.rgb_to_hsv(rgb_values)

    #     hues = hsv_values[:, 0]
    #     saturations = hsv_values[:, 1]
    #     values = hsv_values[:, 2]

    #     yellow_mask = ((hues >= 0.1) & (hues <= 0.2)) & (saturations >= 0.7) & (values >= 0.7)

    #     return yellow_mask

    def downsample_voxel_grid(self, pcl_mat_xyzrgb):
        # Convert the XYZRGB point cloud to Open3D format
        pcl_cloud = o3d.geometry.PointCloud()
        pcl_cloud.points = o3d.utility.Vector3dVector(pcl_mat_xyzrgb[:, :3])
        pcl_cloud.colors = o3d.utility.Vector3dVector(pcl_mat_xyzrgb[:, 3:6] / 255.0)

        # Create a voxel grid filter object
        voxel_grid = pcl_cloud.voxel_down_sample(voxel_size=0.001)  # Adjust the voxel size as needed

        # Perform downsampling
        downsampled_cloud = voxel_grid

        # Convert the downsampled point cloud back to numpy array format
        downsampled_pcl_mat_xyzrgb = np.concatenate([np.asarray(downsampled_cloud.points),
                                                    np.asarray(downsampled_cloud.colors) * 255], axis=1)

        return downsampled_pcl_mat_xyzrgb

    def pcl_transformer(self, pcl_msg):
    # try:
        # print("TEST RUN USING /map topic")
        print("Successfully received pcl_msg at time: ", pcl_msg.header.stamp)
        pcl_mat = self.pcl_matrix_converter(pcl_msg)
        # _, yellow_indices = self.pcl_mat_generator(pcl_msg)
        

        min_z = np.min(pcl_mat[:,2])
        max_z = np.max(pcl_mat[:,2])

        # print("Min z-coordinate: ", min_z)
        # print("Max z-coordinate: ", max_z)
        z_coords = pcl_mat[:,2]

        # Save z-coordinates to a file
        # np.savetxt('z_coords.txt', z_coords)


        maskint1 = ((pcl_mat[:,2] < 0.14) | (pcl_mat[:,2] > 0.38))
        # print("maskint1_shape",maskint1.shape)

        # print("pcl_mat_shape_before",pcl_mat.shape)
        pcl_mat = pcl_mat[np.logical_not(maskint1)]
        # print("pcl_mat_shape_after",pcl_mat.shape)
        t0_dbscan = time.time()
        db = DBSCAN(eps=0.01, min_samples=10, n_jobs=-1)
        for _ in tqdm(range(10), desc='Clustering'):
            db.fit(pcl_mat[:,:3])
        t1_dbscan = time.time()
        # print("DBSCAN computation:", t1_dbscan - t0_dbscan)
        
        labels = db.labels_
        n_clusters_before = len(set(labels)) - (1 if -1 in labels else 0)
        print('Estimated number of clusters before filtering: %d' % n_clusters_before)
        xyz_cl = np.c_[pcl_mat, labels]
        # self.visualize_clusters(xyz_cl)
        
        maskint2 = (xyz_cl[:, 4] == -1)
        xyz_cl = xyz_cl[np.logical_not(maskint2)]

        lbls, counts = np.unique(xyz_cl[:, 4], return_counts=True)
        print("uniquelabels_before_mask",lbls)
        print("counts_of_unique_labels_before_mask",counts)


        mask = np.isin(xyz_cl[:, 4], lbls[counts < 500])
        xyz_cl = xyz_cl[~mask]
        lbls1, counts1 = np.unique(xyz_cl[:, 4], return_counts=True)
        print("uniquelabels_after_mask",lbls1.shape)
        print("counts_of_unique_labels_after mask",counts1)

        xyz_cl_rgb = self.split_rgb(xyz_cl)
        yellow_mask = self.get_yellow_mask(xyz_cl_rgb)
        yellow_points = xyz_cl[yellow_mask]

        print("color filtered point shape",yellow_points.shape)

        transformed_pcl = xyz_cl

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

        pc_msg = ros_numpy.msgify(PointCloud2, pc_array, stamp=pcl_msg.header.stamp, frame_id=pcl_msg.header.frame_id)
        self.pclpublisher.publish(pc_msg)
        # except Exception as e:
        #     print("Error in pcl_transformer: ", e)
        #     traceback.print_exc()

if __name__ == '__main__':
    pcl_node = PointcloudProcessor()
    rospy.spin()

