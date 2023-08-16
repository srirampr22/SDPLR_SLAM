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
from datetime import datetime
from scipy.spatial.distance import euclidean, cdist
import colorsys
from visualization_msgs.msg import Marker, MarkerArray


class PointcloudProcessor():
    def __init__(self):
        rospy.init_node('pcl_processor', anonymous=True)
        rospy.Subscriber('/map', PointCloud2, self.pcl_cluster_meancov)

        self.pclpublisher = rospy.Publisher(
            'dandelion_map', PointCloud2, latch=True, queue_size=100)
        self.marker_publisher = rospy.Publisher(
            'cluster_labels', MarkerArray, queue_size=10)

        self.pcl_buffer = {}
        self.true_buffer = {}
        self.accumulated_points = []
        self.accumulated_label = []
        self.assigned_labels = []

    def publish_cluster_labels(self, mean_values):
        marker_array = MarkerArray()

        for index, (label, mean) in enumerate(mean_values.items()):
            marker = Marker()
            marker.header.frame_id = "map"  # Use your frame here
            marker.id = index
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD
            marker.pose.position.x = mean[0]
            marker.pose.position.y = mean[1]
            # Adding a small offset in z direction to display above the cube
            marker.pose.position.z = mean[2] + 0.06
            marker.text = str(label)
            marker.scale.z = 0.04  # Size of the text
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker_array.markers.append(marker)

        self.marker_publisher.publish(marker_array)

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

    def create_cube_outline(self, center, size=0.05):
        """
        Given a center point, compute the cube's outline centered at that point.
        :param center: (numpy array) 1x3 array containing XYZ coordinates of the center point.
        :param size: (float) side length of the cube.
        :return: (numpy array) Nx3 array containing XYZ coordinates of the cube's outline.
        """
        half_size = size / 2.0

        # Define all 8 vertices of the cube
        vertices = [
            center + np.array([half_size, half_size, half_size]),
            center + np.array([-half_size, half_size, half_size]),
            center + np.array([-half_size, -half_size, half_size]),
            center + np.array([half_size, -half_size, half_size]),
            center + np.array([half_size, half_size, -half_size]),
            center + np.array([-half_size, half_size, -half_size]),
            center + np.array([-half_size, -half_size, -half_size]),
            center + np.array([half_size, -half_size, -half_size]),
        ]

        # Define the 12 edges of the cube using pairs of vertices
        edges = [
            (vertices[0], vertices[1]),
            (vertices[1], vertices[2]),
            (vertices[2], vertices[3]),
            (vertices[3], vertices[0]),
            (vertices[4], vertices[5]),
            (vertices[5], vertices[6]),
            (vertices[6], vertices[7]),
            (vertices[7], vertices[4]),
            (vertices[0], vertices[4]),
            (vertices[1], vertices[5]),
            (vertices[2], vertices[6]),
            (vertices[3], vertices[7]),
        ]

        outline_points = []

        # For each edge, generate points to represent that edge
        for edge in edges:
            # The following code linearly interpolates between the start and end of each edge
            num_points = int(size / 0.0005)  # For every 5 mm
            for i in range(num_points + 1):
                point = edge[0] * (1 - i/num_points) + edge[1] * (i/num_points)
                outline_points.append(point)

        return np.array(outline_points)

    def pcl_cluster_meancov(self, pcl_msg):

        pcl_xyzcol = pointcloud2_to_array(pcl_msg)
        pcl_mat = split_rgb_field(pcl_xyzcol)

        maskintcol1 = (pcl_mat['r'] == 0)
        maskintcol2 = (pcl_mat['g'] == 255)
        maskintcol3 = (pcl_mat['b'] == 0)

        mask_c = np.logical_and(np.logical_and(
            maskintcol1, maskintcol2), maskintcol3)
        pcl_mat = pcl_mat[mask_c]

        pcl_mat_xyzrgb = self.pcl_mat_to_XYZRGB(pcl_mat)

        filtered_pcl_mat = pcl_mat_xyzrgb

        if (filtered_pcl_mat.shape[0] > 0):

            t0_dbscan = time.time()

            db = DBSCAN(eps=0.02, min_samples=5, n_jobs=-1)
            for _ in tqdm(range(10), desc='Clustering'):
                db.fit(filtered_pcl_mat[:, :3])

            print("DBSCAN computation time:", time.time() - t0_dbscan)

            labels = db.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            # will be n x 7 cols array
            xyz_cl = np.c_[filtered_pcl_mat, labels]
            # de-noises data
            maskint2 = (xyz_cl[:, -1] == -1)
            xyz_cl = xyz_cl[np.logical_not(maskint2)]

            if (xyz_cl.shape[0] > 0):

                DBSCAN_labels, DBSCAN_counts = np.unique(
                    xyz_cl[:, -1], return_counts=True)
                # print("Estimated number of clusters after DBSCAN",
                #       DBSCAN_labels, "xyz_cl shape:", xyz_cl.shape )
                # print("counts:", DBSCAN_counts)

                for label in DBSCAN_labels:
                    new_label = random.randint(0, 30)
                    while new_label in self.assigned_labels:
                        new_label = random.randint(0, 30)

                    xyz_cl[xyz_cl[:, -1] == label, -1] = new_label
                    self.assigned_labels.append(new_label)

                new_labels = np.unique(xyz_cl[:,-1])

                print("Newly assgined labels:", new_labels, "xyz_cl shape:", xyz_cl.shape)

                cluster_gaussian_stats = self.compute_cluster_stats(xyz_cl)

                ######################################### BUFEER UPDATE##############################################
                bucket = []

                buffer_entry = {}
                buffer_entry['labels'] = DBSCAN_labels.tolist()
                buffer_entry['means'] = []
                buffer_entry['points'] = []

                for label in DBSCAN_labels:
                    if label in cluster_gaussian_stats:
                        cluster_mean = cluster_gaussian_stats[label]['mean']
                        cluster_points = xyz_cl[xyz_cl[:, -1] == label]

                        buffer_entry['means'].append(cluster_mean.tolist())
                        buffer_entry['points'].append(cluster_points.tolist())

                # Store the buffer entry with timestamp (t = x)
                self.pcl_buffer[pcl_msg.header.stamp] = buffer_entry

                # Most recent previous timestamp (t = x-1)
                prev_time_stamp = None
                for timestamp in self.pcl_buffer.keys():
                    if timestamp < pcl_msg.header.stamp:
                        if prev_time_stamp is None or timestamp > prev_time_stamp:
                            prev_time_stamp = timestamp

                if prev_time_stamp is not None:

                    # prev is t-1
                    prev_buffer_entry = self.pcl_buffer[prev_time_stamp]
                    prev_true_buffer = self.true_buffer[prev_time_stamp]
                    curr_buffer_entry = self.pcl_buffer[pcl_msg.header.stamp]

                    prev_buffer_entry = self.pcl_buffer[prev_time_stamp]
                    prev_cluster_labels = prev_buffer_entry['labels']
                    prev_cluster_mean = prev_buffer_entry['means']
                    prev_cluster_points = prev_buffer_entry['points']

                else:
                    bucket.append(xyz_cl)

                accumulated_points = np.concatenate(bucket, axis=0)

                acc_lbls, acc_counts = np.unique(accumulated_points[:, -1], return_counts=True)

                print("Accumulated points labels:", acc_lbls)
                #####################################################################################################

                mean_values = {}
                for label in new_labels:
                    if label in cluster_gaussian_stats:
                        mean_values[label] = cluster_gaussian_stats[label]['mean']

                # List to store the cube vertices
                bounding_cubes = []

                # Create a cube for each mean value
                for _, mean in mean_values.items():
                    bounding_cubes.extend(self.create_cube_outline(mean))

                self.publish_cluster_labels(mean_values)

                # Convert the cube vertices to the same point cloud format
                bounding_cubes = np.array(bounding_cubes)
                pcl_cubes = np.zeros((len(bounding_cubes),),
                                     dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32),
                                            ('r', np.uint8), ('g', np.uint8), ('b', np.uint8)])
                pcl_cubes['x'] = bounding_cubes[:, 0]
                pcl_cubes['y'] = bounding_cubes[:, 1]
                pcl_cubes['z'] = bounding_cubes[:, 2]
                pcl_cubes['r'] = 255
                pcl_cubes['g'] = 0
                pcl_cubes['b'] = 127

                pcl_unfiltered = self.XYZRGB_to_pcl_mat(
                    xyz_cl)

                # Merge cube vertices with the original point cloud data
                pcl_with_cubes = np.concatenate([pcl_unfiltered, pcl_cubes])
                pc_array_unfiltered = merge_rgb_fields(pcl_with_cubes)
                pc_msg_unfiltered = ros_numpy.msgify(
                    PointCloud2, pc_array_unfiltered, stamp=pcl_msg.header.stamp, frame_id=pcl_msg.header.frame_id)
                self.pclpublisher.publish(pc_msg_unfiltered)
            else:
                print("No clusters found")
        else:
            print("No clusters found")


if __name__ == '__main__':
    pcl_node = PointcloudProcessor()
    rospy.spin()
