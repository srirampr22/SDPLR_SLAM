#!/usr/bin/env python3
import rospy
import numpy as np
from tqdm import tqdm
import random
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
import ros_numpy
import tf.transformations as tr
import ctypes
import struct
from sklearn.cluster import DBSCAN
import time
from ros_numpy.point_cloud2 import pointcloud2_to_array, array_to_pointcloud2, split_rgb_field, merge_rgb_fields
from datetime import datetime
from scipy.spatial.distance import euclidean, cdist
from nav_msgs.msg import Odometry


class PointcloudProcessor():
    def __init__(self):
        rospy.init_node('pcl_processor', anonymous=True)
        # rospy.Subscriber('/velodyne_points_filtered', PointCloud2, self.pcl_cluster_meancov)
        rospy.Subscriber('/map', PointCloud2, self.pcl_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)

        self.pclpublisher = rospy.Publisher(
            'filtered_clusters', PointCloud2, latch=True, queue_size=100)
        self.pclpublisher2 = rospy.Publisher(
            'dandelion_map', PointCloud2, latch=True, queue_size=100)

        self.ref_dandelion_cov = np.array([[1.57329671e-05, -4.12004470e-06, -1.80839649e-05],
                                           [-4.12004470e-06,  2.32158495e-04,
                                               2.14231478e-06],
                                           [-1.80839649e-05,  2.14231478e-06,  1.58321861e-04]])
        
        self.odom_publisher = rospy.Publisher('map_odom', Odometry, latch=True, queue_size=100)
        
        self.previous_pose = None
        self.pcl_buffer = {}
        self.odom_msg = None
        self.assigned_labels = []
        self.true_buffer = {}
        self.accumulated_points = []
        self.accumulated_labels = []


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
        red = 0.0
        green = 0.0
        blue = 0.0

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
                S_reference, S_specimen, atol=0.001, rtol=0.0001)

            if eigenvalue_similarity:
                # Append cluster number to similar_clusters list
                similar_clusters.append(cluster_num)

        return similar_clusters  # Return the list of similar clusters
    
    def compare_cluster_positions(self, predicted_positions, actual_cluster_positions, actual_cluster_labels):
        # new_true_clusters = []

        distances = cdist(predicted_positions, actual_cluster_positions, 'euclidean')

        # Find the closest cluster in the current time step which is closest to the predicted_positions
        # If a match is found, flag it as False; otherwise, flag it as True
        correspondences = {}  # Mapping of true labels to closest labels
        duplicate_labels = []  # List to track labels that have already been matched

        if len(actual_cluster_positions) == 1:

                        # Check distance threshold for non-duplicate cluster
            distance_threshold = 0.5  # Adjust the distance threshold as needed
            for curr_idx, curr_mean in enumerate(actual_cluster_positions):
                is_duplicate = False
                for true_label, true_mean in enumerate(predicted_positions):
                    distances = cdist([curr_mean], [true_mean], 'euclidean')
                    if distances[0][0] < distance_threshold:
                        is_duplicate = True
                        print(distances)
                        break
                if is_duplicate == True:
                    print("is it a False clusters: ", is_duplicate)
                    duplicate_labels.append(actual_cluster_labels[curr_idx])
                    correspondences[true_label] = actual_cluster_labels[curr_idx]

        else:

            for curr_idx, curr_label in enumerate(actual_cluster_labels):
                closest_idx = np.argmin(distances[:,curr_idx])
                closest_label = actual_cluster_labels[closest_idx]
                correspondences[curr_label] = closest_label

                duplicate_labels.append(closest_label)

        True_labels = [ label for label in actual_cluster_labels if label not in duplicate_labels]

        return duplicate_labels, True_labels
    
    def get_predicted_cluster_positions(self, relative_transform, prev_cluster_mean):

        predicted_positions = []

        # Get the cluster positions from the previous frame
        previous_cluster_positions = prev_cluster_mean # 

        # Apply the relative transform to the previous cluster positions
        for previous_pos in previous_cluster_positions:
            # Apply the relative transform to each previous position
            previous_pos_homogeneous = np.append(previous_pos, 1)

        # Apply the relative transform to each previous position
            predicted_pos = relative_transform @ previous_pos_homogeneous

            predicted_pos = predicted_pos[:-1]

            # Add the predicted position to the list of predicted positions
            predicted_positions.append(predicted_pos)

        return predicted_positions
    
    def is_duplicate(self, mean, prev_cluster_mean):
        # Check if the mean matches any of the previous cluster means exactly
        for prev_mean in prev_cluster_mean:
            if np.array_equal(mean, prev_mean):
                return True

        return False

    def remove_duplicates(self, curr_cluster_mean, curr_cluster_labels, prev_cluster_mean):
        unique_curr_cluster_mean = []
        unique_curr_cluster_labels = []

        for curr_idx, curr_mean in enumerate(curr_cluster_mean):
            # Check if the mean of the current cluster is unique
            if not self.is_duplicate(curr_mean, prev_cluster_mean):
                unique_curr_cluster_mean.append(curr_mean)
                unique_curr_cluster_labels.append(
                    curr_cluster_labels[curr_idx])

        return np.array(unique_curr_cluster_mean), np.array(unique_curr_cluster_labels)

    
    def odom_callback(self, odom_msg):

        self.odom_msg = odom_msg
        
        current_pose = self.msg_to_se3(odom_msg)

        if self.previous_pose is not None:
            relative_transform = np.linalg.inv(self.previous_pose) @ current_pose
            # Process the relative_transform as needed (e.g., use it for motion-based association)

        # Update previous_pose with current_pose
            return relative_transform
    
        self.previous_pose = current_pose

        self.odom_publisher.publish(odom_msg)


        return None

    
    def pcl_callback(self, pcl_msg):
        # pcl_mat = self.pcl_matrix_converter(pcl_msg)
        pcl_xyzcol = pointcloud2_to_array(pcl_msg)
        pcl_mat = split_rgb_field(pcl_xyzcol)
        print("CHECKPOINT 1 pcl_mat shape: ", pcl_mat.shape)

        maskint1 = ((pcl_mat['z'] < 0.03) | (pcl_mat['z'] > 0.30))
        # maskint1 = ((pcl_mat['z'] < -0.04) | (pcl_mat['z'] > 0.30))
        pcl_mat = pcl_mat[np.logical_not(maskint1)]

        pcl_mat_xyzrgb = self.pcl_mat_to_XYZRGB(pcl_mat)

        distances = np.linalg.norm(pcl_mat_xyzrgb[:, :3], axis=1)
        max_distance = 0.8
        maskd = distances <= max_distance
        filtered_pcl_mat = pcl_mat_xyzrgb[maskd]

        print("CHECKPOINT 1.4 distance filter pcl_mat_xyzrgb shape: ",
              filtered_pcl_mat.shape)

        t0_dbscan = time.time()

        db = DBSCAN(eps=0.0095, min_samples=18, n_jobs=-1)
        for _ in tqdm(range(10), desc='Clustering'):
            db.fit(filtered_pcl_mat[:, :3])

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

        DBSCAN_labels = np.unique(xyz_cl[:, -1])
        print("DBSCAN labels:", len(DBSCAN_labels))

        # for label in DBSCAN_labels:
        #     new_label = random.randint(0, 30)
        #     while new_label in self.assigned_labels:
        #         new_label = random.randint(0, 30)

        #     xyz_cl[xyz_cl[:, -1] == label, -1] = new_label
        #     self.assigned_labels.append(new_label)

        # new_labels = np.unique(xyz_cl[:,-1])

        # print("Newly assgined labels:", new_labels)

        t0 = time.time()
        cluster_gaussian_stats = self.compute_cluster_stats(xyz_cl)

        # Use SVD to compare covariance structure of clusters with reference dandelion
        similar_clusters = self.compare_covariances_SVD(cluster_gaussian_stats)
        similar_clusters_labels = np.array(similar_clusters)

        print("CHECKPOINT 3 similar_clusters : ", similar_clusters_labels,
              "number:", len(similar_clusters_labels))

        cluster_labels = similar_clusters_labels

        selected_points = np.concatenate(
            [xyz_cl[xyz_cl[:, -1] == label] for label in cluster_labels], axis=0)
        

        buffer_entry = {}
        buffer_entry['labels'] = similar_clusters_labels
        buffer_entry['means'] = []
        buffer_entry['points'] = []

        for label in similar_clusters_labels:
            if label in cluster_gaussian_stats:
                cluster_mean = cluster_gaussian_stats[label]['mean']
                cluster_points = xyz_cl[xyz_cl[:, -1] == label]

                buffer_entry['means'].append(cluster_mean.tolist())
                buffer_entry['points'].append(cluster_points.tolist())

        self.pcl_buffer[pcl_msg.header.stamp] = buffer_entry

        if self.odom_msg.header.stamp == pcl_msg.header.stamp:

            relative_transform = self.odom_callback(self.odom_msg)

            # most recent previous timestamp
            prev_time_stamp = None
            for timestamp in self.pcl_buffer.keys():
                if timestamp < pcl_msg.header.stamp:
                    if prev_time_stamp is None or timestamp > prev_time_stamp:
                        prev_time_stamp = timestamp

            if prev_time_stamp is not None:

                ########################## Buffer comparision step##################################

                prev_clusters = self.pcl_buffer[prev_time_stamp]
                true_clusters = self.true_buffer[prev_time_stamp]

                prev_cluster_labels = prev_clusters['labels']
                prev_cluster_mean = prev_clusters['means']
                prev_cluster_points = np.array(prev_clusters['points'], dtype=object)

                true_cluster_labels = true_clusters['labels']
                true_cluster_mean =  true_clusters['means']

                predicted_positions = self.get_predicted_cluster_positions(relative_transform, true_cluster_mean)

                predicted_positions_array = np.array(predicted_positions)

                # print(predicted_positions)

                curr_cluster_mean = buffer_entry['means'] # actual current cluster position
                curr_cluster_labels = buffer_entry['labels']

                unique_cluster_mean, unique_cluster_labels = self.remove_duplicates(curr_cluster_mean,curr_cluster_labels, prev_cluster_mean)

                # print(unique_cluster_mean)

                duplicate_labels, True_labels = self.compare_cluster_positions(predicted_positions, unique_cluster_mean, unique_cluster_labels)

                print(predicted_positions_array.shape)

                mask1 = np.isin(selected_points[:, -1], np.unique(True_labels))
                new_cluster_points = selected_points[mask1]

                self.accumulated_labels.append(True_labels)
                self.accumulated_points.append(new_cluster_points)

                pcl_unfiltered = self.XYZRGB_to_pcl_mat(predicted_positions_array[:, :7])

                pcl_unfiltered['r'] = 255
                pcl_unfiltered['g'] = 0
                pcl_unfiltered['b'] = 0

                pc_array_unfiltered = merge_rgb_fields(pcl_unfiltered)
                pc_msg_unfiltered = ros_numpy.msgify(
                    PointCloud2, pc_array_unfiltered, stamp=pcl_msg.header.stamp, frame_id=pcl_msg.header.frame_id)
                self.pclpublisher.publish(pc_msg_unfiltered)

            else:
                unique_labels = np.unique(selected_points[:,-1])
                self.accumulated_labels.append(np.unique(selected_points[:,-1]))
                self.accumulated_points.append(selected_points)


            accumulated_points = np.concatenate(self.accumulated_points, axis=0)
            # accumulated_points = np.unique(accumulated_points[:,-1])
            accumulated_labels = np.concatenate(self.accumulated_labels, axis=0)
            accumulated_labels = np.unique(accumulated_labels)

            # all the accumulated points get stores in the buffer.
            true_buffer = {}
            true_buffer['labels'] = accumulated_labels.tolist()
            true_buffer['means'] = []
            true_buffer['points'] = []

            for label in accumulated_labels:
                # cluster_mean = cluster_gaussian_stats[label]['mean']
                cluster_points = accumulated_points[accumulated_points[:, -1] == label]
                cluster_mean = self.compute_cluster_stats(cluster_points)

                true_buffer['means'].append(cluster_mean[label]['mean'].tolist())
                true_buffer['points'].append(cluster_points.tolist())

            self.true_buffer[pcl_msg.header.stamp] = true_buffer

            pcl_color_accumulated = self.XYZRGB_to_pcl_mat(
            selected_points[:, :6])
            pcl_color_accumulated['r'] = 255
            pcl_color_accumulated['g'] = 255
            pcl_color_accumulated['b'] = 0
            pc_array_accumulated = merge_rgb_fields(pcl_color_accumulated)
            pc_msg_accumulated = ros_numpy.msgify(
                PointCloud2, pc_array_accumulated, stamp=pcl_msg.header.stamp, frame_id=pcl_msg.header.frame_id)
            self.pclpublisher2.publish(pc_msg_accumulated)




if __name__ == '__main__':
    pcl_node = PointcloudProcessor()
    rospy.spin()
