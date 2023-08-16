#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, CameraInfo
from sensor_msgs.msg import PointCloud2, PointField
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from ros_numpy.point_cloud2 import pointcloud2_to_array, array_to_pointcloud2, split_rgb_field, merge_rgb_fields
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
import pyrealsense2 as rs
import ros_numpy
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import message_filters
import pyrealsense2 as rs2
from sklearn.cluster import DBSCAN
import tf
from geometry_msgs.msg import PointStamped
import threading
import tf2_ros
from geometry_msgs.msg import TransformStamped
import random


class ImageProcessor:
    def __init__(self):
        self.image_sub = message_filters.Subscriber(
            '/camera/color/image_raw', Image)
        self.image_pub = rospy.Publisher(
            '/segmented_image', Image, queue_size=10)
        self.pointcloud_sub = rospy.Subscriber(
            '/velodyne_points_filtered', PointCloud2, self.pointcloud_callback)
        self.bbox_pointcloud_pub = rospy.Publisher(
            '/filtered_projected_points', PointCloud2, queue_size=10)
        self.map_frame_pub = rospy.Publisher(
            '/map_frame_pub', PointCloud2, queue_size=10)
        self.camera_info_sub = rospy.Subscriber(
            '/camera/color/camera_info', CameraInfo, self.imageDepthInfoCallback)
        self.depth_sub = message_filters.Subscriber(
            '/camera/aligned_depth_to_color/image_raw', Image)
        self.tf_listener = tf.TransformListener()

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.image_depth_callback)

        # Start the TF listener in a separate thread
        tf_thread = threading.Thread(target=self.start_tf_listener)
        # This makes sure the thread is terminated when the main program exits
        tf_thread.daemon = True
        tf_thread.start()

        self.bridge = CvBridge()
        self.Segmented_image = None
        self.intrinsic_params = None
        self.depth_values = None
        self.foreground_points = None
        self.intrinsics = None
        self.bbx_pcl_points = None
        self.obj_centre_pcl = None
        self.bboxes = None
        self.T_camera_to_map = None
        self.fixed_subject_position_map = None
        self.latest_subject_positions_map = {}
        self.new_label = []
        self.pcl_buffer = {}
        self.accumulated_points = None
        self.camera_pos_t0 = None

        Rz = np.array([[0, -1, 0],
                       [1, 0, 0],
                       [0, 0, 1]])

        Ry = np.array([[0, 0, -1],
                       [0, 1, 0],
                       [1, 0, 0]])

        Rx = np.array([[1, 0, 0],
                       [0, 0, -1],
                       [0, 1, 0]])

        self.camera_extrinsic_params = {
            'rotation': np.dot(Ry, Rx),
            'translation': np.array([0.0, 0.0, 0.0])
        }

    def start_tf_listener(self):
        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer)

        rate = rospy.Rate(10)  # 10 Hz

        try:
            while not rospy.is_shutdown():
                try:
                    transform_msg = tf_buffer.lookup_transform(
                        'map', 'camera_link', rospy.Time(0))
                    self.tf_callback(transform_msg)
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                    continue

                rate.sleep()
        except rospy.ROSInterruptException:
            pass

    def tf_callback(self, transform_msg):
        # Assuming the frame_id is 'map' and the child_frame_id is 'camera_link'
        if transform_msg.header.frame_id == 'map' and transform_msg.child_frame_id == 'camera_link':
            # Get the translation and rotation from the transform message
            translation = transform_msg.transform.translation
            rotation = transform_msg.transform.rotation

            # Convert the rotation to a 4x4 homogeneous transformation matrix
            self.T_camera_to_map = tf.TransformerROS().fromTranslationRotation(
                (translation.x, translation.y, translation.z),
                (rotation.x, rotation.y, rotation.z, rotation.w)
            )
        else:
            self.T_camera_to_map = None

    def XYZRGB_to_pcl_mat(self, xyzrgb):
        # Extract XYZRGB components
        x = xyzrgb[:, 0]
        y = xyzrgb[:, 1]
        z = xyzrgb[:, 2]
        # red = xyzrgb[:, 3]
        # green = xyzrgb[:, 4]
        # blue = xyzrgb[:, 5]
        # labels = xyzrgb[:, -1]

        # Create a new point cloud matrix with separate components
        pcl_mat = np.zeros((xyzrgb.shape[0],),
                           dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32),
                                  ('r', np.uint8), ('g', np.uint8), ('b', np.uint8)])
        pcl_mat['x'] = x
        pcl_mat['y'] = y
        pcl_mat['z'] = z
        pcl_mat['r'] = 255
        pcl_mat['g'] = 0
        pcl_mat['b'] = 0
        # pcl_mat['label'] = labels

        return pcl_mat

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

    def imageDepthInfoCallback(self, cameraInfo):
        try:
            if self.intrinsics:
                return
            self.intrinsics = rs2.intrinsics()
            self.intrinsics.width = cameraInfo.width
            self.intrinsics.height = cameraInfo.height
            self.intrinsics.ppx = cameraInfo.K[2]
            self.intrinsics.ppy = cameraInfo.K[5]
            self.intrinsics.fx = cameraInfo.K[0]
            self.intrinsics.fy = cameraInfo.K[4]
            if cameraInfo.distortion_model == 'plumb_bob':
                self.intrinsics.model = rs2.distortion.brown_conrady
            elif cameraInfo.distortion_model == 'equidistant':
                self.intrinsics.model = rs2.distortion.kannala_brandt4
            self.intrinsics.coeffs = [i for i in cameraInfo.D]
        except CvBridgeError as e:
            print(e)
            return

    # def camera_info_callback(self, info_msg):
    #     self.intrinsic_params = np.array(info_msg.K).reshape(3, 3)

    def image_depth_callback(self, image_msg, depth_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding='passthrough')

            if image_msg.header.stamp != depth_msg.header.stamp:
                print("NOT IN SYNC")
            else:

                if self.intrinsics is not None:

                    bbox_uv = []
                    depth_array = np.array(depth_image, dtype=np.float32)

                    # print("depth_array shape:", depth_array.shape)

                    hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
                    # print("hsv_image:", hsv_image.shape)

                    # Define lower and upper thresholds for yellow color in HSV space
                    lower_yellow = np.array([20, 105, 122], dtype=np.uint8)
                    upper_yellow = np.array([33, 255, 255], dtype=np.uint8)

                    # Create a mask based on the yellow color range
                    yellow_mask = cv2.inRange(
                        hsv_image, lower_yellow, upper_yellow)

                    # print("yellow_mask.shape")

                    # Find contours in the yellow mask
                    contours, _ = cv2.findContours(
                        yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # print("contour.shape")

                    # Iterate over the contours and draw bounding boxes
                    min_w, min_h = 20, 20

                    self.bboxes = [cv2.boundingRect(
                        contour) for contour in contours if cv2.contourArea(contour) > min_w * min_h]
                    # print("bbox:", self.bboxes)

                    if len(self.bboxes) == 0:
                        print("object not found")
                    else:

                        x, y, w, h = self.bboxes[0]

                        # Draw bounding box on the image
                        cv2.rectangle(cv_image, (x, y),
                                      (x + w, y + h), (0, 255, 0), 2)

                        self.bbx_pcl_points = self.uv_to_xyz(depth_array)

                        # Convert point cloud to Numpy array, if not already done
                        points = np.array(self.bbx_pcl_points)

                        # Apply DBSCAN algorithm
                        dbscan = DBSCAN(eps=0.3, min_samples=10).fit(points)

                        # Get labels (these represent the different clusters)
                        labels = dbscan.labels_

                        labelled_points = np.c_[points, labels]

                        # Find the largest cluster (which we'll assume is the foreground object)
                        unique_labels, counts = np.unique(
                            labels, return_counts=True)
                        largest_cluster_label = unique_labels[np.argmax(
                            counts)]

                        # Separate points into foreground and background based on the largest cluster
                        self.foreground_points = labelled_points[labels ==
                                                                 largest_cluster_label]
                        background_points = labelled_points[labels !=
                                                            largest_cluster_label]

                        # print("ROI points:",self.foreground_points.shape)

                        segmented_msg = self.bridge.cv2_to_imgmsg(
                            cv_image, "bgr8")
                        self.image_pub.publish(segmented_msg)

        except CvBridgeError as e:
            print(e)

    def pointcloud_callback(self, pcl_msg):

        if self.camera_pos_t0 is None:
            # Get the camera position at time t=0 from the first point cloud message
            try:
                (trans_t0, rot_t0) = self.tf_listener.lookupTransform(
                    'map', 'camera_link', pcl_msg.header.stamp)
                self.camera_pos_t0 = np.array(trans_t0)
                rospy.loginfo("Camera position at t=0: %s",
                              str(self.camera_pos_t0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.logwarn("Failed to get camera position at time t=0")
                return

        if self.foreground_points is not None:

            ROI_labels, counts = np.unique(
                self.foreground_points[:, -1], return_counts=True)

            ROI_cluster_stats = self.compute_cluster_stats(
                self.foreground_points)
            # ROI_mean_points = []
            # for label in ROI_labels:
            #     label_mean = ROI_cluster_stats[label]['mean']
            #     ROI_mean_points.append(label_mean)

            # ROI_mean_points = np.array(ROI_mean_points)
            ROI_mean_points = self.foreground_points

            transformed_points = self.transform_points_to_map(
                ROI_mean_points, self.T_camera_to_map)
            transformed_points = np.column_stack(
                (transformed_points, ROI_labels))

            # random_labelled_points = []
            # for original_label in (ROI_labels):
            #     mask = transformed_points[:, -1] == original_label
            #     point = transformed_points[mask]
            #     new_label_counter = random.randint(1, 1000)
            #     point[:, -1] == new_label_counter
            #     random_labelled_points.extend(point)

            # random_labelled_points = np.array(random_labelled_points)

            # # print("random_labelled_points:", random_labelled_points.shape)

            # new_labels, counts = np.unique(
            #     random_labelled_points[:, -1], return_counts=True)

            # # Get the camera position at the current time
            # try:
            #     (trans_current, rot_current) = self.tf_listener.lookupTransform(
            #         'map', 'camera_link', pcl_msg.header.stamp)
            #     camera_pos_current = np.array(trans_current)
            # except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            #     rospy.logwarn(
            #         "Failed to get camera position at the current time")
            #     return

            # print("camera_pos_t0:", self.camera_pos_t0)
            # print("camera_pos_current", camera_pos_current)
            # origin = [0,0,0]
            # l1 = line formed by origin and self.camera_pos_t0
            # l2 = line formed by origin and camera_pos_current
            # angle = between l1 and l2
            # Calculate the dot product of l1 and l2
            # l1 = self.camera_pos_t0  # Origin to camera_pos_t0 vector
            # l2 = camera_pos_current
            # dot_product = np.dot(l1, l2)

            # # Calculate the magnitudes of l1 and l2
            # magnitude_l1 = np.linalg.norm(l1)
            # magnitude_l2 = np.linalg.norm(l2)

            # # Calculate the angle between l1 and l2 (in radians)
            # angle_rad = np.arccos(dot_product / (magnitude_l1 * magnitude_l2))

            # # Convert the angle to degrees
            # angle_deg = np.degrees(angle_rad)

            # # Calculate the displacement between the two camera positions
            # displacement = np.linalg.norm(
            #     camera_pos_current - self.camera_pos_t0)

            # ordered_labelled_points = self.calculate_fixed_ROI_dist(
            #     random_labelled_points, camera_pos_current, self.camera_pos_t0)
            
            # print("ordered_labelled_points:", ordered_labelled_points[:,-1])

            # prev_time_stamp = None
            # for timestamp in self.pcl_buffer.keys():
            #     if timestamp < pcl_msg.header.stamp:
            #         if prev_time_stamp is None or timestamp > prev_time_stamp:
            #             prev_time_stamp = timestamp

            # # Note: compare the position of the labeleld points from current time step to the once in the previous time step which should be in self.pcl_buffer
            # # Note: for each point in current timestep the label of the closest point from previos time step shoudl be assigned to it

            # if prev_time_stamp is not None:
            #     # Compare points in the current time step to the previous time step
            #     true_labelled_points = []
            #     prev_points = self.pcl_buffer[prev_time_stamp]['points']
            #     for point1 in random_labelled_points:
            #         for point2 in prev_points:
            #             # Calculate the distance between point1 and point2
            #             distance = np.linalg.norm(point1[:3] - point2[:3])
            #             # Use a threshold to determine if the points are close
            #             threshold = 0.2
            #             if distance < threshold:
            #                 # Assign the label of point2 to point1
            #                 point1[-1] = point2[-1]
            #                 true_labelled_points.extend(point1)
            #                 # Break the loop since we found a close point in prev_points
            #                 break
            #             else:
            #                 point1[-1] = point1[-1]
            #                 true_labelled_points.extend(point1)

            #     true_labelled_points = np.array(true_labelled_points)

            #     # print("random_labelled_points:", random_labelled_points.shape)

            #     self.accumulated_points = true_labelled_points
            # else:
            #     self.accumulated_points = random_labelled_points

            # # accumulated_points = np.concatenate(
            # #     self.accumulated_points, axis=0)
            # # self.pcl_buffer[pcl_msg.header.stamp]['points'] = new_transformed_points
            # print("transformed_points:", self.accumulated_points)

            # for label, points in zip(ROI_labels, transformed_points):
            #     self.latest_subject_positions_map[label] = points

            transformed_pcl_array = self.XYZRGB_to_pcl_mat(
                transformed_points[:, :-1])

            transformed_pc_array = merge_rgb_fields(transformed_pcl_array)
            transformed_pc_msg = ros_numpy.msgify(
                PointCloud2, transformed_pc_array, stamp=pcl_msg.header.stamp, frame_id='map')  # Set the frame_id to 'map'
            self.map_frame_pub.publish(transformed_pc_msg)

        else:
            return None

        
    def calculate_fixed_ROI_dist(self, random_labelled_points, camera_pos_current, camera_pos_t0):

        map_dist = []
        for point in random_labelled_points:
            d1 = np.linalg.norm(point[:3] - camera_pos_current[:3])
            d2 = np.linalg.norm(camera_pos_current[:3] - camera_pos_t0[:3])
            angle_degrees = self.calculate_angle(d1, d2)
            d3 = np.sqrt(d1**2 + d2**2 - 2 * d1 * d2 * np.cos(np.radians(angle_degrees)))
            map_dist.append(d3)
        map_dist = np.array(map_dist)
        ordered_labelled_points = np.hstack((random_labelled_points, map_dist.reshape(-1, 1)))

        return ordered_labelled_points
    
    def calculate_angle(self, d1, d2):

        dot_product = np.dot(d1, d2)

        norm_d1 = np.linalg.norm(d1)
        norm_d2 = np.linalg.norm(d2)

        cosine_theta = dot_product / (norm_d1 * norm_d2)

        theta = np.arccos(cosine_theta)

        theta_degrees = np.degrees(theta)

        return theta_degrees

    def transform_points_to_map(self, points_in_camera_frame, T_camera_to_map):
        # Convert points from 3D to 4D homogeneous coordinates
        points_homogeneous = np.c_[
            points_in_camera_frame, np.ones(len(points_in_camera_frame))]

        # Perform the transformation
        points_in_map_frame = np.dot(points_homogeneous, T_camera_to_map.T)

        # Convert points back to 3D coordinates by dividing by the fourth coordinate value
        points_in_map_frame /= points_in_map_frame[:, 3][:, None]

        return points_in_map_frame[:, :3]

    # def transform_points_to_map(self, points_camera_frame):
    #     # Ensure you have the camera-to-map transformation matrix available
    #     if self.T_camera_to_map is not None:
    #         # Convert points from the camera_link frame to the map frame
    #         points_map_frame = np.dot(self.T_camera_to_map, np.hstack((points_camera_frame, np.ones((points_camera_frame.shape[0], 1)))).T).T[:, :3]
    #         return points_map_frame
    #     else:
    #         rospy.logwarn("Camera-to-map transformation not available. Unable to transform points.")
    #         return None

    def compute_cluster_stats(self, xyz_cl):

        cluster_labels = np.unique(xyz_cl[:, -1])
        cluster_stats = {}

        for label in cluster_labels:
            if label == -1:
                continue  # Skip noise points

            cluster_points = xyz_cl[xyz_cl[:, -1] == label][:, :3]
            cluster_mean = np.mean(cluster_points, axis=0)
            cluster_cov = np.cov(cluster_points, rowvar=False)

            cluster_stats[label] = {
                'mean': cluster_mean, 'covariance': cluster_cov}

        return cluster_stats

    def uv_to_xyz(self, depth_array):

        P3D_array = []
        intrinsic_param = self.intrinsics

        for bbox in self.bboxes:
            x, y, w, h = bbox
            for u in range(x, x + w):
                for v in range(y, y + h):
                    # Get the depth value at the pixel
                    depth = depth_array[v, u] * 0.001
                    if np.isfinite(depth):
                        # Convert pixel coordinates and depth value to 3D coordinates in the camera frame
                        xyz_coord = self.pixel_to_3d(
                            (u, v), depth, intrinsic_param)
                        # Convert 3D coordinates in the camera frame to the Velodyne frame
                        transformed_coord = self.rgbd_to_velodyne(
                            xyz_coord, self.camera_extrinsic_params)
                        P3D_array.append(transformed_coord)
        # Convert the list to a numpy array
        P3D_array = np.array(P3D_array, dtype=np.float32)

        # print(P3D_array.shape)

        return P3D_array

    def pixel_to_3d(self, pixel_coord, depth, intrinsic_param):
        # Convert pixel coordinates to camera frame using depth value and intrinsic camera parameters
        # pixel_coord is a tuple (u, v), depth is a float, intrinsic_params is a dictionary
        u, v = pixel_coord
        # Accessing fx
        fx = self.intrinsics.fx

        # Accessing fy
        fy = self.intrinsics.fy

        # Accessing cx
        cx = self.intrinsics.ppx

        # Accessing cy
        cy = self.intrinsics.ppy
        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        # print("X:",x,"Y:",y,"Z:",z)
        return np.array([x, y, z])

    def rgbd_to_velodyne(self, xyz_coord, extrinsic_params):
        # Convert camera frame to Velodyne frame using extrinsic camera parameters
        # cam_coord is a np.array([x, y, z]), extrinsic_params is a dictionary
        R = extrinsic_params['rotation']
        T = extrinsic_params['translation']
        R_inv = R.T

        # Inverse of translation vector (T) in a transformation matrix is -R.T * T
        T_inv = -np.dot(R.T, T)

        # Using the inverted rotation and translation for depth_to_color transformation
        return np.dot(R_inv, xyz_coord) + T_inv


if __name__ == '__main__':
    rospy.init_node('image_processor_node')
    image_processor = ImageProcessor()
    rospy.spin()
