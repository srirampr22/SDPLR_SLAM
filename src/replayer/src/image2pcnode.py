#!/usr/bin/env python3
import pyrealsense2 as rs2
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from ros_numpy.point_cloud2 import pointcloud2_to_array, array_to_pointcloud2, split_rgb_field, merge_rgb_fields
import ros_numpy


class Image2PCnode:
    def __init__(self):
        self.camera_info_sub = rospy.Subscriber(
            '/camera/color/camera_info', CameraInfo, self.imageInfoCallback)
        self.mask_sub = rospy.Subscriber(
            '/yellow_mask', Image, self.callback_yellow_mask)
        self.pointcloud_sub = rospy.Subscriber(
            '/camera/depth/color/points', PointCloud2, self.pointcloud_callback)
        self.pclpublisher = rospy.Publisher(
            '/filtered_clusters', PointCloud2, latch=True, queue_size=100)
        
        self.intrinsics = None
        self.yellow_mask = None
        self.color_mapped_pcl = None 

    def callback_yellow_mask(self, yellow_mask_msg):
        try:
            self.yellow_mask = CvBridge().imgmsg_to_cv2(yellow_mask_msg, desired_encoding='mono8')
        except CvBridgeError as e:
            print(e)
            return

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

    def imageInfoCallback(self, cameraInfo):
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
        
    def project_to_image_plane(self, point):
        fx = self.intrinsics.fx  # Focal length in x
        fy = self.intrinsics.fy  # Focal length in y
        cx = self.intrinsics.ppx  # Principal point in x
        cy = self.intrinsics.ppy  # Principal point in y
        x, y, z = point
        pixel_x = int((fx * x) / z + cx)
        pixel_y = int((fy * y) / z + cy)
        return pixel_x, pixel_y
    
    def pcl_to_image(self, pcl_xyzrgb):
        # Initialize a blank image with the same shape as the yellow_mask
        image = np.zeros_like(self.yellow_mask, shape=(self.yellow_mask.shape[0], self.yellow_mask.shape[1], 3), dtype=np.uint8)

        for point in pcl_xyzrgb:
            # Extract XYZRGB components
            x, y, z, r, g, b = point
            # Project the 3D point to the image plane
            pixel_x, pixel_y = self.project_to_image_plane((x, y, z))

            # Verify that the pixel is inside the image bounds
            if 0 <= pixel_x < image.shape[1] and 0 <= pixel_y < image.shape[0]:
                image[pixel_y, pixel_x, :] = [r, g, b]

        return image
    
    def transform_pcl(self, pcl_mat_xyzrgb):
        # Translation
        translation = np.array([0.001, 0.014, -0.007])

        # Rotation quaternion (w, x, y, z)
        quaternion = np.array([1.00, -0.012, -0.001, -0.003])
        
        # Normalize the quaternion
        quaternion /= np.linalg.norm(quaternion)

        # Convert the quaternion into a rotation matrix
        w, x, y, z = quaternion
        rotation_matrix = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])

        # Apply the transformation to the point cloud
        transformed_pcl = pcl_mat_xyzrgb.copy()
        transformed_pcl[:, 0:3] = (rotation_matrix @ pcl_mat_xyzrgb[:, 0:3].T).T + translation

        return transformed_pcl

    def pointcloud_callback(self, pc_msg):

        pcl_xyzcol = pointcloud2_to_array(pc_msg)
        pcl_mat = split_rgb_field(pcl_xyzcol)
        pcl_mat_xyzrgb = self.pcl_mat_to_XYZRGB(pcl_mat)
        transformed_pcl = self.transform_pcl(pcl_mat_xyzrgb)
        print("transformed pcl shape:", transformed_pcl.shape)


        if self.yellow_mask is not None and self.intrinsics is not None:
            # self.color_mapped_pcl = pcl_mat_xyzrgb.copy()  # Update the variable value within the if block
            # pcl_image = self.pcl_to_image(transformed_pcl)
            # print(pcl_image.shape)

            # # Convert yellow_mask to a 3-channel image
            # yellow_mask_3ch = cv2.merge((self.yellow_mask, self.yellow_mask, self.yellow_mask))

            # # Overlay pcl_image and yellow_mask by taking the maximum value for each pixel
            # overlay_image = np.maximum(pcl_image, yellow_mask_3ch)

            # # Display the overlay image
            # cv2.imshow('Overlay Image', overlay_image)
            # cv2.waitKey(1)

            self.color_mapped_pcl = transformed_pcl.copy()

            # colored_pixel_coordinates = []
            for i in range(self.color_mapped_pcl.shape[0]):
                x, y, z, r, g, b = self.color_mapped_pcl[i]
                pixel_x, pixel_y = self.project_to_image_plane((x, y, z))
                if 0 <= pixel_x < self.yellow_mask.shape[1] and 0 <= pixel_y < self.yellow_mask.shape[0]:
                    if self.yellow_mask[pixel_y, pixel_x] > 0:
                        # Store the pixel coordinates (x, y) that are colored in the image
                        # colored_pixel_coordinates.append((pixel_x, pixel_y))
                        self.color_mapped_pcl[i, 3:6] = [255, 0, 0]

        pcl_filtered = self.XYZRGB_to_pcl_mat(self.color_mapped_pcl)

        pc_array_filtered = merge_rgb_fields(pcl_filtered)
        pc_msg_filtered = ros_numpy.msgify(
            PointCloud2, pc_array_filtered, stamp=pc_msg.header.stamp, frame_id=pc_msg.header.frame_id)
        self.pclpublisher.publish(pc_msg_filtered)

            


 
if __name__ == '__main__':
    rospy.init_node('image2PCnode', anonymous=True)
    image_processor = Image2PCnode()
    rospy.spin()
