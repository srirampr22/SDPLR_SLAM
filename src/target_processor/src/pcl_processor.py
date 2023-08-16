#!/usr/bin/env python3
import rospy
import numpy as np
import pdb
from tqdm import tqdm
import traceback
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
import std_msgs
import ros_numpy
# from ros_numpy.point_cloud2 import pointcloud2_to_array, array_to_pointcloud2, split_rgb_field, merge_rgb_fields
import message_filters
import tf.transformations as tr
import pdb
import ctypes
import struct
# import hdbscan
# import open3d as o3d
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

class PointcloudProcessor():
    def __init__(self):
        print("1")
        rospy.init_node('pcl_processor', anonymous=True)
        # rospy.Subscriber('/map', PointCloud2, self.callback)
        # rospy.Subscriber('/velodyne_points_filtered', PointCloud2, self.pcl_filter)
        # rospy.Subscriber('/camera/depth/color/target_points', PointCloud2, self.pcl_transformer)

        pcl_sub = message_filters.Subscriber('/map', PointCloud2)
        # odom_sub = message_filters.Subscriber('/odom', Odometry)
        # ts = message_filters.TimeSynchronizer([pcl_sub, odom_sub], 100)
        # ts.registerCallback(self.pcl_transformer)
        
        self.pclpublisher = rospy.Publisher('transformed_Ypcl', PointCloud2, latch=True, queue_size=100)
        print("2")
    

    
    def pcl_callback(self, msg):
        print("Received a PointCloud2 message at time: ", msg.header.stamp)

    def odom_callback(self, msg):
        print("Received an Odometry message at time: ", msg.header.stamp)



    def pose_to_pq(self, msg):
        """Convert a C{nav_msgs/Odometry} into position/quaternion np arrays

        @param msg: ROS message to be converted
        @return:
        - p: position as a np.array
        - q: quaternion as a numpy array (order = [x,y,z,w])
        """
        print("3")
        p = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
        q = np.array([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        print("4")
        return p, q
    

    def msg_to_se3(self, msg):
        """Conversion from geometric ROS messages into SE(3)

        @param msg: Message to transform. Acceptable type - C{nav_msgs/Odometry}
        @return: a 4x4 SE(3) matrix as a numpy array
        @note: Throws TypeError if we receive an incorrect type.
        """
        print("5")
        if isinstance(msg, Odometry):
            p, q = self.pose_to_pq(msg)
        else:
            raise TypeError("Invalid type for conversion to SE(3)")

        norm = np.linalg.norm(q)
        if np.abs(norm - 1.0) > 1e-3:
            raise ValueError(
                "Received un-normalized quaternion (q = {0:s} ||q|| = {1:3.6f})".format(str(q), np.linalg.norm(q)))
        elif np.abs(norm - 1.0) > 1e-6:
            q = q / norm #(change / to //)
        
        g = tr.quaternion_matrix(q)
        g[0:3, -1] = p
        print("6")
        return g

    def fit_gmm_EM(self, pcl_mat, n_components):
        # Initialize the Gaussian Mixture Model with the desired number of clusters
        print("7")
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        # Fit the GMM using the first three columns (x, y, z) of the point cloud data
        gmm.fit(pcl_mat[:, :3])
        # Get the mean and covariance of each cluster
        means = gmm.means_
        covariances = gmm.covariances_
        print("8")
        return means, covariances

    def pcl_mat_generator(self, ros_point_cloud):
        print("9")
        xyz = np.array([[0,0,0]])
        rgb = np.array([[0,0,0]])
        #self.lock.acquire()
        gen = pc2.read_points(ros_point_cloud, skip_nans=True)
        int_data = list(gen)

        for x in int_data:
            test = x[3] 
            # cast float32 to int so that bitwise operations are possible
            s = struct.pack('>f' ,test)
            i = struct.unpack('>l',s)[0]
            # you can get back the float value by the inverse operations
            pack = ctypes.c_uint32(i).value
            r = (pack & 0x00FF0000)>> 16
            g = (pack & 0x0000FF00)>> 8
            b = (pack & 0x000000FF)
            # prints r,g,b values in the 0-255 range
                        # x,y,z can be retrieved from the x[0],x[1],x[2]
            xyz = np.append(xyz,[[x[0],x[1],x[2]]], axis = 0)
            rgb = np.append(rgb,[[r,g,b]], axis = 0)
        print("10")    

        return [xyz, rgb]

    def pcl_matrix_converter(self, point_cloud_msg):
        print("11")   
        # print('CHECKPOINT1\n')
        pc = ros_numpy.numpify(point_cloud_msg)
        # print('CHECKPOINT2\n')
        points=np.zeros((pc.shape[0],4))
        points[:,0]=pc['x']
        points[:,1]=pc['y']
        points[:,2]=pc['z']
        points[:,3]=pc['rgb'] 
        print("12")    
        return points

    def pcl_transformer(self, pcl_msg, odom_msg):
        print("13")   
        try:
            print("Successfully received pcl_msg at time: ", pcl_msg.header.stamp)
            print("Successfully received odom_msg at time: ", odom_msg.header.stamp)
            
            # pose_se3 = self.msg_to_se3(odom_msg)
            # pose_se3_inv = np.linalg.inv(pose_se3)
            pcl_mat = self.pcl_matrix_converter(pcl_msg)

            print("CHECKPOINT 1 pcl_array shape: ", pcl_mat.shape)
            # maskint1 = (pcl_mat[:,2] > -0.05)
            maskint1 = ((pcl_mat[:,2] < -0.05) | (pcl_mat[:,2] > 0.10))
            # maskint1 = ((pcl_mat[:,2] > -0.05) | (pcl_mat[:,2] < 0.10))
            pcl_mat = pcl_mat[np.logical_not(maskint1)]
            print("CHECKPOINT 2 pcl_array shape: ", pcl_mat.shape)
            # !!!NOTE: condition which pcl_mat.shape is array with 0 
            # sample(s) (shape=(0, 3)) while a minimum of 1 is required!!!
            db = DBSCAN(eps=0.01, min_samples=10, n_jobs=-1)
            for _ in tqdm(range(10), desc='Clustering'):
                db.fit(pcl_mat[:,:3])

            labels = db.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            print('Estimated number of clusters: %d' % n_clusters)
            xyz_cl = np.c_[pcl_mat, labels] #will be n x 5 cols array
            maskint2 = (xyz_cl[:, 4] == -1)
            xyz_cl = xyz_cl[np.logical_not(maskint2)]

            # Count the number of occurrences of each unique label
            lbls, counts = np.unique(xyz_cl[:, 4], return_counts=True)
            mask = np.isin(xyz_cl[:, 4], lbls[counts > 500])
            xyz_cl = xyz_cl[~mask]
            print("CHECKPOINT 2 xyz_cl shape: ", xyz_cl.shape)
            # pdb.set_trace()

            # find the gaussian mixture model of the clustered pcl
            label_components = np.unique(xyz_cl[:, 4])
            gmm_means, gmm_covariances = self.fit_gmm_EM(pcl_mat, label_components)
            print("CHECKPOINT 3: completed EM GMM")

            pcl_xyz = pcl_mat[:,:3]
            # pcl_xyz = xyz_clustered[:,:3]
            ones_column = np.ones([pcl_xyz.shape[0], 1])
            transformed_pcl = np.hstack((pcl_xyz, ones_column))
            transformed_pcl_cloud = np.matmul(pose_se3_inv, np.transpose(transformed_pcl))
            transformed_pcl_cloud[-1, :] = pcl_mat[:, -1]
            # transformed_pcl_cloud[-1, :] = pcl_xyz[:, -1]
            transformed_pcl_cloud = np.transpose(transformed_pcl_cloud)

            pc_array = np.zeros(len(transformed_pcl_cloud), dtype=[
                ('x', np.float32),
                ('y', np.float32),
                ('z', np.float32),
                # ('intensity', np.float32),
                ('rgb', np.float32),
            ])
            pc_array['x'] = transformed_pcl_cloud[:, 0]
            pc_array['y'] = transformed_pcl_cloud[:, 1]
            pc_array['z'] = transformed_pcl_cloud[:, 2]
            pc_array['rgb'] = transformed_pcl_cloud[:, 3]

            # Define the PointCloud2 message header
            pc_msg = ros_numpy.msgify(PointCloud2, pc_array, stamp=pcl_msg.header.stamp, frame_id=pcl_msg.header.frame_id)
            self.pclpublisher.publish(pc_msg)
        except Exception as e:
            print("Error in pcl_transformer: ", e)
            traceback.print_exc()  # This will print the traceback of the error.
        # print("14")   


if __name__ == '__main__':
    print("15")   
    try:
        pcl_node = PointcloudProcessor()
        rospy.spin()
    except Exception as e:
        print("Error in main execution: ", e)
        traceback.print_exc()  # This will print the traceback of the error.


