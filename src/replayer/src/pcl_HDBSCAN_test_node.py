#!/usr/bin/env python3
import rospy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
import std_msgs
import ros_numpy
import message_filters
import tf.transformations as tr
import ctypes
import struct
import hdbscan
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
import time

class PointcloudProcessor():
    def __init__(self):
        rospy.init_node('pcl_HDBSCAN_test_processor', anonymous=True)

        pcl_sub = rospy.Subscriber('/map', PointCloud2, self.pcl_transformer)
        self.pclpublisher = rospy.Publisher('transformed_Ypcl', PointCloud2, latch=True, queue_size=100)
        
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

    def pcl_transformer(self, pcl_msg):

        print("TEST RUN HDBSCAN algorithm on /map topic")
        print("Successfully received pcl_msg at time: ", pcl_msg.header.stamp)
        pcl_mat = self.pcl_matrix_converter(pcl_msg)

        maskint1 = ((pcl_mat[:,2] < -0.05) | (pcl_mat[:,2] > 0.10))
        pcl_mat = pcl_mat[np.logical_not(maskint1)]

        t0_dbscan = time.time()
        # db = DBSCAN(eps=0.01, min_samples=10, n_jobs=-1)
        db = hdbscan.HDBSCAN(min_cluster_size=10)

        for _ in tqdm(range(10), desc='Clustering'):
            db.fit(pcl_mat[:,:3])
        t1_dbscan = time.time()
        print("HDBSCAN computation:", t1_dbscan - t0_dbscan)


        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print('Estimated number of clusters: %d' % n_clusters)
        xyz_cl = np.c_[pcl_mat, labels] 
        maskint2 = (xyz_cl[:, 4] == -1)
        xyz_cl = xyz_cl[np.logical_not(maskint2)]

        lbls, counts = np.unique(xyz_cl[:, 4], return_counts=True)
        mask = np.isin(xyz_cl[:, 4], lbls[counts > 500])
        xyz_cl = xyz_cl[~mask]

        label_components = np.unique(xyz_cl[:, 4])
        # gmm_means, gmm_covariances = self.fit_gmm_EM(pcl_mat, label_components)
        n_components = len(label_components)

        t0_gmm = time.time()
        gmm_means, gmm_covariances = self.fit_gmm_EM(pcl_mat, n_components)
        t1_gmm = time.time()
        print("EM-GMM computation:", t1_gmm-t0_gmm)

        print("Completed EM GMM")

        pcl_xyz = pcl_mat[:,:3]
        ones_column = np.ones([pcl_xyz.shape[0], 1])
        transformed_pcl = np.hstack((pcl_xyz, ones_column))

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


if __name__ == '__main__':

    pcl_node = PointcloudProcessor()
    rospy.spin()

