#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/extract_clusters.h>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <ctime>
#include <cmath>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <mutex>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
// #include <boost/bind.hpp>
#include <pcl/console/print.h>
using namespace std;

class PointcloudProcessor{
public:
    PointcloudProcessor() {
        ros::NodeHandle nh;
        pcl_sub_.subscribe(nh, "/camera/depth/color/target_points", 10);
        odom_sub_.subscribe(nh, "/odom", 10);
        sync_.reset(new message_filters::Synchronizer<MySyncPolicy>(MySyncPolicy(10), pcl_sub_, odom_sub_));
        sync_->registerCallback(boost::bind(&PointcloudProcessor::pcl_transformer, this, _1, _2));
        pcl_publisher_ = nh.advertise<sensor_msgs::PointCloud2>("transformed_Ypcl", 10);
    }

    void pose_to_pq(nav_msgs::Odometry msg, Eigen::Vector3d& p, Eigen::Quaterniond& q) {
        p(0) = msg.pose.pose.position.x;
        p(1) = msg.pose.pose.position.y;
        p(2) = msg.pose.pose.position.z;
        q.x() = msg.pose.pose.orientation.x;
        q.y() = msg.pose.pose.orientation.y;
        q.z() = msg.pose.pose.orientation.z;
        q.w() = msg.pose.pose.orientation.w;
    }

    Eigen::Matrix4d msg_to_se3(nav_msgs::Odometry msg) {
        Eigen::Vector3d p;
        Eigen::Quaterniond q;
        pose_to_pq(msg, p, q);
        Eigen::Matrix3d rot = q.toRotationMatrix();
        Eigen::Matrix4d g = Eigen::Matrix4d::Identity();
        g.block<3,3>(0,0) = rot;
        g.block<3,1>(0,3) = p;
        return g;
    }

    void pcl_matrix_generator(const sensor_msgs::PointCloud2ConstPtr& pcl_msg, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud) {
        pcl::fromROSMsg(*pcl_msg, *cloud);
    }

    void pcl_transformer(const sensor_msgs::PointCloud2ConstPtr& pcl_msg, const nav_msgs::OdometryConstPtr& odom_msg) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl_matrix_generator(pcl_msg, cloud);

        Eigen::Matrix4d pose_se3 = msg_to_se3(*odom_msg);
        Eigen::Matrix4d pose_se3_inv = pose_se3.inverse();
        pcl::transformPointCloud(*cloud, *cloud, pose_se3_inv);

        ROS_INFO("CHECKPOINT 1");
        // Perform Euclidean clustering
        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
        tree->setInputCloud(cloud);
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
        ec.setClusterTolerance(0.05); // 5cm
        ec.setMinClusterSize(50);
        // ec.setMaxClusterSize(25000);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud);
        ROS_INFO("CHECKPOINT 2");
        ec.extract(cluster_indices);
        ROS_INFO("CHECKPOINT 3");
        // ROS_INFO_THROTTLE(0.5, "CHECKPOINT 1");
        // Estimate GMM for each cluster
        /*for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit) {
            cluster_cloud->points.push_back(cloud->points[*pit]);
        }
        cluster_cloud->width = cluster_cloud->points.size();
        cluster_cloud->height = 1;
        cluster_cloud->is_dense = true;

        pcl::GaussianMixtureModel<pcl::PointXYZRGB> gmm;
        gmm.setModelType(pcl::GaussianMixtureModel<pcl::PointXYZRGB>::DIAGONAL_COVARIANCE);
        gmm.setNumberOfModels(3); // number of components
        gmm.setInputCloud(cluster_cloud);
        gmm.setThreshold(1e-5);
        gmm.setVerbose(false);
        gmm.estimate();

        std::vector<pcl::GMMComponent<pcl::PointXYZRGB> > gmm_components = gmm.getModels();
        for (std::vector<pcl::GMMComponentpcl::PointXYZRGB >::const_iterator gmmit = gmm_components.begin(); gmmit != gmm_components.end(); ++gmmit) {
            Eigen::Vector3f mean = gmmit->getMean();
            Eigen::Matrix3f covariance = gmmit->getCovariance();
            float weight = gmmit->getWeight();
        }
        }*/

        sensor_msgs::PointCloud2 transformed_pcl_msg;
        pcl::toROSMsg(*cloud, transformed_pcl_msg);

        transformed_pcl_msg.header = pcl_msg->header;

        pcl_publisher_.publish(transformed_pcl_msg);
    }

private:
    message_filters::Subscriber<sensor_msgs::PointCloud2> pcl_sub_;
    message_filters::Subscriber<nav_msgs::Odometry> odom_sub_;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, nav_msgs::Odometry> MySyncPolicy;
    typedef message_filters::Synchronizer<MySyncPolicy> Sync;
    boost::shared_ptr<Sync> sync_; 
    ros::Publisher pcl_publisher_;
};

int main(int argc, char** argv) {
ros::init(argc, argv, "pcl_processor_node");
PointcloudProcessor processor;
ros::spin();
return 0;
}
