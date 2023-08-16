#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Point32.h>
#include <visualization_msgs/MarkerArray.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>

ros::Publisher mapped_points_pub;
std::vector<pcl::PointXYZRGB> yellow_points;

void yellowMaskCallback(const sensor_msgs::Image::ConstPtr& yellow_mask_msg) {
    // Implement your logic to extract yellow mask information
    // For simplicity, let's assume yellow_points contains the 2D pixel coordinates of the yellow mask
    // You can convert the sensor_msgs/Image to OpenCV format and process it to get the yellow pixels' positions.

    // For example:
    int img_width = yellow_mask_msg->width;
    int img_height = yellow_mask_msg->height;
    yellow_points.clear();
    for (int y = 0; y < img_height; ++y) {
        for (int x = 0; x < img_width; ++x) {
            if (yellow_mask_msg->data[y * img_width + x] == 255) {
                pcl::PointXYZRGB yellow_point;
                yellow_point.x = x;
                yellow_point.y = y;
                yellow_point.z = 0.0; // Assume all points are at the same depth (not using depth information here)
                yellow_point.r = 255; // Set color to red
                yellow_point.g = 0;
                yellow_point.b = 0;
                yellow_points.push_back(yellow_point);
            }
        }
    }
}

void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& point_cloud_msg) {
    // Convert the PointCloud2 message to PCL PointCloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromROSMsg(*point_cloud_msg, *pcl_cloud);

    // Transform the yellow points from 2D to 3D by setting a constant depth value
    double depth_value = 1.0; // You can set the desired depth value here
    for (auto& point : yellow_points) {
        point.z = depth_value;
    }

    // Concatenate the original point cloud with the yellow points
    pcl_cloud->points.insert(pcl_cloud->points.end(), yellow_points.begin(), yellow_points.end());

    // Convert the PCL PointCloud back to PointCloud2
    sensor_msgs::PointCloud2 output_cloud_msg;
    pcl::toROSMsg(*pcl_cloud, output_cloud_msg);

    // Publish the mapped points as a new topic
    mapped_points_pub.publish(output_cloud_msg);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "point_cloud_mapper");
    ros::NodeHandle nh;

    ros::Subscriber yellow_mask_sub = nh.subscribe("/yellow_mask", 1, yellowMaskCallback);
    ros::Subscriber point_cloud_sub = nh.subscribe("/camera/depth/color/points", 1, pointCloudCallback);
    mapped_points_pub = nh.advertise<sensor_msgs::PointCloud2>("/mapped_points", 1);

    ros::spin();

    return 0;
}
