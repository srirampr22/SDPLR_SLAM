#!/usr/bin/env python3

import rospy
import tf2_ros
import geometry_msgs.msg

def static_tf_publisher():
    rospy.init_node('static_tf_publisher')
    
    static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
    
    static_transformStamped = geometry_msgs.msg.TransformStamped()
    
    static_transformStamped.header.stamp = rospy.Time.now()
    static_transformStamped.header.frame_id = "base_link"
    static_transformStamped.child_frame_id = "camera_link"
    
    static_transformStamped.transform.translation.x = 1.0  # Adjust the translation values as needed
    static_transformStamped.transform.translation.y = 1.0
    static_transformStamped.transform.translation.z = 1.0
    
    static_transformStamped.transform.rotation.x = 0.0  # Adjust the rotation values as needed
    static_transformStamped.transform.rotation.y = 0.0
    static_transformStamped.transform.rotation.z = 0.0
    static_transformStamped.transform.rotation.w = 1.0
    
    rate = rospy.Rate(10.0)  # Adjust the publish rate as needed
    
    while not rospy.is_shutdown():
        static_tf_broadcaster.sendTransform(static_transformStamped)
        rate.sleep()

if __name__ == '__main__':
    try:
        static_tf_publisher()
    except rospy.ROSInterruptException:
        pass
