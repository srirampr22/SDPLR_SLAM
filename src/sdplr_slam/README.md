# SSL_SLAM
## Lightweight 3-D Localization and Mapping for Solid-State LiDAR (Intel Realsense L515 as an example)


## 2. Prerequisites
### 2.1 **Ubuntu** and **ROS**
Ubuntu 64-bit 18.04.

ROS Noetic. [ROS Installation](http://wiki.ros.org/ROS/Installation)

### 2.2. **Ceres Solver**
Follow [Ceres Installation](http://ceres-solver.org/installation.html).

### 2.3. **PCL**
Follow [PCL Installation](http://www.pointclouds.org/downloads/linux.html).

Tested with 1.8.1

```

### 2.5. **Trajectory visualization**
For visualization purpose, this package uses hector trajectory sever, you may install the package by 
```
sudo apt-get install ros-noetic-hector-trajectory-server
```
Alternatively, you may remove the hector trajectory server node if trajectory visualization is not needed

## 3. Build 
### 3.1 Clone repository:
```
    cd ~/catkin_ws/src
    git clone git@github.com:srirampr22/SDPLR_SLAM.git
    cd ..
    catkin_make
    source ~/catkin_ws/devel/setup.bash
```


### 3.3 Launch ROS
if you would like to create the map at the same time, you can run 
```
    roslaunch sdplr_slam sdpl_slam_mapping.launch
```

if only localization is required, you may refer to run
```
    roslaunch sdplr_slam sdplr_slam.launch
```

## 4. Sensor Setup
If you have new Realsense L515 sensor, you may follow the below setup instructions


### 4.1 L515
<p align='center'>
<img width="35%" src="/img/realsense_L515.jpg"/>
</p>

### 4.2 Librealsense
Follow [Librealsense Installation](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md)

### 4.3 Realsense_ros
Copy [realsense_ros](https://github.com/IntelRealSense/realsense-ros) package to your catkin folder
```
    cd ~/catkin_ws/src
    git clone https://github.com/IntelRealSense/realsense-ros.git
    cd ..
    catkin_make
```

### 4.4 Launch ROS
```
    roslaunch sdplr_slam sdplr_slam_L515.launch
```

This runs `sdplr_slam_mapping.launch` with live L515 data.


