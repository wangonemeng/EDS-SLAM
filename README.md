# EDS-SLAM

## A Tightly Coupled LiDAR-Inertial-Visual SLAM With Enhanced Dual Subsystems for Limited FoV LiDAR

## 1. Introduction

**EDS-SLAM** a LiDAR-inertial-visual SLAM method for solid-state LiDAR, which overcomes limitations of small FoV. This method consists of a LiDAR-inertial odometry (LIO) to provide geometric constraint and a visual-inertial odometry (VIO) to provide visual constraint. To resist the LiDAR degeneration, a novel residual construction method is proposed, which enhances the dimensionality of the residual in visual-inertial odometry. Furthermore, a keyframe-based static point registration method is proposed to effectively decrease incorrect constraints from dynamic points in LiDAR-inertial odometry.


### 1.1 Our paper

Our paper has been accepted to **TIM2025**.

If our code is used in your project, please cite our paper following the bibtex below:

```
@article{EDS-SLAM,
  author={Wang, Yimeng and Fang, Susu and Shen, Kangpeng and Liu, Yinhua},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={A Tightly Coupled LiDAR-Inertial-Visual SLAM With Enhanced Dual Subsystems for Limited FoV LiDAR}, 
  year={2025}
}
```

### 1.2 Our related video

Our accompanying videos are now available on [**Bilibili**](https://www.bilibili.com/video/BV18L5DzLEhA/?spm_id_from=333.1387.homepage.video_card.click).


## 2. Prerequisited

### 2.1 Ubuntu and ROS

Ubuntu 16.04~20.04.  [ROS Installation](http://wiki.ros.org/ROS/Installation).

### 2.2 PCL && Eigen && OpenCV

PCL>=1.6, Follow [PCL Installation](https://pointclouds.org/). 

Eigen>=3.3.4, Follow [Eigen Installation](https://eigen.tuxfamily.org/index.php?title=Main_Page).

OpenCV>=3.2, Follow [Opencv Installation](http://opencv.org/).

### 2.3 Sophus

 Sophus Installation for the non-templated/double-only version.

```bash
git clone https://github.com/strasdat/Sophus.git
cd Sophus
git checkout a621ff
mkdir build && cd build && cmake ..
make
sudo make install
```

### 2.4 Vikit

Vikit contains camera models, some math and interpolation functions that we need. Vikit is a catkin project, therefore, download it into your catkin workspace source folder.

```bash
cd catkin_ws/src
git clone https://github.com/uzh-rpg/rpg_vikit.git
```

### 2.5 **livox_ros_driver**

Follow [livox_ros_driver Installation](https://github.com/Livox-SDK/livox_ros_driver).

### 2.6 **CUDA and CUDNN**

Please install CUDA 11.6 and the corresponding version of cuDNN.

### 2.7 **Model and TensorRT8**

Due to the large memory footprint of the "Model" and "TensorRT8“, they have been uploaded to Baidu Netdisk for readers to download. Additionally, please place the "Model" and "TensorRT8" in the "src" directory. 

**Model**

链接: https://pan.baidu.com/s/1jjNuo_J1SsylnjG0uzwYUA 
提取码: riy3 

**TensorRT8**

链接: https://pan.baidu.com/s/1vCNO8IDQsRsEKfN56Oy__A 
提取码: np4q 



## 3. Build

Clone the repository and catkin_make:

```
cd ~/catkin_ws/src
git clone https://github.com/wangonemeng/EDS-SLAM
cd ../
catkin_make
source ~/catkin_ws/devel/setup.bash
```

## 4. Run the package

```
roslaunch point_livo fastlivo.launch
rosbag play YOUR_BAG.bag
```
**note**
The first time it runs, it will take approximately 10 minutes to convert the ONNX model to a TRT model.
