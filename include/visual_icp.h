#pragma once

#include <vector>
#include <memory>
#include <deque>

#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <ros/publisher.h>
#include <sophus/se3.h>
#include <vikit/abstract_camera.h>
#include "common_lib.h"
#include "use-ikfom.hpp"

namespace visual_icp {

class VisualICP {
 public:
  VisualICP(ros::NodeHandle& nh);

  void UpdateState(esekfom::esekf<state_ikfom, 12, input_ikfom>& kf,
                   const cv::Mat& img, const PointCloudXYZI::Ptr& cloud);
  void Publish() const;
  Eigen::Vector3f getpixel(const cv::Mat& img, const Eigen::Vector2d& pc) const;

 private:
  void GetDepthImage(const PointCloudXYZI::Ptr& cloud);
  void RecoverDepth(const cv::Mat& depth_img, std::vector<cv::Point2f>& uvs,
                    PointCloudXYZRGB::Ptr cloud,
                    const Eigen::Vector3d& color = Eigen::Vector3d(255., 255., 255.)) const;
  void OpticalFlowTrack();
  void ExtractNewPoints();
  void ObservationModel(const state_ikfom& x);
  void IESKF(esekfom::esekf<state_ikfom, 12, input_ikfom>& kf);
  void UpdateStateWithKF(const state_ikfom& x);
  bool InFov(const Eigen::Vector2d& uv, int32_t boundry) const;

 public:
  vk::AbstractCamera* cam = nullptr;
  //PointCloudXYZRGB::Ptr all_frames_cloud;
 private:
  // pointcloud
  cv::Mat last_img, last_gray;
  cv::Mat curr_img, curr_gray;
  cv::Mat depth_img;
  std::vector<cv::Point2f> last_uvs;
  std::vector<cv::Point2f> curr_uvs;
  std::vector<cv::Point2f> new_uvs;
  PointCloudXYZRGB::Ptr last_cloud = nullptr;
  PointCloudXYZRGB::Ptr curr_cloud = nullptr;
  PointCloudXYZRGB::Ptr curr_cloud_cam = nullptr;
  PointCloudXYZRGB::Ptr new_cloud = nullptr;
  std::vector<Eigen::Vector3d> residuals;
  std::vector<Eigen::Matrix<double, 3, 6>> jacobians;
  double meas_cov = 0.01;
  int32_t max_iter_num = 3;
  int32_t nn_patch_size = 5;
  // camera and image
  int32_t grid_size, grid_num;
  int32_t width, height;
  double fx, fy, cx, cy;
  Eigen::Vector3d tcl;
  Eigen::Matrix3d rcl;
  Sophus::SE3 tfcl;
  Sophus::SE3 tfci;
  Sophus::SE3 tfwc;
  // publisher
  ros::Publisher pub_depth_img;
  ros::Publisher pub_tracked_img;
  ros::Publisher pub_match_marker;
  ros::Publisher pub_visual_points;
  bool log_info = true;
};

typedef boost::shared_ptr<VisualICP> VisualICPPtr;

}  // namespace visual_icp
