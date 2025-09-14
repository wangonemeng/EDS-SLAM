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

namespace visual_pnp {

class VisualPNP {
 public:
  VisualPNP(ros::NodeHandle& nh);

  void UpdateState(esekfom::esekf<state_ikfom, 12, input_ikfom>& kf,
                   const cv::Mat& img, const PointCloudXYZI::Ptr& cloud);
  void Publish() const;
  Eigen::Vector3f getpixel(const cv::Mat& img, const Eigen::Vector2d& pc) const;
  const cv::Mat img() const { return curr_img; }

 private:
  void OpticalFlowTrack();
  void SelectNewPoints(const PointCloudXYZI::Ptr& cloud);
  void ObservationModel(const state_ikfom& x);
  void IESKF(esekfom::esekf<state_ikfom, 12, input_ikfom>& kf);
  void UpdateStateWithKF(const state_ikfom& x);
  int32_t GetGridIndex(const Eigen::Vector2d& uv) const;
  bool InFov(const Eigen::Vector2d& uv, int32_t boundry) const;
  float ShiTomasiScore(const cv::Mat& img, const Eigen::Vector2d& uv) const;

 public:
  vk::AbstractCamera* cam = nullptr;
  Sophus::SE3 tfwc;
 private:
  // pointcloud
  cv::Mat last_img, last_gray, last_score, last_desc;
  cv::Mat curr_img, curr_gray, curr_score, curr_desc;
  std::vector<cv::Point2f> last_uvs;
  std::vector<cv::Point2f> curr_uvs;
  PointCloudXYZRGB::Ptr curr_cloud = nullptr;
  std::vector<uchar> track_status;
  PointCloudXYZRGB::Ptr color_cloud = nullptr;
  // visual grid
  std::vector<float> curr_scores;
  std::vector<bool> exist_status;
  std::vector<bool> new_status;
  std::vector<cv::Point2f> curr_uvs_grid;
  PointCloudXYZRGB::Ptr curr_cloud_grid = nullptr;
  std::vector<Eigen::Vector2d> residuals;
  std::vector<Eigen::Matrix<double, 2, 6>> jacobians;
  double meas_cov = 0.01;
  int32_t max_iter_num = 3;
  // camera and image
  int32_t grid_size, grid_num;
  int32_t width, height;
  double fx, fy, cx, cy;
  Eigen::Vector3d tcl;
  Eigen::Matrix3d rcl;
  Sophus::SE3 tfcl;
  Sophus::SE3 tfci;
  // publisher
  ros::Publisher pub_tracked_img;
  ros::Publisher pub_visual_points;
  ros::Publisher pub_color_cloud;
};

typedef boost::shared_ptr<VisualPNP> VisualPNPPtr;

}  // namespace visual_pnp
