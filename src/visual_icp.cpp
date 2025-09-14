#include "visual_icp.h"

#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <vikit/camera_loader.h>

namespace visual_icp {

namespace {
using CovMat = Eigen::Matrix<double, state_ikfom::DOF, state_ikfom::DOF>;
}  // namespace

VisualICP::VisualICP(ros::NodeHandle& nh) {
  std::cout << "Init VisualICP ... ";
  last_cloud.reset(new PointCloudXYZRGB);
  curr_cloud.reset(new PointCloudXYZRGB);
  curr_cloud_cam.reset(new PointCloudXYZRGB);
  new_cloud.reset(new PointCloudXYZRGB);
  // read parameters
  ros::param::get("/camera/cam_width", width);
  ros::param::get("/camera/cam_height", height);
  ros::param::get("/camera/cam_fx", fx);
  ros::param::get("/camera/cam_fy", fy);
  ros::param::get("/camera/cam_cx", cx);
  ros::param::get("/camera/cam_cy", cy);
  ros::param::get("/camera/grid_size", grid_size);
  ros::param::get("/camera/visual_point_cov", meas_cov);
  ros::param::get("/max_iteration", max_iter_num);
  grid_num = width / grid_size * height / grid_size;
  if (!vk::camera_loader::loadFromRosNs(ros::this_node::getName(), cam))
    throw std::runtime_error("Camera model not correctly specified.");
  std::vector<double> cam_extrinT(3, 0.0);
  std::vector<double> cam_extrinR(9, 0.0);
  ros::param::get("/mapping/Pcl", cam_extrinT);
  ros::param::get("/mapping/Rcl", cam_extrinR);
  rcl << MAT_FROM_ARRAY(cam_extrinR);
  tcl << VEC_FROM_ARRAY(cam_extrinT);
  tfcl = Sophus::SE3(rcl, tcl);
  // advertise publisher
  pub_depth_img = nh.advertise<sensor_msgs::Image>("/depth_img", 100);
  pub_tracked_img = nh.advertise<sensor_msgs::Image>("/tracked_img", 100);
  pub_match_marker = nh.advertise<sensor_msgs::PointCloud2>("/match_marker", 100);
  pub_visual_points = nh.advertise<sensor_msgs::PointCloud2>("/visual_points", 100);
  std::cout << "success." << std::endl;
}

bool VisualICP::InFov(const Eigen::Vector2d& uv, int32_t boundry) const {
  const Eigen::Vector2i& obs = uv.cast<int32_t>();
  if (obs.x() >= boundry && obs.x() < width - boundry &&
      obs.y() >= boundry && obs.y() < height - boundry) {
    return true;
  }
  return false;
}

void VisualICP::UpdateStateWithKF(const state_ikfom& x) {
  Sophus::SE3 tfil = Sophus::SE3(x.offset_R_L_I.matrix(), x.offset_T_L_I.matrix());
  Sophus::SE3 tfwi = Sophus::SE3(x.rot.matrix(), x.pos.matrix());
  tfci = tfcl * tfil.inverse();
  tfwc = tfwi * tfci.inverse();
  return;
}

void VisualICP::UpdateState(esekfom::esekf<state_ikfom, 12, input_ikfom>& kf,
                            const cv::Mat& img, const PointCloudXYZI::Ptr& cloud) {
  const auto t0 = std::chrono::steady_clock::now();
  if (curr_img.empty()) {
    curr_img = img.clone();
    return;
  }
  // reset image, cloud, pixels
  last_img = curr_img.clone();
  last_gray = curr_gray.clone();
  curr_img = img.clone();
  if (img.rows != height || img.cols != width) {
    cv::resize(img, curr_img, cv::Size(width, height));
  }
  cv::cvtColor(curr_img, curr_gray, cv::COLOR_BGR2GRAY);
  *last_cloud = *curr_cloud + *new_cloud;
  last_uvs.swap(curr_uvs);
  last_uvs.insert(last_uvs.end(), new_uvs.begin(), new_uvs.end());
  curr_cloud->clear();
  new_cloud->clear();
  curr_uvs.clear();
  new_uvs.clear();
  residuals.clear();
  jacobians.clear();
  // reset state
  UpdateStateWithKF(kf.get_x());
  GetDepthImage(cloud);
  const auto t1 = std::chrono::steady_clock::now();
  OpticalFlowTrack();
  const auto t2 = std::chrono::steady_clock::now();
  IESKF(kf);
  const auto t3 = std::chrono::steady_clock::now();
  ExtractNewPoints();
  RecoverDepth(depth_img, new_uvs, new_cloud, Eigen::Vector3d(255, 0, 0));
  const auto t4 = std::chrono::steady_clock::now();
  UpdateStateWithKF(kf.get_x());
  ROS_WARN("Visual ICP update state: depth image %f ms, optical track %f ms, ieskf %f ms, incremental %f ms",
           std::chrono::duration<double>(t1 - t0).count() * 1000.,
           std::chrono::duration<double>(t2 - t1).count() * 1000.,
           std::chrono::duration<double>(t3 - t2).count() * 1000.,
           std::chrono::duration<double>(t4 - t3).count() * 1000.);
}

void VisualICP::GetDepthImage(const PointCloudXYZI::Ptr& cloud) {
  depth_img = cv::Mat::zeros(height, width, CV_64FC3);
  const Sophus::SE3& tfcw = tfwc.inverse();
  const Eigen::Matrix3d& rcw = tfcw.rotation_matrix();
  const Eigen::Vector3d& tcw = tfcw.translation();
  for (const auto& pt : cloud->points) {
    const Eigen::Vector3d pt_w{pt.x, pt.y, pt.z};
    const Eigen::Vector3d& pt_c = rcw * pt_w + tcw;
    if (pt_c.z() < 1.e-6) continue;
    const Eigen::Vector2d& uv = cam->world2cam(pt_c);
    if (!InFov(uv, 1)) continue;
    const cv::Vec3d pixel = {pt_c.x(), pt_c.y(), pt_c.z()};
    depth_img.at<cv::Vec3d>(uv.cast<int32_t>().y(),
                            uv.cast<int32_t>().x()) = pixel;
  }
}

void VisualICP::OpticalFlowTrack() {
  if (last_gray.empty() || curr_gray.empty() || last_uvs.empty()) {
    curr_cloud_cam->clear();
    curr_cloud->clear();
    last_cloud->clear();
    return;
  }
  vector<uchar> status;
  vector<float> err;
  cv::calcOpticalFlowPyrLK(last_gray, curr_gray, last_uvs, curr_uvs, status, err, cv::Size(21, 21), 3);
  int32_t i = 0, j = 0;
  assert(last_uvs.size() == curr_uvs.size() && last_uvs.size() == last_cloud->size());
  for (; i < status.size(); ++i) {
    if (status.at(i)) {
      last_uvs.at(j) = last_uvs.at(i);
      curr_uvs.at(j) = curr_uvs.at(i);
      last_cloud->points.at(j) = last_cloud->points.at(i);
      ++j;
    }
  }
  last_uvs.resize(j);
  curr_uvs.resize(j);
  last_cloud->points.resize(j);
  // std::cout << "Optical Flow track: " << j << " / " << i << " points " << std::endl;

  status.clear();
  cv::findFundamentalMat(last_uvs, curr_uvs, cv::FM_RANSAC, 1.0, 0.99, status);
  i = 0, j = 0;
  for (; i < status.size(); ++i) {
    if (status.at(i)) {
      last_uvs.at(j) = last_uvs.at(i);
      curr_uvs.at(j) = curr_uvs.at(i);
      last_cloud->points.at(j) = last_cloud->points.at(i);
      ++j;
    }
  }
  last_uvs.resize(j);
  curr_uvs.resize(j);
  last_cloud->points.resize(j);
  // std::cout << "Reject outlier with FundamentalMat: " << j << " / " << i << " points " << std::endl;

  status.resize(curr_uvs.size());
  curr_cloud->clear();
  curr_cloud->resize(curr_uvs.size());
  curr_cloud_cam->clear();
  curr_cloud_cam->resize(curr_uvs.size());
  for (int32_t k = 0; k < curr_uvs.size(); ++k) {
    const auto& uv = curr_uvs.at(k);
    double min_dist = 1000000.;
    double depth = -1.;
    double range = -1.;
    for (int32_t u = -nn_patch_size / 2; u <= nn_patch_size / 2; ++u) {
      for (int32_t v = -nn_patch_size / 2; v <= nn_patch_size / 2; ++v) {
        if (!InFov(Eigen::Vector2d(uv.x + u, uv.y + v), 1)) continue;
        cv::Vec3d xyz = depth_img.at<cv::Vec3d>(cv::Point(uv.x + u, uv.y + v));
        if (xyz[2] > 0. && std::hypot(u, v) < min_dist) {
          min_dist = std::hypot(u, v);
          depth = xyz[2];
          range = cv::norm(xyz);
        }
      }
    }
    if (range < 0.) {
      status.at(k) = 0;
      continue;
    }
    const Eigen::Vector3d& pt_c = cam->cam2world(Eigen::Vector2d(uv.x, uv.y)) * range;
    const Eigen::Vector3d& pt_w = tfwc * pt_c;
    pcl::PointXYZRGB point;
    point.x = pt_w.x();
    point.y = pt_w.y();
    point.z = pt_w.z();
    point.r = 0;
    point.g = 0;
    point.b = 255;
    curr_cloud->at(k) = point;
    point.x = pt_c.x();
    point.y = pt_c.y();
    point.z = pt_c.z();
    curr_cloud_cam->at(k) = point;
    status.at(k) = 255;
  }
  i = 0, j = 0;
  for (; i < status.size(); ++i) {
    if (status.at(i)) {
      last_uvs.at(j) = last_uvs.at(i);
      curr_uvs.at(j) = curr_uvs.at(i);
      curr_cloud->points.at(j) = curr_cloud->points.at(i);
      curr_cloud_cam->points.at(j) = curr_cloud_cam->points.at(i);
      last_cloud->points.at(j) = last_cloud->points.at(i);
      last_cloud->points.at(j).r = 0;
      last_cloud->points.at(j).g = 255;
      last_cloud->points.at(j).b = 0;
      ++j;
    }
  }
  last_uvs.resize(j);
  curr_uvs.resize(j);
  curr_cloud->resize(j);
  curr_cloud_cam->resize(j);
  last_cloud->resize(j);
  // std::cout << "Recover Depth: " << j << " / " << i << " points " << std::endl;

  return;
}

void VisualICP::ExtractNewPoints() {
  cv::Mat mask = cv::Mat(height, width, CV_8UC1, cv::Scalar(255));
  for (const auto& uv : curr_uvs) {
    cv::circle(mask, uv, grid_size, 0, -1);
  }
  if (curr_uvs.size() > grid_num) return;
  cv::goodFeaturesToTrack(curr_gray, new_uvs, grid_num - curr_uvs.size(), 0.001, grid_size, mask);
  if (log_info) {
    // std::cout << "Extract " << new_uvs.size() << " Shi-Tomas corners." << std::endl;
  }
}

void VisualICP::RecoverDepth(const cv::Mat& depth_img,
                             std::vector<cv::Point2f>& uvs,
                             PointCloudXYZRGB::Ptr cloud,
                             const Eigen::Vector3d& color) const {
  if (uvs.empty() || depth_img.empty()) return;
  cloud->clear();
  cloud->reserve(uvs.size());
  std::vector<cv::Point2f> effect_uvs;
  effect_uvs.reserve(uvs.size());
  for (const auto& uv : uvs) {
    double min_dist = 1000000.;
    double depth = -1.;
    double range = -1.;
    for (int32_t u = -nn_patch_size / 2; u <= nn_patch_size / 2; ++u) {
      for (int32_t v = -nn_patch_size / 2; v <= nn_patch_size / 2; ++v) {
        if (!InFov(Eigen::Vector2d(uv.x + u, uv.y + v), 1)) continue;
        cv::Vec3d xyz = depth_img.at<cv::Vec3d>(cv::Point(uv.x + u, uv.y + v));
        if (xyz[2] > 0. && std::hypot(u, v) < min_dist) {
          min_dist = std::hypot(u, v);
          depth = xyz[2];
          range = cv::norm(xyz);
        }
      }
    }
    if (range < 0.) continue;
    const Eigen::Vector3d& pt_c = cam->cam2world(Eigen::Vector2d(uv.x, uv.y)) * range;
    const Eigen::Vector3d& pt_w = tfwc * pt_c;
    pcl::PointXYZRGB point;
    point.x = pt_w.x();
    point.y = pt_w.y();
    point.z = pt_w.z();
    point.r = color.x();
    point.g = color.y();
    point.b = color.z();
    cloud->emplace_back(point);
    effect_uvs.emplace_back(uv);
  }
  effect_uvs.swap(uvs);
  // std::cout << "Recover " << cloud->size() << " points from " << effect_uvs.size() << std::endl;
}

void VisualICP::ObservationModel(const state_ikfom& x) {
  if (last_cloud->empty() || curr_cloud_cam->empty()) return;
  assert(last_cloud->size() == curr_cloud_cam->size());
  UpdateStateWithKF(x);

  residuals.resize(curr_cloud_cam->size());
  jacobians.resize(curr_cloud_cam->size());
  for (int32_t i = 0; i < curr_cloud_cam->size(); ++i) {
    const Eigen::Vector3d last_pt = {last_cloud->at(i).x, last_cloud->at(i).y,
                                     last_cloud->at(i).z};
    const Eigen::Vector3d pt_cam = {curr_cloud_cam->at(i).x, curr_cloud_cam->at(i).y,
                                    curr_cloud_cam->at(i).z};
    const Eigen::Vector3d& pt_world = tfwc * pt_cam;
    residuals.at(i) = last_pt - pt_world;
    const Eigen::Vector3d& pt_body = tfci.inverse() * pt_cam;
    jacobians.at(i).block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    jacobians.at(i).block<3, 3>(0, 3) = Sophus::SO3::hat(-x.rot.matrix() * pt_body);
    // residuals.at(i) *= 100.;
    // jacobians.at(i) *= 100.;
  }
}

void VisualICP::IESKF(esekfom::esekf<state_ikfom, 12, input_ikfom>& kf) {
  CovMat predict_cov = kf.get_P();
  state_ikfom predict_x = kf.get_x();
  for (int32_t iter = 0; iter < max_iter_num; ++iter) {
    kf.change_P(predict_cov);
    state_ikfom x = kf.get_x();
    CovMat cov = kf.get_P();
    ObservationModel(x);
    Eigen::Matrix<double, state_ikfom::DOF, 1> dx;
    x.boxminus(dx, predict_x);
    CovMat hth = CovMat::Zero();
    for (const auto& jaco : jacobians) {
      hth.block<6, 6>(0, 0) += jaco.transpose() * jaco / meas_cov;
    }
    CovMat P_tmp = cov.inverse() + hth;
    CovMat P_tmp_inv = P_tmp.inverse();
    Eigen::Matrix<double, state_ikfom::DOF, 1>
        K_x = Eigen::Matrix<double, state_ikfom::DOF, 1>::Zero();
    for (int32_t i = 0; i < jacobians.size(); ++i) {
      K_x += P_tmp_inv.block<state_ikfom::DOF, 6>(0, 0) * jacobians.at(i).transpose() / meas_cov * residuals.at(i);
    }
    CovMat K_H = P_tmp_inv * hth;
    Eigen::Matrix<double, state_ikfom::DOF, 1>
        dx_new = K_x + (K_H - CovMat::Identity()) * dx;
    x.boxplus(dx_new);
    // std::cout << x.pos.transpose() << std::endl;
    cov = (CovMat::Identity() - K_H) * cov;
    kf.change_x(x);
    kf.change_P(cov);
  }
}

void VisualICP::Publish() const {
  // publish depth image
  std::vector<cv::Mat> channels(3);
  cv::split(depth_img, channels);
  cv::Mat depth_single = channels[2];
  channels[2].convertTo(depth_single, CV_32FC1);
  const auto& msg_type1 = sensor_msgs::image_encodings::TYPE_32FC1;
  sensor_msgs::ImagePtr msg1 =
      cv_bridge::CvImage(std_msgs::Header(), msg_type1, depth_single).toImageMsg();
  pub_depth_img.publish(msg1);
  // publish tracked image
  cv::Mat tracked_img = curr_img.clone();
  for (const auto& uv : new_uvs) {
    cv::circle(tracked_img, uv, 4, cv::Scalar(0, 0, 255), -1);
  }
  for (int32_t i = 0; i < curr_uvs.size(); ++i) {
    cv::circle(tracked_img, curr_uvs.at(i), 4, cv::Scalar(255, 0, 0), -1);
    cv::line(tracked_img, curr_uvs.at(i), last_uvs.at(i), cv::Scalar(0, 255, 0), 1);
  }
  const auto& msg_type2 = sensor_msgs::image_encodings::BGR8;
  sensor_msgs::ImagePtr msg2 =
      cv_bridge::CvImage(std_msgs::Header(), msg_type2, tracked_img).toImageMsg();
  pub_tracked_img.publish(msg2);
  // publish point cloud
  sensor_msgs::PointCloud2 visual_cloud;
  for (int32_t i = 0; i < curr_cloud->size(); ++i) {
    Eigen::Vector3d pt_c = {curr_cloud_cam->at(i).x, curr_cloud_cam->at(i).y,
                            curr_cloud_cam->at(i).z};
    Eigen::Vector3d pt_w = tfwc * pt_c;
    curr_cloud->at(i).x = pt_w.x();
    curr_cloud->at(i).y = pt_w.y();
    curr_cloud->at(i).z = pt_w.z();
  }
  pcl::toROSMsg(*new_cloud + *curr_cloud + *last_cloud, visual_cloud);
  visual_cloud.header.stamp = ros::Time().now();
  visual_cloud.header.frame_id = "camera_init";
  pub_visual_points.publish(visual_cloud);
}

Eigen::Vector3f VisualICP::getpixel(const cv::Mat& img, const Eigen::Vector2d& pc) const {
  const float u_ref = pc[0];
  const float v_ref = pc[1];
  const int u_ref_i = floorf(pc[0]);
  const int v_ref_i = floorf(pc[1]);
  const float subpix_u_ref = (u_ref - u_ref_i);
  const float subpix_v_ref = (v_ref - v_ref_i);
  const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
  const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
  const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
  const float w_ref_br = subpix_u_ref * subpix_v_ref;
  uint8_t* img_ptr = (uint8_t*)img.data + ((v_ref_i)*width + (u_ref_i)) * 3;
  float B = w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[0 + 3] + w_ref_bl * img_ptr[width * 3] + w_ref_br * img_ptr[width * 3 + 0 + 3];
  float G = w_ref_tl * img_ptr[1] + w_ref_tr * img_ptr[1 + 3] + w_ref_bl * img_ptr[1 + width * 3] + w_ref_br * img_ptr[width * 3 + 1 + 3];
  float R = w_ref_tl * img_ptr[2] + w_ref_tr * img_ptr[2 + 3] + w_ref_bl * img_ptr[2 + width * 3] + w_ref_br * img_ptr[width * 3 + 2 + 3];
  Eigen::Vector3f pixel(B, G, R);
  return pixel;
}

}  // namespace visual_icp
