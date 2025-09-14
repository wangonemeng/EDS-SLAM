#include "visual_pnp.h"

#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <vikit/camera_loader.h>

namespace visual_pnp {

namespace {
using CovMat = Eigen::Matrix<double, state_ikfom::DOF, state_ikfom::DOF>;
}  // namespace

VisualPNP::VisualPNP(ros::NodeHandle& nh) {
  std::cout << "Init VisualPNP ... ";
  curr_cloud.reset(new PointCloudXYZRGB);
  curr_cloud_grid.reset(new PointCloudXYZRGB);
  color_cloud.reset(new PointCloudXYZRGB);
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
  grid_num = (width / grid_size + 1) * (height / grid_size + 1);
  curr_cloud->resize(grid_num);
  curr_cloud_grid->resize(grid_num);
  // last_uvs.resize(grid_num);
  // curr_uvs.resize(grid_num);
  curr_uvs_grid.resize(grid_num);
  curr_scores.assign(grid_num, 0.f);
  exist_status.assign(grid_num, false);
  new_status.assign(grid_num, false);
  track_status.assign(grid_num, 0u);
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
  pub_tracked_img = nh.advertise<sensor_msgs::Image>("/pnp_tracked_img", 100);
  pub_visual_points = nh.advertise<sensor_msgs::PointCloud2>("/pnp_visual_points", 100);
  pub_color_cloud = nh.advertise<sensor_msgs::PointCloud2>("/color_cloud", 100);
  std::cout << "success." << std::endl;
}

bool VisualPNP::InFov(const Eigen::Vector2d& uv, int32_t boundry) const {
  const Eigen::Vector2i& obs = uv.cast<int32_t>();
  if (obs.x() >= boundry && obs.x() < width - boundry &&
      obs.y() >= boundry && obs.y() < height - boundry) {
    return true;
  }
  return false;
}

void VisualPNP::UpdateStateWithKF(const state_ikfom& x) {
  Sophus::SE3 tfil = Sophus::SE3(x.offset_R_L_I.matrix(), x.offset_T_L_I.matrix());
  Sophus::SE3 tfwi = Sophus::SE3(x.rot.matrix(), x.pos.matrix());
  tfci = tfcl * tfil.inverse();
  tfwc = tfwi * tfci.inverse();
  return;
}

void VisualPNP::UpdateState(esekfom::esekf<state_ikfom, 12, input_ikfom>& kf,
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
  last_uvs.swap(curr_uvs_grid);
  *curr_cloud = *curr_cloud_grid;
  int32_t i = 0, j = 0;
  for (; i < exist_status.size(); ++i) {
    if (exist_status.at(i)) {
      last_uvs.at(j) = last_uvs.at(i);
      curr_cloud->at(j) = curr_cloud->at(i);
      ++j;
    }
  }
  last_uvs.resize(j);
  curr_cloud->resize(j);
  curr_uvs.clear();
  curr_uvs_grid.assign(grid_num, cv::Point2f());
  track_status.assign(grid_num, 0);
  new_status.assign(grid_num, false);
  exist_status.assign(grid_num, false);
  residuals.clear();
  jacobians.clear();
  // reset state
  UpdateStateWithKF(kf.get_x());
  const auto t1 = std::chrono::steady_clock::now();
  OpticalFlowTrack();
  const auto t2 = std::chrono::steady_clock::now();
  IESKF(kf);
  const auto t3 = std::chrono::steady_clock::now();
  SelectNewPoints(cloud);
  const auto t4 = std::chrono::steady_clock::now();
  UpdateStateWithKF(kf.get_x());
  ROS_WARN("Visual PNP update state: clear state %f ms, optical track %f ms, ieskf %f ms, incremental %f ms",
           std::chrono::duration<double>(t1 - t0).count() * 1000.,
           std::chrono::duration<double>(t2 - t1).count() * 1000.,
           std::chrono::duration<double>(t3 - t2).count() * 1000.,
           std::chrono::duration<double>(t4 - t3).count() * 1000.);
}

void VisualPNP::OpticalFlowTrack() {
  if (last_gray.empty() || curr_gray.empty() || last_uvs.empty()) {
    curr_cloud_grid->clear();
    curr_cloud->clear();
    curr_cloud_grid->resize(grid_num);
    curr_cloud->resize(grid_num);
    return;
  }
  vector<float> err;

  curr_uvs.reserve(curr_cloud->size());
  for (const auto& pt : curr_cloud->points) {
    const Eigen::Vector3d pt_w = {pt.x, pt.y, pt.z};
    const Eigen::Vector3d& pt_c = tfwc.inverse() * pt_w;
    const Eigen::Vector2d& uv = cam->world2cam(pt_c);
    curr_uvs.emplace_back(cv::Point2f(uv.x(), uv.y()));
  }

  cv::calcOpticalFlowPyrLK(last_gray, curr_gray, last_uvs, curr_uvs, track_status, err, cv::Size(21, 21), 3,
                           cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 30, (0.01)),
                           cv::OPTFLOW_USE_INITIAL_FLOW);

  assert(last_uvs.size() == curr_uvs.size() && last_uvs.size() == curr_cloud->size());
  int32_t i = 0, j = 0;
  for (; i < track_status.size(); ++i) {
    if (track_status.at(i)) {
      last_uvs.at(j) = last_uvs.at(i);
      curr_uvs.at(j) = curr_uvs.at(i);
      curr_cloud->points.at(j) = curr_cloud->points.at(i);
      ++j;
    }
  }
  last_uvs.resize(j);
  curr_uvs.resize(j);
  curr_cloud->points.resize(j);
  // std::cout << "Optical Flow track: " << j << " / " << i << " points " << std::endl;

  cv::findFundamentalMat(last_uvs, curr_uvs, cv::FM_RANSAC, 1.0, 0.999, track_status);
  i = 0, j = 0;
  for (; i < track_status.size(); ++i) {
    if (track_status.at(i)) {
      last_uvs.at(j) = last_uvs.at(i);
      curr_uvs.at(j) = curr_uvs.at(i);
      curr_cloud->points.at(j) = curr_cloud->points.at(i);
      ++j;
    }
  }
  last_uvs.resize(j);
  curr_uvs.resize(j);
  curr_cloud->points.resize(j);
  // std::cout << "Reject outlier with FundamentalMat: " << j << " / " << i << " points " << std::endl;

  for (int i = 0; i < curr_uvs.size(); ++i) {
    int32_t index = GetGridIndex(Eigen::Vector2d(curr_uvs.at(i).x, curr_uvs.at(i).y));
    if (index < 0) continue;
    exist_status.at(index) = true;
    curr_uvs_grid.at(index) = curr_uvs.at(i);
    curr_cloud_grid->at(index) = curr_cloud->at(i);
    curr_cloud_grid->at(index).r = 0;
    curr_cloud_grid->at(index).g = 0;
    curr_cloud_grid->at(index).b = 255;
  }

  return;
}

int32_t VisualPNP::GetGridIndex(const Eigen::Vector2d& uv) const {
  int32_t ret = -1;
  if (!InFov(uv, 1)) return ret;
  int32_t x = uv.x() / grid_size;
  int32_t y = uv.y() / grid_size;
  ret = y * (width / grid_size) + x;
  assert(ret < grid_num);
  return ret;
}

float VisualPNP::ShiTomasiScore(const cv::Mat& img, const Eigen::Vector2d& uv) const {
  assert(img.type() == CV_8UC1);
  int u = uv.x();
  int v = uv.y();
  float dXX = 0.0;
  float dYY = 0.0;
  float dXY = 0.0;
  const int halfbox_size = 4;
  const int box_size = 2 * halfbox_size;
  const int box_area = box_size * box_size;
  const int x_min = u - halfbox_size;
  const int x_max = u + halfbox_size;
  const int y_min = v - halfbox_size;
  const int y_max = v + halfbox_size;

  if (x_min < 1 || x_max >= img.cols - 1 || y_min < 1 || y_max >= img.rows - 1)
    return 0.0;  // patch is too close to the boundary

  const int stride = img.step.p[0];
  for (int y = y_min; y < y_max; ++y) {
    const uint8_t* ptr_left = img.data + stride * y + x_min - 1;
    const uint8_t* ptr_right = img.data + stride * y + x_min + 1;
    const uint8_t* ptr_top = img.data + stride * (y - 1) + x_min;
    const uint8_t* ptr_bottom = img.data + stride * (y + 1) + x_min;
    for (int x = 0; x < box_size; ++x, ++ptr_left, ++ptr_right, ++ptr_top, ++ptr_bottom) {
      float dx = *ptr_right - *ptr_left;
      float dy = *ptr_bottom - *ptr_top;
      dXX += dx * dx;
      dYY += dy * dy;
      dXY += dx * dy;
    }
  }

  // Find and return smaller eigenvalue:
  dXX = dXX / (2.0 * box_area);
  dYY = dYY / (2.0 * box_area);
  dXY = dXY / (2.0 * box_area);
  return 0.5 * (dXX + dYY - sqrt((dXX + dYY) * (dXX + dYY) - 4 * (dXX * dYY - dXY * dXY)));
}

void VisualPNP::SelectNewPoints(const PointCloudXYZI::Ptr& cloud) {
  if (cloud == nullptr) return;
  color_cloud->clear();
  color_cloud->reserve(cloud->size());
  const Sophus::SE3& tfcw = tfwc.inverse();
  const Eigen::Matrix3d& rcw = tfcw.rotation_matrix();
  const Eigen::Vector3d& tcw = tfcw.translation();
  for (const auto& pt : cloud->points) {
    const Eigen::Vector3d pt_w{pt.x, pt.y, pt.z};
    const Eigen::Vector3d& pt_c = rcw * pt_w + tcw;
    if (pt_c.z() < 1.e-6) continue;
    const Eigen::Vector2d& uv = cam->world2cam(pt_c);
    if (!InFov(uv, 1)) continue;
    {
      // color cloud
      Eigen::Vector3f bgr = getpixel(curr_img, uv);
      pcl::PointXYZRGB ptrgb;
      ptrgb.x = pt.x;
      ptrgb.y = pt.y;
      ptrgb.z = pt.z;
      ptrgb.r = bgr.z();
      ptrgb.g = bgr.y();
      ptrgb.b = bgr.x();
      color_cloud->emplace_back(ptrgb);
    }
    int32_t index = GetGridIndex(uv);
    if (index < 0) continue;
    float score = ShiTomasiScore(curr_gray, uv);
    if (exist_status.at(index) && score <= curr_scores.at(index)) continue;

    curr_scores.at(index) = score;
    exist_status.at(index) = true;
    new_status.at(index) = true;
    curr_uvs_grid.at(index) = cv::Point2f(uv.x(), uv.y());
    curr_cloud_grid->at(index).x = pt.x;
    curr_cloud_grid->at(index).y = pt.y;
    curr_cloud_grid->at(index).z = pt.z;
    curr_cloud_grid->at(index).r = 255;
    curr_cloud_grid->at(index).g = 0;
    curr_cloud_grid->at(index).b = 0;
  }
}

void VisualPNP::ObservationModel(const state_ikfom& x) {
  if (curr_cloud->empty() || curr_uvs.empty()) return;
  assert(curr_cloud->size() == curr_uvs.size());
  UpdateStateWithKF(x);

  residuals.resize(curr_cloud->size());
  jacobians.resize(curr_cloud->size());
  for (int32_t i = 0; i < curr_cloud->size(); ++i) {
    residuals.at(i).setZero();
    jacobians.at(i).setZero();
    const Eigen::Vector3d pt_w = {curr_cloud->at(i).x, curr_cloud->at(i).y,
                                  curr_cloud->at(i).z};
    const Eigen::Vector3d& pt_c = tfwc.inverse() * pt_w;
    const Eigen::Vector2d& uv = cam->world2cam(pt_c);
    residuals.at(i) = Eigen::Vector2d(curr_uvs.at(i).x, curr_uvs.at(i).y) - uv;
    // std::cout << residuals.at(i).transpose() << " ";
    const double z_inv = 1. / pt_c.z();
    const double z_inv_square = z_inv * z_inv;
    Eigen::Matrix<double, 2, 3> d_uv_ptc = Eigen::Matrix<double, 2, 3>::Zero();
    d_uv_ptc(0, 0) = fx * z_inv;
    d_uv_ptc(0, 2) = -fx * pt_c.x() * z_inv_square;
    d_uv_ptc(1, 1) = fy * z_inv;
    d_uv_ptc(1, 2) = -fy * pt_c.y() * z_inv_square;
    Eigen::Matrix3d d_ptc_rot = Sophus::SO3::hat(pt_c) * tfci.rotation_matrix() +
                                tfci.rotation_matrix() * Sophus::SO3::hat(tfci.inverse().translation());
    Eigen::Matrix3d d_ptc_pos = -tfci.rotation_matrix() * x.rot.inverse().matrix();
    jacobians.at(i).block<2, 3>(0, 0) = d_uv_ptc * d_ptc_pos;
    jacobians.at(i).block<2, 3>(0, 3) = d_uv_ptc * d_ptc_rot;
  }
}

void VisualPNP::IESKF(esekfom::esekf<state_ikfom, 12, input_ikfom>& kf) {
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

void VisualPNP::Publish() const {
  // publish tracked image
  cv::Mat tracked_img = curr_img.clone();
  for (int32_t i = 0; i < curr_uvs_grid.size(); ++i) {
    cv::Scalar color = cv::Scalar(255, 0, 0);
    if (new_status.at(i))
      color = cv::Scalar(0, 0, 255);
    cv::circle(tracked_img, curr_uvs_grid.at(i), 4, color, -1);
  }
  for (int32_t i = 0; i < curr_uvs.size(); ++i) {
    cv::circle(tracked_img, curr_uvs.at(i), 4, cv::Scalar(255, 0, 0), -1);
    cv::line(tracked_img, curr_uvs.at(i), last_uvs.at(i), cv::Scalar(0, 255, 0), 1);
  }
  for (int row = 0; row < tracked_img.rows; row += grid_size) {
    cv::line(tracked_img, cv::Point(0, row), cv::Point(tracked_img.cols - 1, row),
             cv::Scalar(255, 255, 255), 1);
  }
  for (int col = 0; col < tracked_img.cols; col += grid_size) {
    cv::line(tracked_img, cv::Point(col, 0), cv::Point(col, tracked_img.rows - 1),
             cv::Scalar(255, 255, 255), 1);
  }
  const auto& msg_type2 = sensor_msgs::image_encodings::BGR8;
  sensor_msgs::ImagePtr msg2 =
      cv_bridge::CvImage(std_msgs::Header(), msg_type2, tracked_img).toImageMsg();
  pub_tracked_img.publish(msg2);
  // publish point cloud
  PointCloudXYZRGB::Ptr pub_cloud_grid{new PointCloudXYZRGB()};
  for (int32_t i = 0; i < exist_status.size(); ++i) {
    if (exist_status.at(i)) {
      pub_cloud_grid->emplace_back(curr_cloud_grid->at(i));
    }
  }
  sensor_msgs::PointCloud2 visual_cloud;
  pcl::toROSMsg(*pub_cloud_grid, visual_cloud);
  visual_cloud.header.stamp = ros::Time().now();
  visual_cloud.header.frame_id = "camera_init";
  pub_visual_points.publish(visual_cloud);
  // publish color cloud
  sensor_msgs::PointCloud2 dense_cloud;
  pcl::toROSMsg(*color_cloud, dense_cloud);
  dense_cloud.header.stamp = ros::Time().now();
  dense_cloud.header.frame_id = "camera_init";
  pub_color_cloud.publish(dense_cloud);
}

Eigen::Vector3f VisualPNP::getpixel(const cv::Mat& img, const Eigen::Vector2d& pc) const {
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
  float B = w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[0 + 3] +
            w_ref_bl * img_ptr[width * 3] + w_ref_br * img_ptr[width * 3 + 0 + 3];
  float G = w_ref_tl * img_ptr[1] + w_ref_tr * img_ptr[1 + 3] +
            w_ref_bl * img_ptr[1 + width * 3] + w_ref_br * img_ptr[width * 3 + 1 + 3];
  float R = w_ref_tl * img_ptr[2] + w_ref_tr * img_ptr[2 + 3] +
            w_ref_bl * img_ptr[2 + width * 3] + w_ref_br * img_ptr[width * 3 + 2 + 3];
  Eigen::Vector3f pixel(B, G, R);
  return pixel;
}

}  // namespace visual_pnp
