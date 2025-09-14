#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
// #include <common_lib.h>
#include <image_transport/image_transport.h>
#include "IMU_Processing.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/segmentation/extract_clusters.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver/CustomMsg.h>
#include "preprocess.h"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <vikit/camera_loader.h>
#include <ikd-Tree/ikd_Tree.h>
#include <yolo_segment/yolo_segment.h>
#include "visual_icp.h"
#include "visual_pnp.h"
#include <chrono>

#define INIT_TIME (0.1)
#define MAXN (720000)
#define PUBFRAME_PERIOD (20)

double LASER_POINT_COV = 0.001;
/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
std::chrono::high_resolution_clock::time_point img_process_start, img_process_end;
double img_process_time = 0.0;  // 单位：毫秒
int img_frame_count = 0;        // 统计处理的图像帧数
double lidar_process_time = 0.0;  // 单位：毫秒
int lidar_frame_count = 0;     
double remove_process_time = 0.0;  // 单位：毫秒
int remove_frame_count = 0;  
 double map_process_time = 0.0;  // 单位：毫秒
int map_frame_count = 0;   
 double imu_process_time = 0.0;  // 单位：毫秒
int imu_frame_count = 0; 
/**************************/

float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer;
condition_variable sig_buffer;

string map_file_path, lid_topic, imu_topic, img_topic, config_file;

visual_icp::VisualICPPtr vicp = nullptr;
visual_pnp::VisualPNPPtr vpnp = nullptr;
std::shared_ptr<YoloSeg> yolo_segmenter = nullptr;

double last_timestamp_lidar = 0, last_timestamp_imu = -1.0, last_timestamp_img = -1.0, last_timestamp_boxes = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0;
int effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
bool point_selected_surf[100000] = {0};
bool lidar_pushed = false, flg_first_scan = true, flg_exit = false, flg_EKF_inited = false;
bool scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;

double first_img_time = -1.0, first_lidar_time = -1.0;
bool flg_first_img = true;
int img_en = 1, lidar_en = 1, debug = 0;
bool fast_lio_is_ready = false;
int grid_size, patch_size;
double cam_fx, cam_fy, cam_cx, cam_cy;

vector<BoxPointType> cub_needrm;
vector<PointVector> Nearest_Points;
vector<double> extrinT(3, 0.0);
vector<double> extrinR(9, 0.0);
vector<double> cam_extrinT(3, 0.0);
vector<double> cam_extrinR(9, 0.0);
deque<double> time_buffer;
deque<PointCloudXYZI::Ptr> lidar_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
deque<cv::Mat> img_buffer;
deque<double> img_time_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr map_cur_frame_point(new PointCloudXYZI());
PointCloudXYZI::Ptr sub_map_cur_frame_point(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_static_world(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_dynamic_world(new PointCloudXYZI());
PointCloudXYZI::Ptr key_frames(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));

image_transport::Publisher img_pub;
image_transport::Publisher dyna_img_pub;
ros::Publisher pubLaserCloudFull;
ros::Publisher pubLaserCloudFull_body;
ros::Publisher pubLaserCloudEffect;
ros::Publisher pubLaserCloudMap;
ros::Publisher pubLaserCloudDynamic;
ros::Publisher pubLaserCloudStatic;
ros::Publisher pubLaserDynamicObject;
ros::Publisher pubLaserDynamicCenter;
ros::Publisher pubOdomAftMapped;
ros::Publisher pubKeyFrames;
ros::Publisher pubPath;

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

KD_TREE<PointType> ikdtree;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);
V3D Lidar_T_wrt_Cam(Zero3d);
M3D Lidar_R_wrt_Cam(Eye3d);

MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
vect3 pos_lid;

nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

void SigHandle(int sig) {
  flg_exit = true;
  ROS_WARN("catch sig %d", sig);
  sig_buffer.notify_all();
}

void pointBodyToWorld_ikfom(PointType const* const pi, PointType* const po, state_ikfom& s) {
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);

  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

void pointBodyToWorld(PointType const* const pi, PointType* const po) {
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

template <typename T>
void pointBodyToWorld(const Matrix<T, 3, 1>& pi, Matrix<T, 3, 1>& po) {
  V3D p_body(pi[0], pi[1], pi[2]);
  V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

  po[0] = p_global(0);
  po[1] = p_global(1);
  po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const* const pi, PointType* const po) {
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const* const pi, PointType* const po) {
  V3D p_body_lidar(pi->x, pi->y, pi->z);
  V3D p_body_imu(state_point.offset_R_L_I * p_body_lidar + state_point.offset_T_L_I);

  po->x = p_body_imu(0);
  po->y = p_body_imu(1);
  po->z = p_body_imu(2);
  po->intensity = pi->intensity;
}

void points_cache_collect() {
  PointVector points_history;
  ikdtree.acquire_removed_points(points_history);
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment() {
  cub_needrm.clear();
  kdtree_delete_counter = 0;
  kdtree_delete_time = 0.0;
  pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
  V3D pos_LiD = pos_lid;
  if (!Localmap_Initialized) {
    for (int i = 0; i < 3; i++) {
      LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
      LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
    }
    Localmap_Initialized = true;
    return;
  }
  float dist_to_map_edge[3][2];
  bool need_move = false;
  for (int i = 0; i < 3; i++) {
    dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
    dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
    if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
  }
  if (!need_move) return;
  BoxPointType New_LocalMap_Points, tmp_boxpoints;
  New_LocalMap_Points = LocalMap_Points;
  float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD - 1)));
  for (int i = 0; i < 3; i++) {
    tmp_boxpoints = LocalMap_Points;
    if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE) {
      New_LocalMap_Points.vertex_max[i] -= mov_dist;
      New_LocalMap_Points.vertex_min[i] -= mov_dist;
      tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
      cub_needrm.push_back(tmp_boxpoints);
    } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) {
      New_LocalMap_Points.vertex_max[i] += mov_dist;
      New_LocalMap_Points.vertex_min[i] += mov_dist;
      tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
      cub_needrm.push_back(tmp_boxpoints);
    }
  }
  LocalMap_Points = New_LocalMap_Points;

  points_cache_collect();
  double delete_begin = omp_get_wtime();
  if (cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
  kdtree_delete_time = omp_get_wtime() - delete_begin;
}

void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr& msg) {
  mtx_buffer.lock();
  scan_count++;
  double preprocess_start_time = omp_get_wtime();
  if (msg->header.stamp.toSec() < last_timestamp_lidar) {
    ROS_ERROR("lidar loop back, clear buffer");
    lidar_buffer.clear();
  }

  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);
  lidar_buffer.push_back(ptr);
  if (p_pre->lidar_type == 3) {
    time_buffer.push_back(msg->header.stamp.toSec() - ptr->back().curvature / float(1000));
  } else {
    time_buffer.push_back(msg->header.stamp.toSec());
  }
  last_timestamp_lidar = msg->header.stamp.toSec();
  if (first_lidar_time < 0.) first_lidar_time = time_buffer.back();
  s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool timediff_set_flg = false;
void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr& msg) {
  mtx_buffer.lock();
  double preprocess_start_time = omp_get_wtime();
  scan_count++;
  if (msg->header.stamp.toSec() < last_timestamp_lidar) {
    ROS_ERROR("lidar loop back, clear buffer");
    lidar_buffer.clear();
  }
  last_timestamp_lidar = msg->header.stamp.toSec();

  if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty()) {
    printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n", last_timestamp_imu, last_timestamp_lidar);
  }

  if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty()) {
    timediff_set_flg = true;
    timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
    printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
  }

  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);
  lidar_buffer.push_back(ptr);
  time_buffer.push_back(last_timestamp_lidar);
  if (first_lidar_time < 0.) first_lidar_time = time_buffer.back();

  s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::Imu::ConstPtr& msg_in) {
  publish_count++;
  // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
  sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

  msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() - time_diff_lidar_to_imu);
  if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en) {
    msg->header.stamp =
        ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
  }

  double timestamp = msg->header.stamp.toSec();

  mtx_buffer.lock();

  if (timestamp < last_timestamp_imu) {
    ROS_WARN("imu loop back, clear buffer");
    imu_buffer.clear();
  }

  imu_buffer.push_back(msg);
  mtx_buffer.unlock();
  sig_buffer.notify_all();
  last_timestamp_imu = timestamp;
}

cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr& img_msg) {
  cv::Mat img;
  img = cv_bridge::toCvCopy(img_msg, "bgr8")->image;
  return img;
}

void img_cbk(const sensor_msgs::ImageConstPtr& msg) {
  if (!img_en) {
    return;
  }
  if (msg->header.stamp.toSec() < last_timestamp_img) {
    ROS_ERROR("img loop back, clear buffer");
    img_buffer.clear();
    img_time_buffer.clear();
  }
  mtx_buffer.lock();

  img_buffer.push_back(getImageFromMsg(msg));
  img_time_buffer.push_back(msg->header.stamp.toSec());
  last_timestamp_img = msg->header.stamp.toSec();

  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

bool sync_packages(MeasureGroup& meas) {
  meas.clear();

  assert(img_buffer.size() == img_time_buffer.size());

  if ((lidar_en && lidar_buffer.empty() || img_en && img_buffer.empty()) || imu_buffer.empty()) {  // has lidar topic or img topic?
    return false;
  }

  if (img_en && (img_buffer.empty() || img_time_buffer.back() < time_buffer.front())) {
    return false;
  }

  if (img_en && flg_first_img && imu_buffer.empty()) {
    return false;
  }
  flg_first_img = false;

  // std::cout << "1" << std::endl;

  if (!lidar_pushed) {  // If not in lidar scan, need to generate new meas
    if (lidar_buffer.empty()) {
      return false;
    }
    meas.lidar = lidar_buffer.front();  // push the first lidar topic
    if (meas.lidar->points.size() <= 1) {
      mtx_buffer.lock();
      if (img_buffer.size() > 0)  // temp method, ignore img topic when no lidar points, keep sync
      {
        lidar_buffer.pop_front();
        time_buffer.pop_front();
        img_buffer.pop_front();
        img_time_buffer.pop_front();
      }
      mtx_buffer.unlock();
      sig_buffer.notify_all();
      ROS_ERROR("empty pointcloud");
      return false;
    }
    sort(meas.lidar->points.begin(), meas.lidar->points.end(), time_list);                      // sort by sample timestamp
    meas.lidar_beg_time = time_buffer.front();                                                  // generate lidar_beg_time
    lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);  // calc lidar scan end time
    meas.lidar_end_time = lidar_end_time;
    lidar_pushed = true;  // flag
  }

  while (!flg_EKF_inited && !img_time_buffer.empty() &&
         img_time_buffer.front() < lidar_end_time) {
    mtx_buffer.lock();
    img_buffer.pop_front();
    img_time_buffer.pop_front();
    mtx_buffer.unlock();
  }

  if (last_timestamp_imu <= lidar_end_time) {
    return false;
  }

  if (!img_time_buffer.empty() && last_timestamp_imu <= img_time_buffer.front())
    return false;

  // std::cout << "2" << std::endl;

  if (img_buffer.empty()) {                     // no img topic, means only has lidar topic
    if (last_timestamp_imu < lidar_end_time) {  // imu message needs to be larger than lidar_end_time, keep complete propagate.
      ROS_ERROR("lidar out sync");
      lidar_pushed = false;
      return false;
    }
    struct ImgIMUsGroup m;  // standard method to keep imu message.
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    m.imus_only = true;
    m.imu.clear();
    mtx_buffer.lock();
    while ((!imu_buffer.empty() && (imu_time < lidar_end_time))) {
      imu_time = imu_buffer.front()->header.stamp.toSec();
      if (imu_time > lidar_end_time) break;
      m.imu.push_back(imu_buffer.front());
      imu_buffer.pop_front();
    }
    lidar_buffer.pop_front();
    time_buffer.pop_front();
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    lidar_pushed = false;  // sync one whole lidar scan.
    meas.img_imus.push_back(m);
    return true;
  }
  // std::cout << "3" << std::endl;

  while (imu_buffer.front()->header.stamp.toSec() <= lidar_end_time) {
    struct ImgIMUsGroup m;
    if (img_buffer.empty() || img_time_buffer.front() > lidar_end_time) {  // has img topic, but img topic timestamp larger than lidar end time, process lidar topic.
      if (last_timestamp_imu < lidar_end_time) {
        ROS_ERROR("lidar out sync");
        lidar_pushed = false;
        return false;
      }
      double imu_time = imu_buffer.front()->header.stamp.toSec();
      m.imu.clear();
      m.imus_only = true;
      mtx_buffer.lock();
      while (!imu_buffer.empty() && (imu_time <= lidar_end_time)) {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if (imu_time > lidar_end_time) break;
        m.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
      }
      mtx_buffer.unlock();
      sig_buffer.notify_all();
      meas.img_imus.push_back(m);
    } else {
      double img_start_time = img_time_buffer.front();  // process img topic, record timestamp
      if (last_timestamp_imu < img_start_time) {
        ROS_ERROR("img out sync");
        lidar_pushed = false;
        return false;
      }

      if (img_start_time < meas.last_update_time) {
        mtx_buffer.lock();
        img_buffer.pop_front();
        img_time_buffer.pop_front();
        mtx_buffer.unlock();
        sig_buffer.notify_all();
        continue;
      }

      double imu_time = imu_buffer.front()->header.stamp.toSec();
      m.imu.clear();
      m.img_offset_time = img_start_time - meas.lidar_beg_time;  // record img offset time, it shoule be the Kalman update timestamp.
      m.img = img_buffer.front();
      m.imus_only = false;
      mtx_buffer.lock();
      while ((!imu_buffer.empty() && (imu_time < img_start_time))) {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if (imu_time > img_start_time || imu_time > lidar_end_time) break;
        m.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
      }
      img_buffer.pop_front();
      img_time_buffer.pop_front();
      mtx_buffer.unlock();
      sig_buffer.notify_all();
      meas.img_imus.push_back(m);
    }

    if (imu_buffer.empty()) {
      ROS_ERROR("imu buffer empty");
    }
  }
  // std::cout << "4" << std::endl;
  lidar_pushed = false;  // sync one whole lidar scan.

  if (!meas.img_imus.empty()) {
    mtx_buffer.lock();
    lidar_buffer.pop_front();
    time_buffer.pop_front();
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    return true;
  }
  // std::cout << "5" << std::endl;

  return false;
}

int process_increments = 0;
void map_incremental() {
  PointVector PointToAdd;
  PointVector PointNoNeedDownsample;
  PointToAdd.reserve(feats_down_size);
  PointNoNeedDownsample.reserve(feats_down_size);
  for (int i = 0; i < feats_down_size; i++) {
    /* transform to world frame */
    pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
    /* decide if need add to map */
    if (!Nearest_Points[i].empty() && flg_EKF_inited) {
      const PointVector& points_near = Nearest_Points[i];
      bool need_add = true;
      BoxPointType Box_of_Point;
      PointType downsample_result, mid_point;
      mid_point.x = floor(feats_down_world->points[i].x / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
      mid_point.y = floor(feats_down_world->points[i].y / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
      mid_point.z = floor(feats_down_world->points[i].z / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
      float dist = calc_dist(feats_down_world->points[i], mid_point);
      if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min) {
        PointNoNeedDownsample.push_back(feats_down_world->points[i]);
        continue;
      }
      for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++) {
        if (points_near.size() < NUM_MATCH_POINTS) break;
        if (calc_dist(points_near[readd_i], mid_point) < dist) {
          need_add = false;
          break;
        }
      }
      if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
    } else {
      PointToAdd.push_back(feats_down_world->points[i]);
    }
  }

  double st_time = omp_get_wtime();
  add_point_size = ikdtree.Add_Points(PointToAdd, true);
  ikdtree.Add_Points(PointNoNeedDownsample, false);
  add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
  kdtree_incremental_time = omp_get_wtime() - st_time;
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
void publish_frame_world(const ros::Publisher& pubLaserCloudFull) {
  if (scan_pub_en) {
    PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
    int size = laserCloudFullRes->points.size();
    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++) {
      RGBpointBodyToWorld(&laserCloudFullRes->points[i],
                          &laserCloudWorld->points[i]);
    }

    *pcl_wait_pub = *laserCloudWorld;

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "camera_init";
    pubLaserCloudFull.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
  }

  /**************** save map ****************/
  /* 1. make sure you have enough memories
  /* 2. noted that pcd save will influence the real-time performences **/
  if (pcd_save_en) {
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudWorld(
        new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++) {
      RGBpointBodyToWorld(&feats_undistort->points[i],
                          &laserCloudWorld->points[i]);
    }
    *pcl_wait_save += *laserCloudWorld;

    // static int scan_wait_num = 0;
    // scan_wait_num++;
    // if (pcl_wait_save->size() > 0 && pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval) {
    //   pcd_index++;
    //   string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
    //   pcl::PCDWriter pcd_writer;
    //   cout << "current scan saved to /PCD/" << all_points_dir << endl;
    //   pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    //   pcl_wait_save->clear();
    //   scan_wait_num = 0;
    // }
  }
}

void publish_frame_body(const ros::Publisher& pubLaserCloudFull_body) {
  int size = feats_undistort->points.size();
  PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

  for (int i = 0; i < size; i++) {
    RGBpointBodyLidarToIMU(&feats_undistort->points[i],
                           &laserCloudIMUBody->points[i]);
  }

  sensor_msgs::PointCloud2 laserCloudmsg;
  pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
  laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
  laserCloudmsg.header.frame_id = "body";
  pubLaserCloudFull_body.publish(laserCloudmsg);
  publish_count -= PUBFRAME_PERIOD;
}

void publish_effect_world(const ros::Publisher& pubLaserCloudEffect) {
  PointCloudXYZI::Ptr laserCloudWorld(
      new PointCloudXYZI(effct_feat_num, 1));
  for (int i = 0; i < effct_feat_num; i++) {
    RGBpointBodyToWorld(&laserCloudOri->points[i],
                        &laserCloudWorld->points[i]);
  }
  sensor_msgs::PointCloud2 laserCloudFullRes3;
  pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
  laserCloudFullRes3.header.stamp = ros::Time().fromSec(lidar_end_time);
  laserCloudFullRes3.header.frame_id = "camera_init";
  pubLaserCloudEffect.publish(laserCloudFullRes3);
}

void publish_map(const ros::Publisher& pubLaserCloudMap) {
  sensor_msgs::PointCloud2 laserCloudMap;
  pcl::toROSMsg(*featsFromMap, laserCloudMap);
  laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
  laserCloudMap.header.frame_id = "camera_init";
  pubLaserCloudMap.publish(laserCloudMap);
}
template <typename T>
void set_posestamp(T& out) {
  out.pose.position.x = state_point.pos(0);
  out.pose.position.y = state_point.pos(1);
  out.pose.position.z = state_point.pos(2);
  out.pose.orientation.x = geoQuat.x;
  out.pose.orientation.y = geoQuat.y;
  out.pose.orientation.z = geoQuat.z;
  out.pose.orientation.w = geoQuat.w;
}

void publish_odometry(const ros::Publisher& pubOdomAftMapped) {
  odomAftMapped.header.frame_id = "camera_init";
  odomAftMapped.child_frame_id = "body";
  odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);  // ros::Time().fromSec(lidar_end_time);
  set_posestamp(odomAftMapped.pose);
  pubOdomAftMapped.publish(odomAftMapped);
  auto P = kf.get_P();
  for (int i = 0; i < 6; i++) {
    int k = i < 3 ? i + 3 : i - 3;
    odomAftMapped.pose.covariance[i * 6 + 0] = P(k, 3);
    odomAftMapped.pose.covariance[i * 6 + 1] = P(k, 4);
    odomAftMapped.pose.covariance[i * 6 + 2] = P(k, 5);
    odomAftMapped.pose.covariance[i * 6 + 3] = P(k, 0);
    odomAftMapped.pose.covariance[i * 6 + 4] = P(k, 1);
    odomAftMapped.pose.covariance[i * 6 + 5] = P(k, 2);
  }

  static tf::TransformBroadcaster br;
  tf::Transform transform;
  tf::Quaternion q;
  transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x,
                                  odomAftMapped.pose.pose.position.y,
                                  odomAftMapped.pose.pose.position.z));
  q.setW(odomAftMapped.pose.pose.orientation.w);
  q.setX(odomAftMapped.pose.pose.orientation.x);
  q.setY(odomAftMapped.pose.pose.orientation.y);
  q.setZ(odomAftMapped.pose.pose.orientation.z);
  transform.setRotation(q);
  br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "camera_init", "body"));

    std::ofstream fout(std::string(ROOT_DIR) + "Log/traj.txt", std::ios::app);
    double timestamp = lidar_end_time - first_lidar_time;
    fout << std::fixed << std::setprecision(15) << timestamp << " "
        << std::setprecision(15)
        << odomAftMapped.pose.pose.position.x << " "
        << odomAftMapped.pose.pose.position.y << " "
        << odomAftMapped.pose.pose.position.z << " "
        << odomAftMapped.pose.pose.orientation.w << " "
        << odomAftMapped.pose.pose.orientation.x << " "
        << odomAftMapped.pose.pose.orientation.y << " "
        << odomAftMapped.pose.pose.orientation.z << std::endl;

}

void publish_path(const ros::Publisher pubPath) {
  set_posestamp(msg_body_pose);
  msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
  msg_body_pose.header.frame_id = "camera_init";

  /*** if path is too large, the rvis will crash ***/
  static int jjj = 0;
  jjj++;
  if (jjj % 10 == 0) {
    path.poses.push_back(msg_body_pose);
    pubPath.publish(path);
  }
}

double pos_thre = 2;
double rot_thre = 10. / 180. * M_PI;
int32_t kf_id = 0;

PointCloudXYZI::Ptr dyna_obj_cloud(new PointCloudXYZI());
PointCloudXYZI::Ptr curr_dyna_center(new PointCloudXYZI());
PointCloudXYZI::Ptr curr_center(new PointCloudXYZI());
PointCloudXYZI::Ptr last_center(new PointCloudXYZI());

bool remove_dynamic(const cv::Mat& image) {
  if (image.empty()) return true;

  static Eigen::Vector3d last_pos = Eigen::Vector3d::Zero();
  static Eigen::Quaterniond last_rot = Eigen::Quaterniond::Identity();
  const Eigen::Vector3d curr_pos = kf.get_x().pos;
  const Eigen::Quaterniond curr_rot = kf.get_x().rot;

  if ((last_pos - curr_pos).norm() < pos_thre &&
      last_rot.angularDistance(curr_rot) < rot_thre) {
    return false;
  }
  auto remove_start = std::chrono::high_resolution_clock::now();
  last_pos = curr_pos;
  last_rot = curr_rot;
  PointType frame;
  frame.getVector3fMap() = curr_pos.cast<float>();
  frame.getNormalVector4fMap() = curr_rot.coeffs().cast<float>();
  frame.intensity = kf_id++;
  key_frames->points.emplace_back(frame);

  const std::vector<SegOutput>& seg_results = yolo_segmenter->SegmentDynamic(image);

  cv::Mat dynamic_mask = yolo_segmenter->DrawSegement(image, seg_results);
  std::vector<PointCloudXYZI::Ptr> curr_dynamic;
  curr_dynamic.assign(seg_results.size(), nullptr);
  cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
  for (int32_t i = 0; i < seg_results.size(); ++i) {
    const auto& seg = seg_results.at(i);
    mask(seg.box).setTo(cv::Scalar(i + 1), seg.boxMask);
  }

  feats_static_world->clear();
  feats_dynamic_world->clear();
  feats_static_world->reserve(pcl_wait_pub->size());
  feats_dynamic_world->reserve(pcl_wait_pub->size());
  const Sophus::SE3& tfcw = vpnp->tfwc.inverse();
  const Eigen::Matrix3d& rcw = tfcw.rotation_matrix();
  const Eigen::Vector3d& tcw = tfcw.translation();
  for (const auto& pt : pcl_wait_pub->points) {
    Eigen::Vector3d pt_w{pt.x, pt.y, pt.z};
    const Eigen::Vector3d& pt_c = rcw * pt_w + tcw;
    if (pt_c.z() <= 1.e-6) continue;
    const Eigen::Vector2d& uv = vpnp->cam->world2cam(pt_c);
    if (!vpnp->cam->isInFrame(uv.cast<int>(), 1)) continue;
    int32_t object_id = mask.at<uchar>(uv.y(), uv.x());
    if (object_id == 0) {
      feats_static_world->emplace_back(pt);
      //cv::circle(dynamic_mask, cv::Point(uv.x(), uv.y()), 2, cv::Scalar(0, 255, 0), -1);
    } else {
      object_id -= 1;
      feats_dynamic_world->emplace_back(pt);
      //cv::circle(dynamic_mask, cv::Point(uv.x(), uv.y()), 2, cv::Scalar(0, 0, 255), -1);
      if (curr_dynamic.at(object_id) == nullptr)
        curr_dynamic.at(object_id).reset(new PointCloudXYZI);
      curr_dynamic.at(object_id)->emplace_back(pt);
    }
  }

  dyna_obj_cloud->clear();
  curr_dyna_center->clear();
  curr_center->clear();
  dyna_obj_cloud->reserve(feats_dynamic_world->size());
  curr_dyna_center->reserve(curr_dynamic.size());
  curr_center->reserve(curr_dynamic.size());
  pcl::EuclideanClusterExtraction<PointType> ec;
  for (int32_t i = 0; i < curr_dynamic.size(); ++i) {
    if (curr_dynamic.at(i) == nullptr) continue;
    pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>);
    tree->setInputCloud(curr_dynamic.at(i));
    std::vector<pcl::PointIndices> cluster_indices;
    ec.setClusterTolerance(1.0);
    ec.setMinClusterSize(curr_dynamic.at(i)->size() / 2);
    ec.setMaxClusterSize(curr_dynamic.at(i)->size());
    ec.setSearchMethod(tree);
    ec.setInputCloud(curr_dynamic.at(i));
    ec.extract(cluster_indices);
    if (cluster_indices.empty()) continue;
    pcl::PointIndices largest_cluster = cluster_indices[0];
    for (const auto& indices : cluster_indices) {
      if (indices.indices.size() > largest_cluster.indices.size()) {
        largest_cluster = indices;
      }
    }
    Eigen::Vector3f center(0.0, 0.0, 0.0);
    for (const auto& idx : largest_cluster.indices) {
      center += curr_dynamic.at(i)->points[idx].getVector3fMap();
    }
    center /= largest_cluster.indices.size();

    PointType obj_center;
    obj_center.getVector3fMap() = center;
    obj_center.intensity = i;
    curr_center->emplace_back(obj_center);

    bool dyna_status = true;
    for (const auto c : last_center->points) {
      if ((center - c.getVector3fMap()).norm() < 0.05) {
        dyna_status = false;
        break;
      }
    }

    if (dyna_status) {
      curr_dyna_center->emplace_back(obj_center);
      *dyna_obj_cloud += *curr_dynamic.at(i);
    }
  }
  *last_center = *curr_center;

  cv_bridge::CvImage out_msg;
  out_msg.header.stamp = ros::Time::now();
  out_msg.encoding = sensor_msgs::image_encodings::BGR8;
  out_msg.image = dynamic_mask;
  dyna_img_pub.publish(out_msg.toImageMsg());

  if (pubLaserCloudDynamic.getNumSubscribers() > 0) {
    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*feats_dynamic_world, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().now();
    laserCloudmsg.header.frame_id = "camera_init";
    pubLaserCloudDynamic.publish(laserCloudmsg);
  }
  if (pubLaserCloudStatic.getNumSubscribers() > 0) {
    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*feats_static_world, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().now();
    laserCloudmsg.header.frame_id = "camera_init";
    pubLaserCloudStatic.publish(laserCloudmsg);
  }
  if (pubKeyFrames.getNumSubscribers() > 0) {
    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*key_frames, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().now();
    laserCloudmsg.header.frame_id = "camera_init";
    pubKeyFrames.publish(laserCloudmsg);
  }
  if (pubLaserDynamicObject.getNumSubscribers() > 0) {
    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*dyna_obj_cloud, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().now();
    laserCloudmsg.header.frame_id = "camera_init";
    pubLaserDynamicObject.publish(laserCloudmsg);
  }
  if (pubLaserDynamicCenter.getNumSubscribers() > 0) {
    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*curr_dyna_center, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().now();
    laserCloudmsg.header.frame_id = "camera_init";
    pubLaserDynamicCenter.publish(laserCloudmsg);
  }
  auto remove_end = std::chrono::high_resolution_clock::now();
  double remove_frame_time = std::chrono::duration_cast<std::chrono::microseconds>(remove_end - remove_start).count();
  ROS_INFO("dynamic point filter processing time: %.2f us (%.3f ms)", remove_frame_time, remove_frame_time / 1000.0);
  remove_frame_count++;
  remove_process_time += remove_frame_time/1000;
  return true;
}

void h_share_model(state_ikfom& s, esekfom::dyn_share_datastruct<double>& ekfom_data) {
  laserCloudOri->clear();
  corr_normvect->clear();

/** closest surface search and residual computation **/
#ifdef MP_EN
  omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
  for (int i = 0; i < feats_down_size; i++) {
    PointType& point_body = feats_down_body->points[i];
    PointType& point_world = feats_down_world->points[i];

    /* transform to world frame */
    V3D p_body(point_body.x, point_body.y, point_body.z);
    V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);
    point_world.x = p_global(0);
    point_world.y = p_global(1);
    point_world.z = p_global(2);
    point_world.intensity = point_body.intensity;
    vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

    auto& points_near = Nearest_Points[i];

    if (ekfom_data.converge) {
      /** Find the closest surfaces in the map **/
      ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
      point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false
                                                                                                                          : true;
    }

    if (!point_selected_surf[i]) continue;

    VF(4)
    pabcd;
    point_selected_surf[i] = false;
    if (esti_plane(pabcd, points_near, 0.1f)) {
      float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
      float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

      if (s > 0.9) {
        point_selected_surf[i] = true;
        normvec->points[i].x = pabcd(0);
        normvec->points[i].y = pabcd(1);
        normvec->points[i].z = pabcd(2);
        normvec->points[i].intensity = pd2;
      }
    }
  }

  effct_feat_num = 0;

  for (int i = 0; i < feats_down_size; i++) {
    if (point_selected_surf[i]) {
      laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
      corr_normvect->points[effct_feat_num] = normvec->points[i];
      effct_feat_num++;
    }
  }

  if (effct_feat_num < 1) {
    ekfom_data.valid = false;
    ROS_WARN("No Effective Points! \n");
    return;
  }

  /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
  ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12);  // 23
  ekfom_data.h.resize(effct_feat_num);

  for (int i = 0; i < effct_feat_num; i++) {
    const PointType& laser_p = laserCloudOri->points[i];
    V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
    M3D point_be_crossmat;
    point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
    V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
    M3D point_crossmat;
    point_crossmat << SKEW_SYM_MATRX(point_this);

    /*** get the normal vector of closest surface/corner ***/
    const PointType& norm_p = corr_normvect->points[i];
    V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

    /*** calculate the Measuremnt Jacobian matrix H ***/
    V3D C(s.rot.conjugate() * norm_vec);
    V3D A(point_crossmat * C);
    if (extrinsic_est_en) {
      V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C);  // s.rot.conjugate()*norm_vec);
      ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
    } else {
      ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    }

    /*** Measuremnt: distance to the closest surface/corner ***/
    ekfom_data.h(i) = -norm_p.intensity;
  }
}

void readParameters(ros::NodeHandle& nh) {
  nh.param<bool>("publish/path_en", path_en, true);
  nh.param<bool>("publish/scan_publish_en", scan_pub_en, true);
  nh.param<bool>("publish/dense_publish_en", dense_pub_en, true);
  nh.param<bool>("publish/scan_bodyframe_pub_en", scan_body_pub_en, true);
  nh.param<int>("max_iteration", NUM_MAX_ITERATIONS, 4);
  nh.param<string>("map_file_path", map_file_path, "");
  nh.param<int>("img_enable", img_en, 1);
  nh.param<int>("lidar_enable", lidar_en, 1);
  nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");
  nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");
  nh.param<string>("common/img_topic", img_topic, "/usb_cam/image_raw");
  nh.param<double>("filter_size_surf", filter_size_surf_min, 0.5);
  nh.param<double>("filter_size_map", filter_size_map_min, 0.5);
  nh.param<float>("mapping/det_range", DET_RANGE, 300.f);
  nh.param<double>("mapping/fov_degree", fov_deg, 180);
  nh.param<double>("mapping/gyr_cov", gyr_cov, 0.1);
  nh.param<double>("mapping/acc_cov", acc_cov, 0.1);
  nh.param<double>("mapping/b_gyr_cov", b_gyr_cov, 0.0001);
  nh.param<double>("mapping/b_acc_cov", b_acc_cov, 0.0001);
  nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
  nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
  nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
  nh.param<int>("preprocess/timestamp_unit", p_pre->time_unit, US);
  nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
  nh.param<int>("point_filter_num", p_pre->point_filter_num, 2);
  nh.param<bool>("feature_extract_enable", p_pre->feature_enabled, false);
  nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
  nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
  nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());
  nh.param<vector<double>>("mapping/Pcl", cam_extrinT, vector<double>());
  nh.param<vector<double>>("mapping/Rcl", cam_extrinR, vector<double>());
  nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
  nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
  nh.param<double>("mapping/laser_point_cov", LASER_POINT_COV, 0.001);

  nh.param<int>("camera/debug", debug, 0);
  nh.param<double>("camera/cam_fx", cam_fx, 453.483063);
  nh.param<double>("camera/cam_fy", cam_fy, 453.254913);
  nh.param<double>("camera/cam_cx", cam_cx, 318.908851);
  nh.param<double>("camera/cam_cy", cam_cy, 234.238189);
  nh.param<int>("camera/grid_size", grid_size, 40);
  nh.param<int>("camera/patch_size", patch_size, 4);

  Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
  Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);
  Lidar_T_wrt_Cam << VEC_FROM_ARRAY(cam_extrinT);
  Lidar_R_wrt_Cam << MAT_FROM_ARRAY(cam_extrinR);

  vicp.reset(new visual_icp::VisualICP(nh));
  vpnp.reset(new visual_pnp::VisualPNP(nh));
  yolo_segmenter.reset(new YoloSeg(std::string(ROOT_DIR) + "model", "yolov5s-seg"));
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "laserMapping");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  readParameters(nh);
  cout << "p_pre->lidar_type " << p_pre->lidar_type << endl;
  std::ofstream fout(std::string(ROOT_DIR) + "Log/traj.txt");
  fout << "#timestamp x y z q_w q_x q_y q_z" << std::endl;

  /*** ROS subscribe initialization ***/
  ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
  ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
  ros::Subscriber sub_img = nh.subscribe(img_topic, 200000, img_cbk);
  img_pub = it.advertise("/rgb_img", 1);
  dyna_img_pub = it.advertise("/dynamic_img", 1);
  pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
  pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body", 100000);
  pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100000);
  pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100000);
  pubLaserCloudDynamic = nh.advertise<sensor_msgs::PointCloud2>("/cloud_potential_dynamic", 100000);
  pubLaserCloudStatic = nh.advertise<sensor_msgs::PointCloud2>("/cloud_static", 100000);
  pubLaserDynamicObject = nh.advertise<sensor_msgs::PointCloud2>("/dynamic_object", 100000);
  pubLaserDynamicCenter = nh.advertise<sensor_msgs::PointCloud2>("/dynamic_center", 100000);
  pubKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("/key_frames", 100000);
  pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
  pubPath = nh.advertise<nav_msgs::Path>("/path", 100000);
  //------------------------------------------------------------------------------------------------------

  path.header.stamp = ros::Time::now();
  path.header.frame_id = "camera_init";

  /*** variables definition ***/
  int effect_feat_num = 0, frame_num = 0;
  bool flg_EKF_converged, EKF_stop_flg = 0;

  FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
  HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

  memset(point_selected_surf, true, sizeof(point_selected_surf));
  downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
  downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
  memset(point_selected_surf, true, sizeof(point_selected_surf));

  p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
  p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
  p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
  p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
  p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

  double epsi[23] = {0.001};
  fill(epsi, epsi + 23, 0.001);
  auto init_start_time = std::chrono::high_resolution_clock::now();
  kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);
  auto init_end_time = std::chrono::high_resolution_clock::now();
  double init_frame_time = std::chrono::duration_cast<std::chrono::microseconds>(init_end_time - init_start_time).count();
  signal(SIGINT, SigHandle);
  ros::Rate rate(5000);
  bool status = ros::ok();

  while (status) {
    if (flg_exit) break;
    ros::spinOnce();
    if (!sync_packages(Measures)) {
      status = ros::ok();
      cv::waitKey(1);
      rate.sleep();
      continue;
    }

    if (flg_first_scan) {
      p_imu->first_lidar_time = Measures.lidar_beg_time;
      flg_first_scan = false;
      continue;
    }

    state_point = kf.get_x();
    pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

    std::string process_step;

    assert(Measures.img_imus.size() > 0);

    cv::Mat last_rgb;
    for (const auto& img_imus : Measures.img_imus) {
      auto imu_forward_start = std::chrono::high_resolution_clock::now();
      p_imu->Foward(Measures, img_imus, kf);
      auto imu_forward_end = std::chrono::high_resolution_clock::now();
      double imu_forward_frame_time = std::chrono::duration_cast<std::chrono::microseconds>(imu_forward_end - imu_forward_start).count();
      ROS_INFO("Image processing time: %.2f us (%.3f ms)", imu_forward_frame_time, imu_forward_frame_time / 1000.0);
      imu_frame_count++;
      imu_process_time += imu_forward_frame_time/1000;
      double front_imu_time = 0.;
      double back_imu_time = 0.;
      if (!img_imus.imu.empty()) {
        front_imu_time = img_imus.imu.front()->header.stamp.toSec();
        back_imu_time = img_imus.imu.back()->header.stamp.toSec();
      }
      if (!img_imus.imus_only && first_lidar_time > 0.) {
        assert(!img_imus.img.empty());
        ROS_INFO("Image Process: image at %f, imus from %f to %f",
                 Measures.lidar_beg_time + img_imus.img_offset_time,
                 front_imu_time, back_imu_time);
        process_step += "I";
        auto img_start = std::chrono::high_resolution_clock::now();
        vicp->UpdateState(kf, img_imus.img, pcl_wait_pub);
        vpnp->UpdateState(kf, img_imus.img, pcl_wait_pub);
        auto img_end = std::chrono::high_resolution_clock::now();
        double img_frame_time = std::chrono::duration_cast<std::chrono::microseconds>(img_end - img_start).count();
        ROS_INFO("Image processing time: %.2f us (%.3f ms)", img_frame_time, img_frame_time / 1000.0);
        img_frame_count++;
        img_process_time += img_frame_time/1000;
        last_rgb = vpnp->img().clone();
      } else {
        assert(img_imus.img.empty());
        ROS_INFO("LiDAR Process: lidar at %f, imus from %f to %f",
                 Measures.lidar_end_time, front_imu_time, back_imu_time);
        process_step += "L";
      }
    }


    bool is_keyframe = remove_dynamic(last_rgb);

    auto lidar_start = std::chrono::high_resolution_clock::now();
    process_step += "B";
    p_imu->Backward(Measures, kf, feats_undistort);
    lidar_pushed = false;
    ROS_WARN_STREAM("Porcess Step: " + process_step);

    if (feats_undistort == nullptr || feats_undistort->empty()) {
      ROS_WARN("No point, skip this scan!\n");
      continue;
    }

    flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;
    /*** Segment the map in lidar FOV ***/
    // lasermap_fov_segment();

    /*** downsample the feature points in a scan ***/
    PointCloudXYZI::Ptr mid_pc(new PointCloudXYZI());
    downSizeFilterSurf.setInputCloud(feats_undistort);
    downSizeFilterSurf.filter(*feats_down_body);
    feats_down_size = feats_down_body->points.size();
    /*** initialize the map kdtree ***/
    if (ikdtree.Root_Node == nullptr) {
      if (feats_down_size > 5) {
        ikdtree.set_downsample_param(filter_size_map_min);
        feats_down_world->resize(feats_down_size);
        for (int i = 0; i < feats_down_size; i++) {
          pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        }
        ikdtree.Build(feats_down_world->points);
      }
      continue;
    }

    int featsFromMapNum = ikdtree.validnum();
    kdtree_size_st = ikdtree.size();

    /*** ICP and iterated Kalmaerron filter update ***/
    if (feats_down_size < 5) {
      ROS_WARN("No point, skip this scan!\n");
      continue;
    }

    normvec->resize(feats_down_size);
    feats_down_world->resize(feats_down_size);
    Nearest_Points.resize(feats_down_size);
    /*** iterated state estimation ***/
    double solve_H_time;
    kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
    auto lidar_end = std::chrono::high_resolution_clock::now();
    double lidar_frame_time = std::chrono::duration_cast<std::chrono::microseconds>( lidar_end - lidar_start).count();
    lidar_frame_time += init_frame_time;
    ROS_INFO(" Lidar processing time: %.2f us (%.3f ms)", lidar_frame_time, lidar_frame_time / 1000.0);
    lidar_frame_count++;
    lidar_process_time += lidar_frame_time/1000;
    Measures.last_update_time = lidar_end_time;
    state_point = kf.get_x();
    euler_cur = SO3ToEuler(state_point.rot);
    pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
    geoQuat.x = state_point.rot.coeffs()[0];
    geoQuat.y = state_point.rot.coeffs()[1];
    geoQuat.z = state_point.rot.coeffs()[2];
    geoQuat.w = state_point.rot.coeffs()[3];


    publish_odometry(pubOdomAftMapped);

    auto map_start_time = std::chrono::high_resolution_clock::now();
    /*** add the feature points to map kdtree ***/
    if (is_keyframe) map_incremental();
    auto map_end_time = std::chrono::high_resolution_clock::now();
    if (is_keyframe) {
    double map_frame_time = std::chrono::duration_cast<std::chrono::microseconds>( map_end_time - map_start_time).count();
    ROS_INFO(" map_incremental processing time: %.2f us (%.3f ms)", map_frame_time, map_frame_time / 1000.0);
    map_frame_count++;
    map_process_time += map_frame_time/1000;}
    /******* Publish points *******/
    if (path_en) publish_path(pubPath);
    if (scan_pub_en || pcd_save_en) {
      publish_frame_world(pubLaserCloudFull);
      publish_effect_world(pubLaserCloudEffect);
    }
    if (scan_pub_en || scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body);
    // publish_map(pubLaserCloudMap);
    vicp->Publish();
    vpnp->Publish();
  }
  if (img_frame_count > 0) {
    ROS_INFO("Average image processing time: %.2f ms, img_process_time: %.2f ms, img_frame_count: %d", img_process_time / img_frame_count, img_process_time, img_frame_count);
  }
  if (img_frame_count > 0) {
    ROS_INFO("Average lidar processing time: %.2f ms, lidar_process_time: %.2f ms, lidar_frame_count: %d", lidar_process_time / lidar_frame_count, lidar_process_time, lidar_frame_count);
  }
  if (remove_frame_count > 0) {
    ROS_INFO("Average dynamic point filter processing time: %.4f ms, remove_process_time: %.2f ms, remove_frame_count: %d", remove_process_time / remove_frame_count, remove_process_time, remove_frame_count);
  }
  if (map_frame_count > 0) {
    ROS_INFO("Average map incremental processing time: %.4f ms, map_process_time: %.2f ms, map_frame_count: %d", map_process_time / map_frame_count, map_process_time, map_frame_count);
  }
  if (imu_frame_count > 0) {
    ROS_INFO("Average imu processing time: %.4f ms, imu_process_time: %.2f ms, imu_frame_count: %d", imu_process_time / imu_frame_count, imu_process_time, imu_frame_count);
  }
  /**************** save map ****************/
  /* 1. make sure you have enough memories
  /* 2. pcd save will largely influence the real-time performences **/
  if (pcl_wait_save->size() > 0 && pcd_save_en) {
    string file_name = string("scans.pcd");
    string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
    pcl::PCDWriter pcd_writer;
    cout << "current scan saved to /PCD/" << file_name << endl;
    pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
  }

  return 0;
}
