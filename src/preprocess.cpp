#include "preprocess.h"

#define RETURN0 0x00
#define RETURN0AND1 0x10

Preprocess::Preprocess()
    : feature_enabled(0), lidar_type(AVIA), blind(0.01), point_filter_num(1) {
  inf_bound = 10;
  N_SCANS = 6;
  SCAN_RATE = 10;
  group_size = 8;
  disA = 0.01;
  disA = 0.1;  // B?
  p2l_ratio = 225;
  limit_maxmid = 6.25;
  limit_midmin = 6.25;
  limit_maxmin = 3.24;
  jump_up_limit = 170.0;
  jump_down_limit = 8.0;
  cos160 = 160.0;
  edgea = 2;
  edgeb = 0.1;
  smallp_intersect = 172.5;
  smallp_ratio = 1.2;
  given_offset_time = false;

  jump_up_limit = cos(jump_up_limit / 180 * M_PI);
  jump_down_limit = cos(jump_down_limit / 180 * M_PI);
  cos160 = cos(cos160 / 180 * M_PI);
  smallp_intersect = cos(smallp_intersect / 180 * M_PI);
}

Preprocess::~Preprocess() {}

void Preprocess::set(bool feat_en, int lid_type, double bld, int pfilt_num) {
  feature_enabled = feat_en;
  lidar_type = lid_type;
  blind = bld;
  point_filter_num = pfilt_num;
}

void Preprocess::process(const livox_ros_driver::CustomMsg::ConstPtr& msg, PointCloudXYZI::Ptr& pcl_out) {
  switch (time_unit) {
    case SEC:
      time_unit_scale = 1.e3f;
      break;
    case MS:
      time_unit_scale = 1.f;
      break;
    case US:
      time_unit_scale = 1.e-3f;
      break;
    case NS:
      time_unit_scale = 1.e-6f;
      break;
    default:
      time_unit_scale = 1.f;
      break;
  }

  avia_handler(msg);
  *pcl_out = pl_surf;
}

void Preprocess::process(const sensor_msgs::PointCloud2::ConstPtr& msg, PointCloudXYZI::Ptr& pcl_out) {
  switch (time_unit) {
    case SEC:
      time_unit_scale = 1.e3f;
      break;
    case MS:
      time_unit_scale = 1.f;
      break;
    case US:
      time_unit_scale = 1.e-3f;
      break;
    case NS:
      time_unit_scale = 1.e-6f;
      break;
    default:
      time_unit_scale = 1.f;
      break;
  }

  switch (lidar_type) {
    case OUST64:
      oust64_handler(msg);
      break;

    case VELO16:
      velodyne_handler(msg);
      break;

    case XYZI:
      xyzi_handler(msg);
      break;

    case XYZIT:
      xyzit_handler(msg);
      break;

    default:
      printf("Error LiDAR Type");
      break;
  }
  *pcl_out = pl_surf;
}

void Preprocess::avia_handler(const livox_ros_driver::CustomMsg::ConstPtr& msg) {
  pl_surf.clear();
  pl_full.clear();
  double t1 = omp_get_wtime();
  int plsize = msg->point_num;

  pl_surf.reserve(plsize);
  pl_full.resize(plsize);

  for (int i = 0; i < N_SCANS; i++) {
    pl_buff[i].clear();
    pl_buff[i].reserve(plsize);
  }
  uint valid_num = 0;

  for (uint i = 1; i < plsize; i++) {
    if ((msg->points[i].line < N_SCANS) && ((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00)) {
      valid_num++;
      if (valid_num % point_filter_num == 0) {
        pl_full[i].x = msg->points[i].x;
        pl_full[i].y = msg->points[i].y;
        pl_full[i].z = msg->points[i].z;
        pl_full[i].intensity = msg->points[i].reflectivity;
        pl_full[i].curvature = msg->points[i].offset_time * time_unit_scale;  // use curvature as time of each laser points, curvature unit: ms

        if (((abs(pl_full[i].x - pl_full[i - 1].x) > 1e-7) ||
             (abs(pl_full[i].y - pl_full[i - 1].y) > 1e-7) ||
             (abs(pl_full[i].z - pl_full[i - 1].z) > 1e-7)) &&
            (pl_full[i].x * pl_full[i].x + pl_full[i].y * pl_full[i].y + pl_full[i].z * pl_full[i].z > (blind * blind))) {
          pl_surf.push_back(pl_full[i]);
        }
      }
    }
  }
}

void Preprocess::oust64_handler(const sensor_msgs::PointCloud2::ConstPtr& msg) {
  pl_surf.clear();
  pl_full.clear();
  pcl::PointCloud<ouster_ros::Point> pl_orig;
  pcl::fromROSMsg(*msg, pl_orig);
  int plsize = pl_orig.size();
  pl_surf.reserve(plsize);
  for (int i = 0; i < pl_orig.points.size(); i++) {
    if (i % point_filter_num != 0) continue;
    double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y + pl_orig.points[i].z * pl_orig.points[i].z;
    if (range < (blind * blind)) continue;

    Eigen::Vector3d pt_vec;
    PointType added_pt;
    added_pt.x = pl_orig.points[i].x;
    added_pt.y = pl_orig.points[i].y;
    added_pt.z = pl_orig.points[i].z;
    added_pt.intensity = pl_orig.points[i].intensity;
    added_pt.normal_x = 0;
    added_pt.normal_y = 0;
    added_pt.normal_z = 0;
    added_pt.curvature = pl_orig.points[i].t * time_unit_scale;  // curvature unit: ms

    pl_surf.points.push_back(added_pt);
  }
}

void Preprocess::velodyne_handler(const sensor_msgs::PointCloud2::ConstPtr& msg) {
  pl_surf.clear();
  pl_full.clear();

  pcl::PointCloud<velodyne_ros::Point> pl_orig;
  pcl::fromROSMsg(*msg, pl_orig);
  int plsize = pl_orig.points.size();
  if (plsize == 0) return;
  pl_surf.reserve(plsize);

  for (int i = 0; i < plsize; i++) {
    PointType added_pt;

    added_pt.normal_x = 0;
    added_pt.normal_y = 0;
    added_pt.normal_z = 0;
    added_pt.x = pl_orig.points[i].x;
    added_pt.y = pl_orig.points[i].y;
    added_pt.z = pl_orig.points[i].z;
    added_pt.intensity = pl_orig.points[i].intensity;
    added_pt.curvature = pl_orig.points[i].time * time_unit_scale;  // curvature unit: ms // cout<<added_pt.curvature<<endl;

    if (i % point_filter_num == 0) {
      if (added_pt.x * added_pt.x + added_pt.y * added_pt.y + added_pt.z * added_pt.z > (blind * blind)) {
        pl_surf.points.push_back(added_pt);
      }
    }
  }
}

void Preprocess::xyzi_handler(const sensor_msgs::PointCloud2::ConstPtr& msg) {
  pl_surf.clear();
  pl_full.clear();

  pcl::PointCloud<xyzi_ros::Point> pl_orig;
  pcl::fromROSMsg(*msg, pl_orig);
  int plsize = pl_orig.points.size();
  if (plsize == 0) return;
  pl_surf.reserve(plsize);

  /*** These variables only works when no point timestamps given ***/
  double omega_l = 0.361 * SCAN_RATE;  // scan angular velocity
  std::vector<bool> is_first(N_SCANS, true);
  std::vector<double> yaw_fp(N_SCANS, 0.0);    // yaw of first scan point
  std::vector<float> yaw_last(N_SCANS, 0.0);   // yaw of last scan point
  std::vector<float> time_last(N_SCANS, 0.0);  // last offset time
                                               /*****************************************************************/

  for (int i = 0; i < plsize; i++) {
    PointType added_pt;

    added_pt.normal_x = 0;
    added_pt.normal_y = 0;
    added_pt.normal_z = 0;
    added_pt.x = pl_orig.points[i].x;
    added_pt.y = pl_orig.points[i].y;
    added_pt.z = pl_orig.points[i].z;
    added_pt.intensity = pl_orig.points[i].intensity;
    added_pt.curvature = 0.;  // curvature unit: ms // cout<<added_pt.curvature<<endl;

    if (i % point_filter_num == 0) {
      if (added_pt.x * added_pt.x + added_pt.y * added_pt.y + added_pt.z * added_pt.z > (blind * blind)) {
        pl_surf.points.push_back(added_pt);
      }
    }
  }
}

void Preprocess::xyzit_handler(const sensor_msgs::PointCloud2::ConstPtr& msg) {
  pl_surf.clear();
  pl_full.clear();

  pcl::PointCloud<xyzit_ros::Point> pl_orig;
  pcl::fromROSMsg(*msg, pl_orig);
  int plsize = pl_orig.points.size();
  if (plsize == 0) return;
  pl_surf.reserve(plsize);

  for (int i = 0; i < plsize; i++) {
    PointType added_pt;

    added_pt.normal_x = 0;
    added_pt.normal_y = 0;
    added_pt.normal_z = 0;
    added_pt.x = pl_orig.points[i].x;
    added_pt.y = pl_orig.points[i].y;
    added_pt.z = pl_orig.points[i].z;
    added_pt.intensity = pl_orig.points[i].intensity;
    added_pt.curvature = pl_orig.points[i].timestamp * time_unit_scale;

    if (i % point_filter_num == 0) {
      if (added_pt.x * added_pt.x + added_pt.y * added_pt.y + added_pt.z * added_pt.z > (blind * blind)) {
        pl_surf.points.push_back(added_pt);
      }
    }
  }
}
