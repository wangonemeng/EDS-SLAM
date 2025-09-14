cmake_minimum_required(VERSION 3.0.2)
project(point_livo)

SET(CMAKE_BUILD_TYPE "Release")
add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions" )
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -pthread -fexceptions")

message("Current CPU archtecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)" )
  include(ProcessorCount)
  ProcessorCount(N)
  message("Processer number:  ${N}")
  if(N GREATER 5)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=4)
    message("core for MP:  4")
  elseif(N GREATER 3)
    math(EXPR PROC_NUM "${N} - 2")
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM="${PROC_NUM}")
    message("core for MP:  ${PROC_NUM}")
  else()
    add_definitions(-DMP_PROC_NUM=1)
  endif()
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()

find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

add_subdirectory(include/sophus)
find_package(sophus REQUIRED PATHS include/sophus)

add_subdirectory(include/vikit_common)
find_package(vikit_common REQUIRED PATHS include/vikit_common)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  std_msgs
  pcl_ros
  tf
  livox_ros_driver
  message_generation
  eigen_conversions
  cv_bridge
  image_transport
)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
FIND_PACKAGE(OpenCV 4  REQUIRED)
FIND_PACKAGE(Boost REQUIRED COMPONENTS thread)

include_directories(
	${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${sophus_INCLUDE_DIR}
  ${vikit_common_INCLUDE_DIR}
  include
  include/vikit_ros/include
)

add_message_files(
  FILES
  Pose6D.msg
)

generate_messages(
 DEPENDENCIES
 geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs message_runtime cv_bridge image_transport
  DEPENDS EIGEN3 PCL OpenCV
  INCLUDE_DIRS include
)

add_library(vikit_ros
  include/vikit_ros/src/output_helper.cpp
)
target_link_libraries(vikit_ros
  PRIVATE
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${sophus_LIBRARIES}
  ${vikit_common_LIBRARIES}
)

add_library(ikdtree
  include/ikd-Tree/ikd_Tree.cpp
  include/FOV_Checker/FOV_Checker.cpp
)
target_link_libraries(ikdtree
  PRIVATE
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)


########################
## Yolo configuration ##
########################
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
option(TENSORRT_FOUND "TensorRT installed on system?" OFF)
# first try to find cuda and tensorrt-related stuff (nvinfer, onnx, etc)
find_package(TensorRT PATHS cmake)
if (TENSORRT_FOUND)
  message("TensorRT SUCCESS!\n")
  set(TENSORRT_FOUND ON)
else()
  message("TensorRT NOT FOUND\n")
  set(TENSORRT_FOUND OFF)
endif (TENSORRT_FOUND)

# complain if no backend is installed
if(NOT TENSORRT_FOUND)
  message(FATAL_ERROR
  "TensorRT must be installed
  TENSORRT_FOUND ${TENSORRT_FOUND}\n")
endif()

include_directories(
  ${TENSORRT_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)
add_library(yoloseg
  include/yolo_segment/yolo_segment.cc
)
target_link_libraries(yoloseg
  PRIVATE
  ${TENSORRT_LIBRARIES}
  ${OpenCV_LIBS}
)

add_executable(fastlivo_mapping
  src/laserMapping.cpp
  src/preprocess.cpp
  src/visual_pnp.cpp
  src/visual_icp.cpp
)

target_link_libraries(fastlivo_mapping
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${sophus_LIBRARIES}
  ${vikit_common_LIBRARIES}
  ikdtree
  vikit_ros
  yoloseg
)
