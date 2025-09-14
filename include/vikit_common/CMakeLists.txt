SET(PROJECT_NAME vikit_common)
PROJECT(${PROJECT_NAME})
CMAKE_MINIMUM_REQUIRED (VERSION 3.0.2)
SET(CMAKE_BUILD_TYPE Release)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules/")

# Add plain cmake packages
FIND_PACKAGE(OpenCV 4  REQUIRED)
FIND_PACKAGE(Eigen REQUIRED)
FIND_PACKAGE(sophus REQUIRED PATHS ../sophus)

# Include dirs
INCLUDE_DIRECTORIES(
  include
  ${Eigen_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${sophus_INCLUDE_DIR}
)

message(sophus_INCLUDE_DIR ${sophus_INCLUDE_DIR})

# Set Sourcefiles
LIST(APPEND SOURCEFILES src/atan_camera.cpp
                        src/omni_camera.cpp
                        src/math_utils.cpp
                        src/vision.cpp
                        src/performance_monitor.cpp
                        src/robust_cost.cpp
                        src/user_input_thread.cpp
                        src/pinhole_camera.cpp
                        src/homography.cpp
                        src/img_align.cpp)

# Create vikit library
ADD_LIBRARY(${PROJECT_NAME} SHARED ${SOURCEFILES})
TARGET_LINK_LIBRARIES(${PROJECT_NAME}
  ${OpenCV_LIBS}
  ${sophus_LIBRARIES}
  ${catkin_LIBRARIES})


################################################################################
# Create the vikit_commonConfig.cmake file for other cmake projects.

SET(vikit_common_LIBRARIES ${PROJECT_NAME} )
SET(vikit_common_LIBRARY_DIR ${PROJECT_BINARY_DIR} )
SET(vikit_common_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/vikit_commonConfig.cmake.in
    ${CMAKE_CURRENT_SOURCE_DIR}/vikit_commonConfig.cmake @ONLY IMMEDIATE )
export( PACKAGE vikit_common )
