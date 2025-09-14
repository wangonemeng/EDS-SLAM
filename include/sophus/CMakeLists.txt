SET(PROJECT_NAME sophus)
PROJECT(${PROJECT_NAME})
CMAKE_MINIMUM_REQUIRED(VERSION 3.0.2)
SET(CMAKE_BUILD_TYPE "Release")

# Add local path for finding packages, set the local version first
set( CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules" )
list( APPEND CMAKE_MODULE_PATH "${CMAKE_ROOT}/Modules" )

find_package( Eigen3 REQUIRED )
INCLUDE_DIRECTORIES( ${EIGEN3_INCLUDE_DIR} )
SET( sophus_INCLUDE_DIR ${PROJECT_SOURCE_DIR} ${EIGEN3_INCLUDE_DIR} )

# Set Sourcefiles
LIST(APPEND SOURCEFILES sophus/so2.cpp
                        sophus/se2.cpp
                        sophus/se3.cpp
                        sophus/so3.cpp
                        sophus/scso3.cpp
                        sophus/sim3.cpp)

ADD_LIBRARY(${PROJECT_NAME} SHARED ${SOURCEFILES})

SET(sophus_LIBRARIES ${PROJECT_NAME})
SET(sophus_LIBRARY_DIR ${PROJECT_BINARY_DIR} )
SET(sophus_INCLUDE_DIR ${sophus_INCLUDE_DIR})
CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/sophusConfig.cmake.in
    ${CMAKE_CURRENT_SOURCE_DIR}/sophusConfig.cmake @ONLY IMMEDIATE )
export( PACKAGE sophus )