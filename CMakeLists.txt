cmake_minimum_required(VERSION 2.8.3)
project(line_based_plane_segment)


# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()


find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  message_filters
  roscpp
  sensor_msgs
)
LIST( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED COMPONENTS system signals thread)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 2.4.8 REQUIRED)
find_package(PCL 1.7 REQUIRED)
find_package(Pangolin REQUIRED)

catkin_package(
  INCLUDE_DIRS include
#  LIBRARIES plane_slam2
#  CATKIN_DEPENDS cv_bridge message_filters roscpp sensor_msgs
#  DEPENDS system_lib
)

include_directories(
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${Pangolin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
  src/line_based_plane_segmentation.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  ${EIGEN3_LIBS}
  ${PCL_LIBRARIES}
)

# Build an example
add_library(segment_example SHARED
  examples/plane_segment.cpp
  examples/plane_viewer.cpp
  examples/organized_multi_plane_segmentor.cpp
)
target_link_libraries(segment_example
  ${OpenCV_LIBS}
  ${EIGEN3_LIBS}
  ${PCL_LIBRARIES}
  ${Pangolin_LIBRARIES}
  ${PROJECT_NAME}
)

set(LIBS
  ${OpenCV_LIBS}
  ${EIGEN3_LIBS}
  ${PCL_LIBRARIES}
  ${Pangolin_LIBRARIES}
  ${PROJECT_NAME}
  segment_example
  ${catkin_LIBRARIES}
)

#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/example)
add_executable(plane_segment_rgbd examples/plane_segment_rgbd.cpp)
target_link_libraries(plane_segment_rgbd ${LIBS})

add_executable(plane_segment_images examples/plane_segment_images.cpp)
target_link_libraries(plane_segment_images ${LIBS})

#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/example)
add_executable(omps_segment_rgbd examples/omps_segment_rgbd.cpp)
target_link_libraries(omps_segment_rgbd ${LIBS})

add_executable(omps_segment_images examples/omps_segment_images.cpp)
target_link_libraries(omps_segment_images ${LIBS})
