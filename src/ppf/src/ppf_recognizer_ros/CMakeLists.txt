cmake_minimum_required(VERSION 2.8.3)
project(ppf_recognizer_ros)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin
  REQUIRED
  COMPONENTS  roscpp
              rospy
              ppf_recognizer_ros_msgs
              object_detector_msgs
              genmsg
              cv_bridge
              sensor_msgs
              image_transport
              pcl_conversions
)

# make sure cmake can find PPFRecognizer exported targets from its build tree
list(APPEND CMAKE_PREFIX_PATH "/home/jeremy/projects/ppf/build")
list(APPEND CMAKE_MODULE_PATH "/home/jeremy/projects/ppf/build")

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PPFRecognizer REQUIRED)

# declare dependencies for other packages
catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp
  rospy
  ppf_recognizer_ros_msgs
  object_detector_msgs
  pcl_conversions
)

add_definitions(${PCL_DEFINITIONS})

include_directories(include
  ${PCL_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

set(incs include/recognizer_ros.h)

add_executable(recognition_service src/recognizer_ros.cpp ${incs})
add_executable(test_object_recognition src/test_object_recognizer.cpp)

add_dependencies(recognition_service ${catkin_EXPORTED_TARGETS})
add_dependencies(test_object_recognition ${catkin_EXPORTED_TARGETS})

target_link_libraries(recognition_service
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  PPFRecognizer::ppf-recognizer
)

target_link_libraries(test_object_recognition
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  PPFRecognizer::ppf-recognizer
  ${OpenCV_LIBS}
)


install(TARGETS recognition_service
ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS test_object_recognition
ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/
DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
FILES_MATCHING PATTERN "*.h"
PATTERN ".svn" EXCLUDE
)

install(DIRECTORY launch/
DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
FILES_MATCHING PATTERN "*.launch"
)
