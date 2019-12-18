cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

set(CMAKE_BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

project(hsr-grasping)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


FIND_PACKAGE(PythonLibs 3.5 REQUIRED)

# --- boost
find_package(Boost COMPONENTS thread program_options serialization system filesystem regex)
include_directories(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS}) #${PYTHON_INCLUDE_DIRS})
link_libraries(${Boost_LIBRARIES} ${PYTHON_LIBRARIES}) #${PYTHON_LIBRARIES})

set(Boost_LIB ${Boost_LIBRARIES})
set(Boost_INC ${Boost_INCLUDE_DIRS})

# --- Eigen
find_package( Eigen3 REQUIRED )
include_directories( ${EIGEN3_INCLUDE_DIR} )

# --- PCL
find_package(PCL 1.7.2 REQUIRED COMPONENTS io common visualization features)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# --- Super4PCS
#set(Super4PCS_DIR "../3rdparty/Super4PCS/install/lib/cmake")

#find_package(Super4PCS REQUIRED)
#include_directories(${Super4PCS_INCLUDE_DIR})
#link_directories(${Super4PCS_LIB_DIR})

#message (STATUS "Super4PCS_INCLUDE_DIR: ${Super4PCS_INCLUDE_DIR}")
#message (STATUS "Super4PCS_LIB_DIR    : ${Super4PCS_LIB_DIR}")
#message (STATUS "Super4PCS_LIBRARIES  : ${Super4PCS_LIBRARIES}")

# --- pybind11
find_package(pybind11 REQUIRED)
#add_subdirectory(/home/dominik/projects/Scene-aware-MCTS/3rdparty/pybind11)

# --- PROJECT itself
# ICP wrapper
pybind11_add_module(icp SHARED icp.cpp)
target_link_libraries(icp PRIVATE ${Boost_LIB} ${PCL_LIBRARIES})# ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES})