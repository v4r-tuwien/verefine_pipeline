#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;


#include <pcl/console/parse.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/recognition/ransac_based/auxiliary.h>
#include <pcl/recognition/ransac_based/trimmed_icp.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/console/print.h>
#include <pcl/console/time.h>

using namespace pcl;
using namespace pcl::io;
using namespace pcl::console;


// Types
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

PointCloudT::Ptr cloud_to_pcd(double* points, unsigned long num_points)
{
    PointCloudT in_cloud;
    for (unsigned long i = 0; i < num_points; i++)
    {
        unsigned long point_idx = i*3;
        PointT point;
        point.x = (float) points[point_idx];
        point.y = (float) points[point_idx+1];
        point.z = (float) points[point_idx+2];
        in_cloud.push_back(point);
    }
    PointCloudT::Ptr cloud = boost::shared_ptr<PointCloudT>(new PointCloudT(in_cloud));
    return cloud;
}

Eigen::Matrix4f performTrICP(double* points_ren, unsigned long num_points_ren,
                  double* points_obs, unsigned long num_points_obs,
                  float trim)
{
    // parse input buffer to point cloud
    PointCloudT::Ptr cloud_ren = cloud_to_pcd(points_ren, num_points_ren);
    PointCloudT::Ptr cloud_obs = cloud_to_pcd(points_obs, num_points_obs);

    // initialize trimmed icp
    pcl::recognition::TrimmedICP<PointT, float> tricp;
    tricp.init(cloud_obs);
    tricp.setNewToOldEnergyRatio(1.f);

    // TODO remove already explained points

    // compute trafo
    Eigen::Matrix4f tform;
    tform.setIdentity();
    float num_points = trim * num_points_ren;
    tricp.align(*cloud_ren, abs(num_points), tform);

    return tform;
}

Eigen::Matrix4f performICP(double* points_ren, unsigned long num_points_ren,
                  double* points_obs, unsigned long num_points_obs,
                  unsigned int max_iterations)
{
    // parse input buffer to point cloud
    PointCloudT::Ptr cloud_ren = cloud_to_pcd(points_ren, num_points_ren);
    PointCloudT::Ptr cloud_obs = cloud_to_pcd(points_obs, num_points_obs);


    // initialize ICP
    pcl::IterativeClosestPoint<PointT, PointT> icp;
    icp.setMaximumIterations(max_iterations);
    icp.setInputCloud(cloud_ren);
    icp.setInputTarget(cloud_obs);

    // compute trafo
    PointCloudT cloud_fit;
    icp.align(cloud_fit);
    Eigen::Matrix4f tform = icp.getFinalTransformation();

    return tform;
}

PYBIND11_MODULE(icp, m) {
    m.def("tricp", [](py::array_t<double> points_ren_array,
                      py::array_t<double> points_obs_array,
                      float trim)
    {
        // input
        auto points_ren_buf = points_ren_array.request();
        if (points_ren_buf.ndim != 2)
            throw std::runtime_error("Expect ren array of shape (num_points, XYZ).");
        if (points_ren_buf.shape[1] != 3)
            throw std::runtime_error("Expect ren array of shape (num_points, XYZ).");
        unsigned long num_points_ren = (unsigned long) points_ren_buf.shape[0];
        auto points_ren = (double*) points_ren_buf.ptr;

        auto points_obs_buf = points_obs_array.request();
        if (points_obs_buf.ndim != 2)
            throw std::runtime_error("Expect obs array of shape (num_points, XYZ).");
        if (points_obs_buf.shape[1] != 3)
            throw std::runtime_error("Expect obs array of shape (num_points, XYZ).");

        unsigned long num_points_obs = (unsigned long) points_obs_buf.shape[0];
        auto points_obs = (double*) points_obs_buf.ptr;

        // compute segmentation
        Eigen::Matrix4f T = performTrICP(points_ren, num_points_ren,
                                         points_obs, num_points_obs,
                                         trim);
        T.transposeInPlace(); // s.t. it is correctly transferred to numpy

        // output
        return py::array_t<float>(
                    py::buffer_info(
                       &T, // pointer
                       sizeof(float), //itemsize
                       py::format_descriptor<float>::format(),
                       2, // ndim
                       std::vector<size_t> { 4, 4 }, // shape
                       std::vector<size_t> { 4 * sizeof(float), sizeof(float)} // strides
                   )
            );
    }, "Register input1 to input2 using at most input1.shape[0]*input3 points.");

    m.def("icp", [](py::array_t<double> points_ren_array,
                    py::array_t<double> points_obs_array,
                    unsigned int max_iterations)
    {
        // input
        auto points_ren_buf = points_ren_array.request();
        if (points_ren_buf.ndim != 2)
            throw std::runtime_error("Expect ren array of shape (num_points, XYZ).");
        if (points_ren_buf.shape[1] != 3)
            throw std::runtime_error("Expect ren array of shape (num_points, XYZ).");
        unsigned long num_points_ren = (unsigned long) points_ren_buf.shape[0];
        auto points_ren = (double*) points_ren_buf.ptr;

        auto points_obs_buf = points_obs_array.request();
        if (points_obs_buf.ndim != 2)
            throw std::runtime_error("Expect obs array of shape (num_points, XYZ).");
        if (points_obs_buf.shape[1] != 3)
            throw std::runtime_error("Expect obs array of shape (num_points, XYZ).");

        unsigned long num_points_obs = (unsigned long) points_obs_buf.shape[0];
        auto points_obs = (double*) points_obs_buf.ptr;

        // compute segmentation
        Eigen::Matrix4f T = performICP(points_ren, num_points_ren,
                                       points_obs, num_points_obs,
                                       max_iterations);
        T.transposeInPlace(); // s.t. it is correctly transferred to numpy

        // output
        return py::array_t<float>(
                    py::buffer_info(
                       &T, // pointer
                       sizeof(float), //itemsize
                       py::format_descriptor<float>::format(),
                       2, // ndim
                       std::vector<size_t> { 4, 4 }, // shape
                       std::vector<size_t> { 4 * sizeof(float), sizeof(float)} // strides
                   )
            );
    }, "Register input1 to input2 using at most input3 iterations.");

}
