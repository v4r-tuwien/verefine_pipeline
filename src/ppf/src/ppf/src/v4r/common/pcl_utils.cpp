#include <v4r/common/pcl_utils.h>
#include <v4r/common/point_types.h>
#include <pcl/impl/instantiate.hpp>

namespace v4r {

template <typename PointT>
void setCloudPose(const Eigen::Matrix4f &trans, pcl::PointCloud<PointT> &cloud) {
  cloud.sensor_origin_ = trans.block<4, 1>(0, 3);
  Eigen::Matrix3f rotation = trans.block<3, 3>(0, 0);
  Eigen::Quaternionf q(rotation);
  q.normalize();
  cloud.sensor_orientation_ = q;
}

#define PCL_INSTANTIATE_setCloudPose(T) \
  template  void setCloudPose<T>(const Eigen::Matrix4f &, pcl::PointCloud<T> &);
PCL_INSTANTIATE(setCloudPose, V4R_PCL_XYZ_POINT_TYPES(pcl::Normal))
}  // namespace v4r

namespace pcl {
//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT>
void copyPointCloud(const pcl::PointCloud<PointT> &cloud_in, const std::vector<size_t> &indices,
                    pcl::PointCloud<PointT> &cloud_out) {
  // Do we want to copy everything?
  if (indices.size() == cloud_in.points.size()) {
    cloud_out = cloud_in;
    return;
  }

  // Allocate enough space and copy the basics
  cloud_out.points.resize(indices.size());
  cloud_out.header = cloud_in.header;
  cloud_out.width = static_cast<uint32_t>(indices.size());
  cloud_out.height = 1;
  cloud_out.is_dense = cloud_in.is_dense;
  cloud_out.sensor_orientation_ = cloud_in.sensor_orientation_;
  cloud_out.sensor_origin_ = cloud_in.sensor_origin_;

  // Iterate over each point
  for (size_t i = 0; i < indices.size(); ++i)
    cloud_out.points[i] = cloud_in.points[indices[i]];
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT>
void copyPointCloud(const pcl::PointCloud<PointT> &cloud_in,
                    const std::vector<size_t, Eigen::aligned_allocator<size_t>> &indices,
                    pcl::PointCloud<PointT> &cloud_out) {
  // Do we want to copy everything?
  if (indices.size() == cloud_in.points.size()) {
    cloud_out = cloud_in;
    return;
  }

  // Allocate enough space and copy the basics
  cloud_out.points.resize(indices.size());
  cloud_out.header = cloud_in.header;
  cloud_out.width = static_cast<uint32_t>(indices.size());
  cloud_out.height = 1;
  cloud_out.is_dense = cloud_in.is_dense;
  cloud_out.sensor_orientation_ = cloud_in.sensor_orientation_;
  cloud_out.sensor_origin_ = cloud_in.sensor_origin_;

  // Iterate over each point
  for (size_t i = 0; i < indices.size(); ++i)
    cloud_out.points[i] = cloud_in.points[indices[i]];
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT>
void copyPointCloud(const pcl::PointCloud<PointT> &cloud_in, const std::vector<bool> &mask,
                    pcl::PointCloud<PointT> &cloud_out) {
  assert(cloud_in.points.size() == mask.size());

  // Allocate enough space and copy the basics
  cloud_out.points.resize(cloud_in.points.size());
  cloud_out.header = cloud_in.header;
  cloud_out.width = static_cast<uint32_t>(mask.size());
  cloud_out.height = 1;
  cloud_out.is_dense = cloud_in.is_dense;
  cloud_out.sensor_orientation_ = cloud_in.sensor_orientation_;
  cloud_out.sensor_origin_ = cloud_in.sensor_origin_;

  // Iterate over each point
  size_t kept = 0;
  for (size_t i = 0; i < mask.size(); ++i) {
    if (mask[i]) {
      cloud_out.points[kept] = cloud_in.points[i];
      kept++;
    }
  }
  cloud_out.points.resize(kept);
  cloud_out.width = kept;
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT>
void copyPointCloud(const pcl::PointCloud<PointT> &cloud_in, const boost::dynamic_bitset<> &mask,
                    pcl::PointCloud<PointT> &cloud_out) {
  assert(cloud_in.points.size() == mask.size());

  // Allocate enough space and copy the basics
  cloud_out.points.resize(cloud_in.points.size());
  cloud_out.header = cloud_in.header;
  cloud_out.width = static_cast<uint32_t>(mask.size());
  cloud_out.height = 1;
  cloud_out.is_dense = cloud_in.is_dense;
  cloud_out.sensor_orientation_ = cloud_in.sensor_orientation_;
  cloud_out.sensor_origin_ = cloud_in.sensor_origin_;

  // Iterate over each point
  size_t kept = 0;
  for (size_t i = 0; i < mask.size(); ++i) {
    if (mask[i]) {
      cloud_out.points[kept] = cloud_in.points[i];
      kept++;
    }
  }
  cloud_out.points.resize(kept);
  cloud_out.width = kept;
}

#define PCL_INSTANTIATE_copyPointCloud(T) \
  template  void copyPointCloud<T>(const PointCloud<T> &, const boost::dynamic_bitset<> &, PointCloud<T> &);
PCL_INSTANTIATE(copyPointCloud, V4R_PCL_XYZ_POINT_TYPES(Normal))
#undef PCL_INSTANTIATE_copyPointCloud

#define PCL_INSTANTIATE_copyPointCloud(T) \
  template  void copyPointCloud<T>(const PointCloud<T> &, const std::vector<bool> &, PointCloud<T> &);
PCL_INSTANTIATE(copyPointCloud, V4R_PCL_XYZ_POINT_TYPES(Normal))
#undef PCL_INSTANTIATE_copyPointCloud

#define PCL_INSTANTIATE_copyPointCloud(T)      \
  template  void copyPointCloud<T>( \
      const PointCloud<T> &, const std::vector<size_t, Eigen::aligned_allocator<size_t>> &, PointCloud<T> &);
PCL_INSTANTIATE(copyPointCloud, V4R_PCL_XYZ_POINT_TYPES(Normal))
#undef PCL_INSTANTIATE_copyPointCloud

#define PCL_INSTANTIATE_copyPointCloud(T) \
  template  void copyPointCloud<T>(const PointCloud<T> &, const std::vector<size_t> &, PointCloud<T> &);
PCL_INSTANTIATE(copyPointCloud, V4R_PCL_XYZ_POINT_TYPES(Normal))
#undef PCL_INSTANTIATE_copyPointCloud
}  // namespace pcl
