/****************************************************************************
**
** Copyright (C) 2019 TU Wien, ACIN, Vision 4 Robotics (V4R) group
** Contact: v4r.acin.tuwien.ac.at
**
** This file is part of V4R
**
** V4R is distributed under dual licenses - GPLv3 or closed source.
**
** GNU General Public License Usage
** V4R is free software: you can redistribute it and/or modify
** it under the terms of the GNU General Public License as published
** by the Free Software Foundation, either version 3 of the License, or
** (at your option) any later version.
**
** V4R is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
** GNU General Public License for more details.
**
** Please review the following information to ensure the GNU General Public
** License requirements will be met: https://www.gnu.org/licenses/gpl-3.0.html.
**
**
** Commercial License Usage
** If GPL is not suitable for your project, you must purchase a commercial
** license to use V4R. Licensees holding valid commercial V4R licenses may
** use this file in accordance with the commercial license agreement
** provided with the Software or, alternatively, in accordance with the
** terms contained in a written agreement between you and TU Wien, ACIN, V4R.
** For licensing terms and conditions please contact office<at>acin.tuwien.ac.at.
**
**
** The copyright holder additionally grants the author(s) of the file the right
** to use, copy, modify, merge, publish, distribute, sublicense, and/or
** sell copies of their contributions without any restrictions.
**
****************************************************************************/

#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <ppf/local_coordinate.h>

// Forward-declare CMPH hash function type.
typedef struct __cmph_t cmph_t;

namespace ppf {

/// A search object that supports quick look ups of local coordinates (LC) for point pairs on a model.
///
/// Below is a description of the purpose of this class in very simple terms. For a complete discussion refer to
/// [VLLM18], especially section 2.1.
///
/// Suppose a pair of points (and their normals) are given. We can create a new coordinate system that is attached to
/// the first point and oriented such that the x axis is parallel to the first normal and the second point lies on the y
/// axis. In this coordinate system the whole arrangement can be described by only 4 numbers: 3 angles between normals
/// and the line joining the points, and the distance between the points. This 4-tuple is called Point Pair Feature
/// (PPF). What this class is able to do is answer queries of the following form: given a pair of points, does the model
/// have pairs of points with the same (or very similar) PPF? If so, what are the LCs of each of such pairs?
///
/// For increased efficiency, the returned LCs are "partial". Instead of the full rotation angle alpha they store a
/// partial rotation angle alpha_m. In order to obtain a "full" LC one needs to compute and subtract alpha_s from the
/// angle stored in "partial" LCs. See [VLLM18] figure 4 and explanation in preceding text for more details.
///
/// The users of this class will eventually want to get SE(3) transformations associated with "full" LCs. This can be
/// accomplished with the computeTransform() method.
///
/// Note that differently from [VLLM18] feature spreading (section 2.2.3) is applied at offline stage (i.e. when the
/// search object is built). There is no spreading at online stage, thus every find() call performs a single hash
/// lookup.
///
/// Additionally, this class supports look ups of reduced LC sets for models that have rotational symmetry. Point
/// clouds of such models can be divided into two parts: the "unique" part and the "symmetrical" part. The second part
/// can be obtained by applying rotations to the "unique" part. This class can output reduced LC sets that are anchored
/// only at the "unique" points. To enable this behavior, at construction time the user should pass the number of
/// anchor points. The point cloud should be arranged such that the anchor points are located at the beginning of
/// the point vector.
class ModelSearch {
 public:
  using Ptr = std::shared_ptr<ModelSearch>;
  using ConstPtr = std::shared_ptr<const ModelSearch>;

  /// Enum to control whether the search object should be constructed with feature spreading.
  enum class Spreading {
    On,  ///< Enable feature spreading.
    Off  ///< Disable feature spreading.
  };

  /// Build a search object for a given model point cloud.
  /// \param[in] model point cloud with normals
  /// \param[in] distance_quantization_step step used to quantize distances in PPFs
  /// \param[in] angle_quantization_step step used to quantize angles in PPFs
  /// \param[in] spreading flag that controls whether feature spreading is enabled
  /// \param[in] num_anchor_points number of anchor points, see class description for details. Zero means that all
  //             points are anchor points, i.e. the model has no rotational symmetries.
  template <typename PointT>
  ModelSearch(const pcl::PointCloud<PointT>& model, float distance_quantization_step, float angle_quantization_step,
              Spreading spreading, size_t num_anchor_points = 0)
  : ModelSearch(model.getMatrixXfMap(3, sizeof(PointT) / sizeof(float), 0),
                model.getMatrixXfMap(3, sizeof(PointT) / sizeof(float), 4), distance_quantization_step,
                angle_quantization_step, spreading, num_anchor_points) {
    static_assert(pcl::traits::has_xyz<PointT>::value, "PointT should have xyz fields");
    static_assert(pcl::traits::has_normal<PointT>::value, "PointT should have normal fields");
  };

  /// Load a pre-built search object from a file.
  /// Such a file can be created using the save() method.
  explicit ModelSearch(const std::string&);

  ~ModelSearch();

  /// Look up local coordinates on the model that correspond to the first point in a given pair.
  /// \param[in] p1 coordinates of the first point in the pair
  /// \param[in] n1 normal of the first point in the pair
  /// \param[in] p2 coordinates of the second point in the pair
  /// \param[in] n2 normal of the second point in the pair
  const LocalCoordinate::Vector& find(const Eigen::Vector3f& p1, const Eigen::Vector3f& n1, const Eigen::Vector3f& p2,
                                      const Eigen::Vector3f& n2) const;

  /// Look up local coordinates on the model that correspond to the first point in a given pair.
  /// \param[in] p1 first point in the pair
  /// \param[in] p2 second point in the pair
  /// \tparam PointT PCL point type with xyz and normal fields
  template <typename PointT>
  const LocalCoordinate::Vector& find(const PointT& p1, const PointT& p2) const {
    return find(p1.getVector3fMap(), p1.getNormalVector3fMap(), p2.getVector3fMap(), p2.getNormalVector3fMap());
  }

  /// Compute SE(3) transform associated with a given local coordinate.
  /// This implements R_x(alpha) T_m part of [VLLM18] equation 3.
  /// \param[in] lc local coordinate on the model
  /// \param[out] transform SE(3) transform that corresponds to the local coordinate
  void computeTransform(const LocalCoordinate& lc, Eigen::Affine3f& transform) const;

  /// Get the total number of points in the model point cloud.
  inline size_t getNumPoints() const {
    return num_points_;
  }

  /// Get the number of anchor points.
  /// The LocalCoordinate::model_point_index values output by this search object are guaranteed to be in the range from
  /// 0 to this value (exclusive).
  inline size_t getNumAnchorPoints() const {
    return num_anchor_points_;
  }

  /// Get diameter of the model (largest distance between any pair of model points).
  inline float getModelDiameter() const {
    return model_diameter_;
  }

  /// Get step used to quantize distances in PPFs before search.
  inline float getDistanceQuantizationStep() const {
    return distance_quantization_step_;
  }

  /// Get step used to quantize angles in PPFs before search.
  inline float getAngleQuantizationStep() const {
    return angle_quantization_step_;
  }

  /// Get a flag indicating whether the search hash table was constructed with feature spreading or not.
  inline Spreading getSpreading() const {
    return spreading_;
  }

  /// Get the model point cloud.
  ///
  /// This function supports outputting model points in point clouds of arbitrary type. If the output cloud type has
  /// XYZ field, then the point coordinates will be written. If the output cloud type has normal field, then the point
  /// normals will be written. All the other fields in the output cloud will remain unmodified.
  ///
  /// A static assert will be triggered if the output point type has neither XYZ nor normal fields.
  template <typename PointT>
  void getModelPointCloud(pcl::PointCloud<PointT>& cloud) const {
    static_assert(pcl::traits::has_xyz<PointT>::value || pcl::traits::has_normal<PointT>::value,
                  "PointT should have either xyz or normal fields");
    cloud.resize(num_points_);
    cloud.is_dense = true;
    if (pcl::traits::has_xyz<PointT>::value) {
      getModelPoints(cloud.getMatrixXfMap(3, sizeof(PointT) / sizeof(float), 0));
      if (pcl::traits::has_normal<PointT>::value) {
        getModelNormals(cloud.getMatrixXfMap(3, sizeof(PointT) / sizeof(float), 4));
      }
    } else {
      getModelNormals(cloud.getMatrixXfMap(3, sizeof(PointT) / sizeof(float), 0));
    }
  }

  /// Get the model point cloud.
  ///
  /// This overload returns a point cloud instead of writing it to a given reference. See documentation for the other
  /// overload for more information.
  template <typename PointT>
  pcl::PointCloud<PointT> getModelPointCloud() const {
    pcl::PointCloud<PointT> cloud;
    getModelPointCloud(cloud);
    return cloud;
  }

  /// Save the search object to a file.
  void save(const std::string&) const;

  /// Load a pre-built search object from a file.
  /// This works the same as loading constructor.
  static ModelSearch load(const std::string&);

 private:
  /// Build a search object for a given model represented by matrices of points and normals.
  /// This is an internal constructor that does not need to be templated on point type.
  ModelSearch(const Eigen::Matrix<float, 3, Eigen::Dynamic>& model_points,
              const Eigen::Matrix<float, 3, Eigen::Dynamic>& model_normals, float distance_quantization_step,
              float angle_quantization_step, Spreading spreading, size_t num_anchor_points);

  /// Get model points.
  /// This is an internal getter that writes into a mapped Eigen matrix to avoid being templated on point type.
  void getModelPoints(Eigen::Map<Eigen::MatrixXf, Eigen::Aligned, Eigen::OuterStride<>> model_points) const;

  /// Get model normals.
  /// This is an internal getter that writes into a mapped Eigen matrix to avoid being templated on point type.
  void getModelNormals(Eigen::Map<Eigen::MatrixXf, Eigen::Aligned, Eigen::OuterStride<>> model_normals) const;

  // In principle, all components of a PPF are supposed to be non-negative, thus an unsigned int would be more
  // appropriate to store quantized values. However, using signed ints makes implementation of spreading more
  // straightforward.
  using QuantizedPPF = std::array<int32_t, 4>;

  /// A helper function to quantize a PPF.
  /// This does not perform feature spreading, thus the output is always a single quantized PPF.
  QuantizedPPF quantizePPF(const float ppf[4]) const;

  /// A helper function to quantize a PPF, optionally spreading it.
  /// \param[in] ppf PPF
  /// \param[in] with_spreading flag indicating whether spreading should be performed
  /// \returns a vector of quantized PPFs, will contain a single item if spreading is disabled
  std::vector<QuantizedPPF> quantizePPF(const float ppf[4], bool with_spreading) const;

  /// Number of points in the model point cloud
  size_t num_points_;
  /// Number of anchor points in the model point cloud
  size_t num_anchor_points_;
  /// Diameter of the model (largest distance between any pair of model points)
  float model_diameter_;
  /// Quantization step for distances in PPF
  float distance_quantization_step_;
  /// Quantization step for angles in PPF
  float angle_quantization_step_;
  /// State of feature spreading (on/off)
  Spreading spreading_;
  /// CMPH hash function used to convert quantized PPFs into an index into a table with LCs
  cmph_t* hash_function_ = nullptr;
  /// A table with LCs for each possible (present in the model) quantized PPF
  std::vector<LocalCoordinate::Vector> lcs_table_;
  /// A table with each possible (present in the model) quantized PPF
  /// This table, like lcs_table_, is indexed by the CMPH hash function
  std::vector<QuantizedPPF> qppf_table_;
  /// Translations of local coordinate frames associated with model points.
  std::vector<Eigen::Translation3f, Eigen::aligned_allocator<Eigen::Translation3f>> lc_translations_;
  /// Rotations of local coordinate frames associated with model points.
  std::vector<Eigen::Quaternionf, Eigen::aligned_allocator<Eigen::Quaternionf>> lc_rotations_;
};

}  // namespace ppf
