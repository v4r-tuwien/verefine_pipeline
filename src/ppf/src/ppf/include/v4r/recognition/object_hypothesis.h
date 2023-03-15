/****************************************************************************
**
** Copyright (C) 2017 TU Wien, ACIN, Vision 4 Robotics (V4R) group
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

/**
 * @file object_hypothesis.h
 * @author Aitor Aldoma (aldoma@acin.tuwien.ac.at), Thomas Faeulhammer (faeulhammer@acin.tuwien.ac.at)
 * @date 2017
 * @brief
 *
 */

#pragma once

#include <pcl/common/common.h>
#include <Eigen/Sparse>
#include <Eigen/StdVector>
#include <atomic>
#include <boost/dynamic_bitset.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/vector.hpp>
#include <opencv2/core/matx.hpp>

#include <v4r/common/point_types.h>
#include <v4r/recognition/model.h>
#include <v4r/recognition/source.h>

namespace v4r {

class  ObjectHypothesis {
 private:
  friend class boost::serialization::access;

  template <class Archive>
   void serialize(Archive& ar, const unsigned int version) {
    (void)version;
    ar& class_id_& model_id_& transform_& pose_refinement_& confidence_& is_verified_& unique_id_;
  }

  static std::atomic<size_t> s_counter_;  /// unique identifier to avoid transferring hypotheses multiple times when
                                          /// using multi-view recognition

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef std::shared_ptr<ObjectHypothesis> Ptr;
  typedef std::shared_ptr<ObjectHypothesis const> ConstPtr;

  pcl::Correspondences corr_;  ///< local feature matches / keypoint correspondences between model and scene (only for
                               /// visualization purposes)

  ObjectHypothesis() : unique_id_(s_counter_.fetch_add(1)) {}

  ObjectHypothesis(const ObjectHypothesis& other)
  : class_id_(other.class_id_), model_id_(other.model_id_), transform_(other.transform_),
    pose_refinement_(other.pose_refinement_), is_verified_(other.is_verified_), unique_id_(other.unique_id_) {}

  std::string class_id_ = "";  ///< category
  std::string model_id_ = "";  ///< instance
  Eigen::Matrix4f transform_ =
      Eigen::Matrix4f::Identity();  ///< 4x4 homogenous transformation to project model into camera coordinate system.
  Eigen::Matrix4f pose_refinement_ =
      Eigen::Matrix4f::Identity();  ///< pose refinement (to be multiplied by transform to get refined pose)
  float confidence_ = 0.f;          ///< confidence score (coming from HV)
  float confidence_wo_hv_ = 0.f;    ///< confidence score (coming from feature matching stage)
  bool is_verified_ = false;
  size_t unique_id_;

  virtual ~ObjectHypothesis() {}
};

class  ModelSceneCorrespondence {
 private:
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive& ar, const unsigned int file_version) {
    (void)file_version;
    ar& scene_id_& model_id_& color_distance_& normals_dotp_& fitness_;
  }

 public:
  size_t scene_id_;       /// Index of scene point.
  size_t model_id_;       /// Index of matching model point.
  float color_distance_;  /// Distance between the corresponding points in color
  float normals_dotp_;    /// Angle in degree between surface normals
  float fitness_;         /// model fitness score

  bool operator<(const ModelSceneCorrespondence& other) const {
    return this->fitness_ > other.fitness_;
  }

  /** \brief Constructor. */
  ModelSceneCorrespondence(size_t scene_id, size_t model_id)
  : scene_id_(scene_id), model_id_(model_id), color_distance_(std::numeric_limits<float>::quiet_NaN()),
    normals_dotp_(M_PI / 2.f), fitness_(0.f) {}
};

template <typename PointT>
class  HVRecognitionModel {
 private:
  using PointTWithNormal = v4r::add_normal_t<PointT>;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef std::shared_ptr<HVRecognitionModel> Ptr;
  typedef std::shared_ptr<HVRecognitionModel const> ConstPtr;

  ObjectHypothesis::Ptr oh_;  ///< object hypothesis
  size_t num_pts_full_model_ = 0;
  typename pcl::PointCloud<PointTWithNormal>::Ptr visible_cloud_;
  std::vector<boost::dynamic_bitset<>> image_mask_;  ///< image mask per view (in single-view case, there will be only
                                                     /// one element in outer vector). Used to compute pairwise
  /// intersection
  std::vector<int> visible_indices_;  ///< visible indices computed by z-Buffering (for model self-occlusion) and
                                      /// occlusion reasoning with scene cloud
  std::vector<int> visible_indices_by_octree_;  ///< visible indices computed by creating an octree for the model and
                                                /// checking which leaf nodes are occupied by a visible point computed
  /// from the z-buffering approach
  std::vector<ModelSceneCorrespondence> model_scene_c_;  ///< correspondences between visible model points and scene
  float model_fit_ =
      0.f;  ///< the fitness score of the visible cloud to the model scene (sum of model_scene_c correspondenes
            /// weight divided by the number of visible points)

  std::vector<cv::Vec3b> pt_color_;  ///< color values for each point of the (complete) model
  float mean_brigthness_;            ///< average value of the L channel for all visible model points
  float mean_brigthness_scene_;      ///< average value of the L channel for all scene points close to the visible model
                                     /// points
  std::vector<int> scene_indices_in_crop_box_;  ///< indices of the scene that are occupied from the bounding box of the
                                                ///(complete) hypothesis
  float L_value_offset_ = 0.f;  ///< the offset being added to the computed L color values to compensate for different
                                /// lighting conditions

  Eigen::SparseVector<float>
      scene_explained_weight_;  ///< stores for each scene point how well it is explained by the visible model points

  bool rejected_due_to_low_visibility_ =
      false;                 ///< true if the object model rendered in the view is not visible enough
  bool is_outlier_ = false;  ///< true if the object model is not able to explain the scene well enough
  bool rejected_due_to_better_hypothesis_in_group_ =
      false;  ///< true if there is any other object model in the same hypotheses
              /// group which explains the scene better
  bool rejected_globally_ = false;
  bool rejected_due_to_similar_hypothesis_exists_ =
      false;  ///< true if there is another hypothesis that is very similar in
  ///< terms of object pose
  bool rejected_due_to_outline_mismatch_ = false;  ///< true if outline of model does not match scene outlines

  boost::dynamic_bitset<> on_smooth_cluster_;  ///< each bit represents whether or not hypotheses lies on the smooth
                                               ///< cluster with the id corresponding to the bit id

  bool violates_smooth_cluster_check_ = false;  ///< true if the visible hypotheses violates the smooth cluster check
  bool rejected_due_to_smooth_cluster_violation = false;  ///< true if the hypotheses is the only one describing a
                                                          ///< cluster that it violates

  bool rejected_due_to_be_under_floor_ = false;

  explicit HVRecognitionModel(ObjectHypothesis::Ptr& oh) {
    oh_ = oh;
  }

  bool isRejected() const {
    return is_outlier_ || rejected_due_to_low_visibility_ || rejected_globally_ ||
           rejected_due_to_better_hypothesis_in_group_ || rejected_due_to_similar_hypothesis_exists_ ||
           rejected_due_to_smooth_cluster_violation || rejected_due_to_be_under_floor_ ||
           rejected_due_to_outline_mismatch_;
  }

  /**
   * @brief does dilation and erosion on the occupancy image of the rendered point cloud
   * @param do_smoothing
   * @param smoothing_radius
   * @param do_erosion
   * @param erosion_radius
   * @param img_width
   */
  void processSilhouette(bool do_smoothing = true, int smoothing_radius = 2, bool do_erosion = true,
                         int erosion_radius = 4, int img_width = 640);
};

class  ObjectHypothesesGroup {
 private:
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    (void)version;
    ar& global_hypotheses_& ohs_;
    //        size_t nItems = ohs_.size();
    //        ar & nItems;
    //        for (const auto &oh : ohs_) {  ar & *oh; }
  }

 public:
  typedef std::shared_ptr<ObjectHypothesesGroup> Ptr;
  typedef std::shared_ptr<ObjectHypothesesGroup const> ConstPtr;

  ObjectHypothesesGroup() {}
  //    ObjectHypothesesGroup(const std::string &filename, const Source<PointT> &src);
  //    void save(const std::string &filename) const;

  std::vector<ObjectHypothesis::Ptr> ohs_;  ///< Each hypothesis can have several object model (e.g. global
                                            /// recognizer tries to macht several object instances for a
  /// clustered point cloud segment).
  bool global_hypotheses_;  ///< if true, hypothesis was generated by global recognition pipeline. Otherwise, from local
                            /// feature matches-
};
}  // namespace v4r
