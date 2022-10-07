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
 * @file recognition_pipeline.h
 * @author Aitor Aldoma (aldoma@acin.tuwien.ac.at), Thomas Faeulhammer (faeulhammer@acin.tuwien.ac.at)
 * @date 2017
 * @brief
 *
 */

#pragma once

#include <pcl/common/common.h>

// #include <v4r/common/pcl_visualization_utils.h>   //@TODO: remove if not needed
#include <v4r/common/point_types.h>
// #include <v4r/config.h>                           //@TODO: remove if not needed
#include <v4r/recognition/object_hypothesis.h>
#include <v4r/recognition/source.h>
#include <boost/date_time/posix_time/posix_time.hpp>
// #include <boost/none.hpp>
#include <boost/optional/optional.hpp>

namespace v4r {

/**
 * @brief The recognition pipeline class is an abstract class that represents a
 * pipeline for object recognition. It will generated groups of object hypotheses.
 * For a global recognition pipeline, each segmented cluster from the input cloud will store its object hypotheses into
 * one group.
 * For all other pipelines, each group will only contain one object hypothesis.
 * @author Thomas Faeulhammer, Aitor Aldoma
 */
template <typename PointT>
class RecognitionPipeline {
 public:
  typedef std::shared_ptr<RecognitionPipeline<PointT>> Ptr;
  typedef std::shared_ptr<RecognitionPipeline<PointT> const> ConstPtr;

 protected:
  using PointTWithNormal = v4r::add_normal_t<PointT>;

  typedef Model<PointT> ModelT;

  typename pcl::PointCloud<PointT>::ConstPtr scene_;      ///< Point cloud to be recognized
  pcl::PointCloud<pcl::Normal>::ConstPtr scene_normals_;  ///< associated normals
  typename Source<PointT>::ConstPtr m_db_;                ///< model data base
  std::vector<ObjectHypothesesGroup> obj_hypotheses_;     ///< generated object hypotheses
  Eigen::Vector4f table_plane_;
  bool table_plane_set_;
  Eigen::Matrix4f transform_to_world_;  ///< rigid transform that aligns camera to world reference frame
  bool transform_to_world_set_;

  static std::vector<std::pair<std::string, float>> elapsed_time_;  ///< to measure performance

  virtual void doInit(const bf::path &trained_dir, bool retrain,
                      const std::vector<std::string> &object_instances_to_load) = 0;

  void deInit();

  class StopWatch {
    std::string desc_;
    boost::posix_time::ptime start_time_;

   public:
    StopWatch(const std::string &desc) : desc_(desc), start_time_(boost::posix_time::microsec_clock::local_time()) {}

    ~StopWatch();
  };

  // PCLVisualizationParams::ConstPtr vis_param_;

  virtual void do_recognize(const std::vector<std::string> &model_ids_to_search) = 0;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  RecognitionPipeline() : table_plane_(Eigen::Vector4f::Identity()), table_plane_set_(false) {}

  virtual ~RecognitionPipeline() = default;

  virtual size_t getFeatureType() const = 0;

  virtual bool needNormals() const = 0;

  /**
   * @brief initialize the recognizer (extract features, create FLANN,...)
   * @param[in] path to model database. If training directory exists, will load trained model from disk; if not,
   * computed features will be stored on disk (in each
   * object model folder, a feature folder is created with data)
   * @param[in] retrain if set, will re-compute features and store to disk, no matter if they already exist or not
   * @param[in] object_instances_to_load vector of object models to load from model_database_path. If empty, all objects
   * in directory will be loaded.
   */
  void initialize(const bf::path &trained_dir = "", bool retrain = false,
                  const std::vector<std::string> &object_instances_to_load = {}) {
    doInit(trained_dir, retrain, object_instances_to_load);
  }

  /**
   * @brief setInputCloud
   * @param cloud to be recognized
   */
  void setInputCloud(const typename pcl::PointCloud<PointT>::ConstPtr cloud) {
    scene_ = cloud;
  }

  /**
   * @brief getObjectHypothesis
   * @return generated object hypothesis
   */
  std::vector<ObjectHypothesesGroup> getObjectHypothesis() const {
    return obj_hypotheses_;
  }

  /**
   * @brief setSceneNormals
   * @param normals normals of the input cloud
   */
  void setSceneNormals(const pcl::PointCloud<pcl::Normal>::ConstPtr &normals) {
    scene_normals_ = normals;
  }

  /**
   * @brief setModelDatabase
   * @param m_db model database
   */
  void setModelDatabase(const typename Source<PointT>::ConstPtr &m_db) {
    m_db_ = m_db;
  }

  /**
   * @brief getModelDatabase
   * @return model database
   */
  typename Source<PointT>::ConstPtr getModelDatabase() const {
    return m_db_;
  }

  void setTablePlane(const Eigen::Vector4f &table_plane) {
    table_plane_ = table_plane;
    table_plane_set_ = true;
  }

  /**
   * @brief set the rigid transform that aligns camera to world reference frame
   * @param transform_to_world rigid transform that aligns camera to world reference frame
   */
  void setTransformToWorld(const Eigen::Matrix4f &transform_to_world) {
    transform_to_world_ = transform_to_world;
    transform_to_world_set_ = true;
  }

  /**
   * @brief setVisualizationParameter sets the PCL visualization parameter (only used if some visualization is enabled)
   * @param vis_param
   */
  // void setVisualizationParameter(const PCLVisualizationParams::ConstPtr &vis_param) {
  //   vis_param_ = vis_param;
  // }

  /**
   * @brief getElapsedTimes
   * @return compuation time measurements for various components
   */
  std::vector<std::pair<std::string, float>> getElapsedTimes() const {
    return elapsed_time_;
  }

  /**
   * @brief recognize objects
   * @param model_ids_to_search object model identities to search
   */
  void recognize(const std::vector<std::string> &model_ids_to_search,
                 const boost::optional<Eigen::Matrix4f> &camera_pose = boost::none);
};

template <typename PointT>
std::vector<std::pair<std::string, float>> RecognitionPipeline<PointT>::elapsed_time_;

}  // namespace v4r
