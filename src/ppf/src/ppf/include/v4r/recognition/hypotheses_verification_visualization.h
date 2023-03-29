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
 * @file hypotheses_verification_visualization.h
 * @author Thomas Faeulhammer (faeulhammer@acin.tuwien.ac.at)
 * @date 2017
 * @brief Visualization functions for hypotheses verificatino
 *
 */

#pragma once

#include <v4r/common/pcl_visualization_utils.h>
#include <v4r/common/point_types.h>

#include <v4r/recognition/hypotheses_verification.h>
#include <v4r/recognition/object_hypothesis.h>

#include <pcl/visualization/pcl_visualizer.h>

namespace v4r {

template <typename PointT>
class HypothesisVerification;

template <typename PointT>
class  HV_ModelVisualizer {
 private:
  friend class HypothesisVerification<PointT>;

  using PointTWithNormal = v4r::add_normal_t<PointT>;

  int vp_model_scene_, vp_model_, vp_model_scene_overlay_, vp_model_outliers_, vp_model_scene_fit_, vp_model_visible_,
      vp_model_total_fit_, vp_model_3d_fit_, vp_model_color_fit_, vp_model_normals_fit_, vp_scene_normals_,
      vp_model_normals_, vp_scene_curvature_, vp_model_curvature_;

  pcl::visualization::PCLVisualizer::Ptr vis_model_;

  PCLVisualizationParams::ConstPtr vis_param_;

 public:
  explicit HV_ModelVisualizer(
      const PCLVisualizationParams::ConstPtr &vis_params = std::make_shared<PCLVisualizationParams>())
  : vis_param_(vis_params) {}

  void visualize(const HypothesisVerification<PointT> *hv, const HVRecognitionModel<PointT> &rm);

  typedef std::shared_ptr<HV_ModelVisualizer<PointT>> Ptr;
  typedef std::shared_ptr<HV_ModelVisualizer<PointT> const> ConstPtr;
};

template <typename PointT>
class  HV_CuesVisualizer {
 private:
  using PointTWithNormal = v4r::add_normal_t<PointT>;

  int vp_scene_scene_, vp_scene_active_hypotheses_, vp_model_fitness_, vp_model_scene_color_dist_, vp_scene_fitness_,
      vp_scene_duplicity_, vp_scene_smooth_regions_;

  pcl::visualization::PCLVisualizer::Ptr vis_go_cues_;

  PCLVisualizationParams::ConstPtr vis_param_;

  void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event,
                             const HypothesisVerification<PointT> *hv) const;

  boost::dynamic_bitset<> active_solution_;  ///< for keyboard callback

 public:
  explicit HV_CuesVisualizer(
      const PCLVisualizationParams::ConstPtr &vis_params = std::make_shared<PCLVisualizationParams>())
  : vis_param_(vis_params) {}

  void visualize(const HypothesisVerification<PointT> *hv, const boost::dynamic_bitset<> &active_solution, float cost);

  typedef std::shared_ptr<HV_CuesVisualizer<PointT>> Ptr;
  typedef std::shared_ptr<HV_CuesVisualizer<PointT> const> ConstPtr;
};

template <typename PointT>
class  HV_PairwiseVisualizer {
 private:
  using PointTWithNormal = v4r::add_normal_t<PointT>;

  pcl::visualization::PCLVisualizer::Ptr vis_pairwise_;
  int vp_pair_1_, vp_pair_2_, vp_pair_3_;

  PCLVisualizationParams::ConstPtr vis_param_;

 public:
  explicit HV_PairwiseVisualizer(
      const PCLVisualizationParams::ConstPtr &vis_params = std::make_shared<PCLVisualizationParams>())
  : vis_param_(vis_params) {}

  void visualize(const HypothesisVerification<PointT> *hv);

  typedef std::shared_ptr<HV_PairwiseVisualizer<PointT>> Ptr;
  typedef std::shared_ptr<HV_PairwiseVisualizer<PointT> const> ConstPtr;
};
}  // namespace v4r
