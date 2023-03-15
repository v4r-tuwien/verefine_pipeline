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

#include <ppf/correspondence_finder.h>
#include <ppf/hough_voting.h>

namespace ppf {

template <typename PointT>
Correspondence::Vector CorrespondenceFinder<PointT>::find(uint32_t scene_index) const {

  Correspondence::Vector correspondences;
  HoughVoting hv(model_search_->getNumAnchorPoints(), num_angular_bins_);

  const auto& p1 = scene_->at(scene_index).getVector3fMap();
  const auto& n1 = scene_->at(scene_index).getNormalVector3fMap();

  Eigen::Affine3f transform_sg;
  LocalCoordinate::computeTransform(p1, n1, transform_sg);

  std::vector<int> indices;
  std::vector<float> sqr_distances;
  scene_search_tree_->radiusSearch(scene_->at(scene_index), model_search_->getModelDiameter(), indices, sqr_distances);

  // Start from second output index because the first one is always the query point itself.
  for (size_t i = 1; i < indices.size(); ++i) {

    const auto& index = indices[i];
    const auto& p2 = scene_->at(index).getVector3fMap();
    const auto& n2 = scene_->at(index).getNormalVector3fMap();

    // We have a pair of scene points:
    //  1. at scene_index - this is our reference point
    //  2. at index - this is our second point
    // We proceed to query for similar pairs on the model. Each of them will give us a "partial" local coordinate on the
    // model. To turn them into "full" LCs we need alpha_s. Then we send LCs to the Hough Voting scheme to find the most
    // popular LCs.

    // Compute alpha_s angle (see [VLLM18], page 6).
    auto alpha_s = LocalCoordinate::computeAngle(transform_sg * p2);

    const auto& lcs = model_search_->find(p1, n1, p2, n2);

    for (const auto& lc : lcs)
      hv.castVote({lc.model_point_index, lc.rotation_angle - alpha_s});
  }

  HoughVoting::Peak::Vector peaks;
  if (max_correspondences_ == 1) {
    // getPeak() is faster than extractPeaks(1) because it does not modify the accumulator
    peaks = {hv.getPeak()};
    // It may happen that it does not have enough votes, in which case we return empty list of correspondences.
    if (peaks.back().votes < min_votes_)
      return {};
  } else
    peaks = hv.extractPeaks(max_correspondences_, min_votes_);

  correspondences.reserve(peaks.size());
  for (const auto& peak : peaks) {
    Eigen::Affine3f transform;
    model_search_->computeTransform(peak.lc, transform);
    transform = transform_sg.inverse() * transform;
    correspondences.emplace_back(scene_index, peak.lc.model_point_index, peak.votes, transform);
  }

  return correspondences;
}

}  // namespace ppf
