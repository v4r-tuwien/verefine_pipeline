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

#pragma once

#include <pcl/point_cloud.h>

#include <boost/dynamic_bitset.hpp>

namespace v4r {
/**
 *
 * @tparam PointTA
 * @tparam PointTB
 * @param cloud_to_be_filtered cloud to be checked for occlusion
 * @param occlusion_cloud cloud that can cause occlusion
 * @param occlusion_threshold occlusion threshold in meter. If point in occlusion_cloud is closer to camera by more than
 * this threshold, pixel is considered occluded
 * @return pixel mask with visible points being marked true
 */
template <typename PointTA, typename PointTB>
boost::dynamic_bitset<> computeVisiblePoints(const pcl::PointCloud<PointTA> &cloud_to_be_filtered,
                                             const pcl::PointCloud<PointTB> &occlusion_cloud,
                                             float occlusion_threshold = 0.01f);
}  // namespace v4r
