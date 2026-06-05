#pragma once

#include "mvo/parameters.h"
#include "mvo/types.h"

#include <opencv2/core.hpp>

#include <vector>

namespace mvo {

bool run_pnp(std::vector<cv::Point3f>* map_points,
             std::vector<cv::Point2f>* image_points,
             const CameraIntrinsics& camera,
             const Pose& initial_pose,
             const PnpParameters& parameters,
             bool debug_geometry,
             Pose* pose);

}  // namespace mvo
