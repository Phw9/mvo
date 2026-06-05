#pragma once

#include "parameters.h"
#include "types.h"

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace mvo {

bool run_two_view_bundle_adjustment(
    const Pose& reference_pose,
    Pose* current_pose,
    const CameraIntrinsics& camera,
    const std::vector<cv::Point2f>& reference_points,
    const std::vector<cv::Point2f>& current_points,
    std::vector<cv::Point3f>* map_points,
    const BundleAdjustmentParameters& parameters,
    const std::string& tag,
    bool debug_geometry);

}  // namespace mvo
