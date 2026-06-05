#pragma once

#include "mvo/parameters.h"
#include "mvo/types.h"

#include <opencv2/core.hpp>

#include <vector>

namespace mvo {

bool initialize_two_view(const std::vector<cv::Point2f>& points0,
                         const std::vector<cv::Point2f>& points1,
                         const CameraIntrinsics& camera,
                         const MvoParameters& parameters,
                         bool run_ba,
                         bool debug_geometry,
                         TrackState* state);

}  // namespace mvo
