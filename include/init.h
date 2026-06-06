#pragma once

#include "parameters.h"
#include "types.h"

#include <opencv2/core.hpp>

#include <vector>

namespace mvo {

bool initialize_two_view(const std::vector<cv::Point2f>& points0,
                         const std::vector<cv::Point2f>& points1,
                         const std::vector<cv::Mat>* point1_descriptors,
                         const CameraIntrinsics& camera,
                         const MvoParameters& parameters,
                         bool run_ba,
                         bool debug_geometry,
                         TrackState* state);

}  // namespace mvo
