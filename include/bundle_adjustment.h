#pragma once

#include "parameters.h"
#include "types.h"

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace mvo {

// Defined in loop_closure.h; only a pointer is needed here so the windowed
// local BA can propagate refined poses into the keyframes the trajectory dump
// reads, without pulling the DBoW2 headers into every consumer.
struct BowDatabase;

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

// Windowed local bundle adjustment over the most recent monocular poses in
// state->pose_window and the persistent map points their KLT tracks observe
// across the window. Mirrors the stereo local BA minus the stereo rows: the
// oldest surviving window camera is held fixed as the gauge anchor and the
// refined poses/points are written back into the pose window, the archive,
// state->map_points, and the matching bow_db keyframes.
bool run_mono_local_ba(const CameraIntrinsics& camera,
                       const MonoLocalBaParameters& parameters,
                       int32_t frame_id,
                       bool debug_geometry,
                       MapArchive* archive,
                       BowDatabase* bow_db,
                       TrackState* state);

}  // namespace mvo
