#pragma once

#include "parameters.h"
#include "types.h"

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>

namespace mvo {

void camera_points_to_world(const std::vector<cv::Point3f>& camera_points,
                            const Pose& pose,
                            std::vector<cv::Point3f>* world_points);
Pose compose_reference_relative_pose(const Pose& reference_pose,
                                     const Pose& relative_pose);
bool recover_two_view_from_reference(const cv::Mat& reference_image,
                                     const cv::Mat& image,
                                     const Pose& reference_pose,
                                     const CameraIntrinsics& camera,
                                     const MvoParameters& parameters,
                                     bool run_ba,
                                     bool debug_geometry,
                                     int32_t frame_id,
                                     TrackState* state);
int32_t refresh_map_points(const cv::Mat& prev_image,
                           const cv::Mat& image,
                           const std::vector<cv::Point2f>& prev_existing,
                           const Pose& prev_pose,
                           Pose* current_pose,
                           const CameraIntrinsics& camera,
                           const MvoParameters& parameters,
                           int32_t frame_id,
                           bool run_ba,
                           bool debug_geometry,
                           std::vector<cv::Point2f>* current_points,
                           std::vector<cv::Point3f>* map_points,
                           std::vector<cv::Point3f>* all_map_points);

}  // namespace mvo
