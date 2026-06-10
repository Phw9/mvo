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
MapPoint make_map_point(const cv::Point3f& position,
                        int32_t frame_id,
                        int32_t track_length,
                        double reprojection_error,
                        const MappingParameters& parameters,
                        const cv::Mat& descriptor = cv::Mat());
MapPoint make_pending_map_point(const cv::Point2f& observation,
                                const Pose& anchor_pose,
                                int32_t frame_id,
                                const MappingParameters& parameters);
int32_t count_positioned_map_points(const std::vector<MapPoint>& map_points);
std::vector<cv::Point3f> map_point_positions(
    const std::vector<MapPoint>& map_points);
int32_t add_pending_feature_tracks(const cv::Mat& image,
                                   const Pose& current_pose,
                                   int32_t frame_id,
                                   const MvoParameters& parameters,
                                   bool debug_geometry,
                                   std::vector<cv::Point2f>* current_points,
                                   std::vector<MapPoint>* map_points);
int32_t triangulate_pending_map_points(
    const std::vector<cv::Point2f>& current_points,
    const Pose& current_pose,
    const CameraIntrinsics& camera,
    const MvoParameters& parameters,
    int32_t frame_id,
    bool debug_geometry,
    std::vector<MapPoint>* map_points,
    std::vector<cv::Point3f>* all_map_points);
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

}  // namespace mvo
