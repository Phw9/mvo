#include "map_data.h"

#include "bundle_adjustment.h"
#include "config.h"
#include "converter.h"
#include "feature.h"
#include "init.h"
#include "visualization.h"

#include <calib3d/multiview.h>

#include <iostream>
#include <string>

namespace mvo {

cv::Point3f camera_point_to_world(const cv::Point3f& point,
                                  const Pose& pose) {
    const double px = static_cast<double>(point.x) - pose.t[0];
    const double py = static_cast<double>(point.y) - pose.t[1];
    const double pz = static_cast<double>(point.z) - pose.t[2];
    const cv::Point3f world_point(
        static_cast<float>(pose.r[0] * px + pose.r[3] * py +
                           pose.r[6] * pz),
        static_cast<float>(pose.r[1] * px + pose.r[4] * py +
                           pose.r[7] * pz),
        static_cast<float>(pose.r[2] * px + pose.r[5] * py +
                           pose.r[8] * pz));
    return world_point;
}

void camera_points_to_world(const std::vector<cv::Point3f>& camera_points,
                            const Pose& pose,
                            std::vector<cv::Point3f>* world_points) {
    world_points->clear();
    world_points->reserve(camera_points.size());
    for (const cv::Point3f& point : camera_points) {
        world_points->push_back(camera_point_to_world(point, pose));
    }
}

MapPoint make_map_point(const cv::Point3f& position,
                        int32_t frame_id,
                        int32_t track_length,
                        double reprojection_error,
                        const MappingParameters& parameters,
                        const cv::Mat& descriptor) {
    MapPoint point;
    point.position = position;
    if (!descriptor.empty()) {
        point.descriptor = descriptor.clone();
    }
    point.created_frame = frame_id;
    point.last_seen_frame = frame_id;
    point.age = 0;
    point.track_length = track_length;
    point.last_reprojection_error = reprojection_error;
    point.candidate = track_length < parameters.candidate_min_track_length;
    return point;
}

std::vector<cv::Point3f> map_point_positions(
    const std::vector<MapPoint>& map_points) {
    std::vector<cv::Point3f> positions;
    positions.reserve(map_points.size());
    for (const MapPoint& point : map_points) {
        positions.push_back(point.position);
    }
    return positions;
}

Pose compose_reference_relative_pose(const Pose& reference_pose,
                                     const Pose& relative_pose) {
    Pose pose;
    for (int32_t row = 0; row < 3; ++row) {
        for (int32_t col = 0; col < 3; ++col) {
            double value = 0.0;
            for (int32_t k = 0; k < 3; ++k) {
                value += relative_pose.r[row * 3 + k] *
                         reference_pose.r[k * 3 + col];
            }
            pose.r[row * 3 + col] = value;
        }
    }
    for (int32_t row = 0; row < 3; ++row) {
        double value = relative_pose.t[row];
        for (int32_t k = 0; k < 3; ++k) {
            value += relative_pose.r[row * 3 + k] * reference_pose.t[k];
        }
        pose.t[row] = value;
    }
    return pose;
}

bool recover_two_view_from_reference(const cv::Mat& reference_image,
                                     const cv::Mat& image,
                                     const Pose& reference_pose,
                                     const CameraIntrinsics& camera,
                                     const MvoParameters& parameters,
                                     bool run_ba,
                                     bool debug_geometry,
                                     int32_t frame_id,
                                     TrackState* state) {
    bool ok = false;
    std::vector<cv::Point2f> initial_points;
    std::vector<cv::Point2f> tracked_reference;
    std::vector<cv::Point2f> tracked_current;
    std::vector<cv::Mat> tracked_descriptors;
    TrackState recovered;
    set_identity_pose(&recovered.prev_pose);
    set_identity_pose(&recovered.last_pose);
    if (detect_initial_points(reference_image, parameters.feature,
                              &initial_points)) {
        ok = track_points(reference_image, image, initial_points,
                          parameters.feature,
                          &tracked_reference, &tracked_current, nullptr,
                          debug_geometry,
                          "reinit_" + std::to_string(frame_id), false,
                          &tracked_descriptors);
    }
    if (ok) {
        ok = initialize_two_view(tracked_reference, tracked_current,
                                 &tracked_descriptors, camera, parameters,
                                 run_ba, debug_geometry, &recovered);
    }
    if (ok) {
        std::vector<cv::Point3f> world_points;
        camera_points_to_world(map_point_positions(recovered.map_points),
                               reference_pose,
                               &world_points);
        const Pose world_pose = compose_reference_relative_pose(
            reference_pose, recovered.last_pose);
        state->prev_image = image;
        state->prev_pose = world_pose;
        state->last_pose = world_pose;
        state->prev_points = recovered.prev_points;
        state->map_points.clear();
        state->map_points.reserve(world_points.size());
        for (int32_t i = 0; i < static_cast<int32_t>(world_points.size());
             ++i) {
            const cv::Mat descriptor =
                i < static_cast<int32_t>(recovered.map_points.size())
                    ? recovered.map_points[static_cast<std::size_t>(i)]
                          .descriptor
                    : cv::Mat();
            state->map_points.push_back(make_map_point(
                world_points[static_cast<std::size_t>(i)], frame_id, 2, 0.0,
                parameters.mapping, descriptor));
        }
        state->all_map_points.insert(state->all_map_points.end(),
                                     world_points.begin(),
                                     world_points.end());
        ++state->keyframes;
        std::cout << "tracking_reinitialized frame=" << frame_id
                  << " tracks=" << state->prev_points.size()
                  << " map_points=" << state->map_points.size()
                  << std::endl;
    } else if (debug_geometry) {
        std::cout << "tracking_reinit_failed frame=" << frame_id
                  << std::endl;
    }
    return ok;
}

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
                           bool aggressive_refresh,
                           std::vector<cv::Point2f>* current_points,
                           std::vector<MapPoint>* map_points,
                           std::vector<cv::Point3f>* all_map_points) {
    int32_t added = 0;
    const int32_t target_points = aggressive_refresh
                                      ? parameters.mapping
                                            .aggressive_target_tracked_map_points
                                      : parameters.mapping
                                            .target_tracked_map_points;
    const int32_t max_candidates = aggressive_refresh
                                       ? parameters.mapping
                                             .aggressive_max_refresh_candidates
                                       : parameters.mapping
                                             .max_refresh_candidates;
    const int32_t need = target_points -
                         static_cast<int32_t>(current_points->size());
    std::vector<cv::Point2f> candidates_prev;
    if (need > 0 && detect_refresh_points(prev_image, prev_existing,
                                          max_candidates,
                                          parameters.feature,
                                          &candidates_prev)) {
        std::vector<cv::Point2f> tracked_prev;
        std::vector<cv::Point2f> tracked_current;
        std::vector<cv::Mat> tracked_descriptors;
        track_points(prev_image, image, candidates_prev, parameters.feature,
                     &tracked_prev,
                     &tracked_current, nullptr, debug_geometry,
                     "refresh_" + std::to_string(frame_id), true,
                     &tracked_descriptors);

        std::vector<cv::Point2f> filtered_prev;
        std::vector<cv::Point2f> filtered_current;
        std::vector<cv::Mat> filtered_descriptors;
        std::vector<cv::Point2f> occupied = *current_points;
        const double refresh_min_distance =
            parameters.feature.frontend_mode == 1
                ? parameters.feature.orb_min_distance
                : parameters.feature.klt_min_distance;
        for (int32_t i = 0; i < static_cast<int32_t>(tracked_current.size());
            ++i) {
            if (point_is_far_from_existing(tracked_current[i], occupied,
                                           refresh_min_distance)) {
                filtered_prev.push_back(tracked_prev[i]);
                filtered_current.push_back(tracked_current[i]);
                if (i < static_cast<int32_t>(tracked_descriptors.size())) {
                    filtered_descriptors.push_back(
                        tracked_descriptors[static_cast<std::size_t>(i)]);
                } else {
                    filtered_descriptors.push_back(cv::Mat());
                }
                occupied.push_back(tracked_current[i]);
            }
        }

        const int32_t min_refresh_points =
            parameters.feature.frontend_mode == 1
                ? parameters.mapping.orb_min_refresh_map_points
                : parameters.mapping.min_refresh_map_points;
        if (static_cast<int32_t>(filtered_current.size()) >=
            min_refresh_points) {
            cvlib::Matrix p0 = pose_to_projection(prev_pose);
            cvlib::Matrix p1 = pose_to_projection(*current_pose);
            cvlib::Matrix norm0 = points2f_to_normalized_matrix(
                filtered_prev, camera);
            cvlib::Matrix norm1 = points2f_to_normalized_matrix(
                filtered_current, camera);
            cvlib::Matrix points3d = cvlib::matrix_create(norm0.rows, 3);
            const cvlib::ErrorCode ec = cvlib::calib3d::triangulate_points(
                &p0, &p1, &norm0, &norm1, &points3d);
            if (ec == cvlib::ErrorCode::kSuccess) {
                std::vector<cv::Point3f> accepted_points;
                std::vector<cv::Point2f> accepted_prev;
                std::vector<cv::Point2f> accepted_current;
                std::vector<cv::Mat> accepted_descriptors;
                const double triangulation_threshold =
                    parameters.feature.frontend_mode == 1
                        ? parameters.mapping.orb_max_triangulation_p90
                        : parameters.mapping.max_triangulation_p90;
                accepted_points.reserve(static_cast<std::size_t>(need));
                accepted_prev.reserve(static_cast<std::size_t>(need));
                accepted_current.reserve(static_cast<std::size_t>(need));
                accepted_descriptors.reserve(static_cast<std::size_t>(need));
                for (int32_t i = 0;
                     i < points3d.rows &&
                     static_cast<int32_t>(accepted_points.size()) < need;
                     ++i) {
                    const cv::Point3f point3d(
                        static_cast<float>(
                            cvlib::matrix_get(&points3d, i, 0)),
                        static_cast<float>(
                            cvlib::matrix_get(&points3d, i, 1)),
                        static_cast<float>(
                            cvlib::matrix_get(&points3d, i, 2)));
                    const double z0 = depth_in_pose(point3d, prev_pose);
                    const double z1 = depth_in_pose(point3d, *current_pose);
                    const double reproj0 = reprojection_residual(
                        point3d, filtered_prev[i], prev_pose, camera);
                    const double reproj1 = reprojection_residual(
                        point3d, filtered_current[i], *current_pose, camera);
                    const double parallax = parallax_deg_for_point(
                        point3d, prev_pose, *current_pose);
                    if (z0 > 1.0e-6 && z1 > 1.0e-6 &&
                        reproj0 <= triangulation_threshold &&
                        reproj1 <= triangulation_threshold &&
                        parallax >=
                            parameters.mapping.min_refresh_parallax_deg) {
                        accepted_points.push_back(point3d);
                        accepted_prev.push_back(filtered_prev[i]);
                        accepted_current.push_back(filtered_current[i]);
                        accepted_descriptors.push_back(
                            filtered_descriptors[static_cast<std::size_t>(i)]);
                    }
                }
                if (run_ba &&
                    static_cast<int32_t>(accepted_points.size()) >=
                        parameters.bundle_adjustment.min_points) {
                    run_two_view_bundle_adjustment(
                        prev_pose, current_pose, camera, accepted_prev,
                        accepted_current, &accepted_points,
                        parameters.bundle_adjustment,
                        "refresh_" + std::to_string(frame_id),
                        debug_geometry);
                }
                for (int32_t i = 0;
                     i < static_cast<int32_t>(accepted_points.size());
                     ++i) {
                    const std::size_t index = static_cast<std::size_t>(i);
                    const double residual = reprojection_residual(
                        accepted_points[index], accepted_current[index],
                        *current_pose, camera);
                    const int32_t initial_track_length =
                        parameters.feature.frontend_mode == 1 ? 2 : 1;
                    map_points->push_back(make_map_point(
                        accepted_points[index], frame_id,
                        initial_track_length, residual, parameters.mapping,
                        accepted_descriptors[index]));
                    current_points->push_back(
                        accepted_current[index]);
                    all_map_points->push_back(
                        accepted_points[index]);
                    ++added;
                }
            }
            cvlib::matrix_destroy(&p0);
            cvlib::matrix_destroy(&p1);
            cvlib::matrix_destroy(&norm0);
            cvlib::matrix_destroy(&norm1);
            cvlib::matrix_destroy(&points3d);
        }
    }
    if (debug_geometry) {
        std::cout << "map_refresh frame=" << frame_id
                  << " before=" << (current_points->size() - added)
                  << " added=" << added
                  << " after=" << current_points->size()
                  << " aggressive=" << aggressive_refresh
                  << std::endl;
    }
    return added;
}

}  // namespace mvo
