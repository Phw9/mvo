#include "map_data.h"

#include "config.h"
#include "converter.h"
#include "feature.h"
#include "init.h"

#include <calib3d/multiview.h>

#include <algorithm>
#include <iostream>
#include <string>

namespace mvo {
namespace {

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

}  // namespace

void camera_points_to_world(const std::vector<cv::Point3f>& camera_points,
                            const Pose& pose,
                            std::vector<cv::Point3f>* world_points) {
    world_points->clear();
    world_points->reserve(camera_points.size());
    for (const cv::Point3f& point : camera_points) {
        world_points->push_back(camera_point_to_world(point, pose));
    }
}

namespace {

// Persistent map-point identity for the observation archive; the pipeline
// is single-threaded so a plain counter is safe.
int32_t allocate_map_point_id() {
    static int32_t next_map_point_id = 0;
    return next_map_point_id++;
}

}  // namespace

MapPoint make_map_point(const cv::Point3f& position,
                        int32_t frame_id,
                        int32_t track_length,
                        double reprojection_error,
                        const MappingParameters& parameters) {
    MapPoint point;
    point.id = allocate_map_point_id();
    point.position = position;
    point.created_frame = frame_id;
    point.last_seen_frame = frame_id;
    point.anchor_frame = frame_id;
    point.age = 0;
    point.track_length = track_length;
    point.last_reprojection_error = reprojection_error;
    point.has_position = true;
    point.has_anchor = false;
    point.candidate = track_length < parameters.candidate_min_track_length;
    return point;
}

MapPoint make_pending_map_point(const cv::Point2f& observation,
                                const Pose& anchor_pose,
                                int32_t frame_id,
                                const MappingParameters& parameters) {
    MapPoint point;
    point.id = allocate_map_point_id();
    point.anchor_observation = observation;
    point.anchor_pose = anchor_pose;
    point.created_frame = frame_id;
    point.last_seen_frame = frame_id;
    point.anchor_frame = frame_id;
    point.age = 0;
    point.track_length = 1;
    point.last_reprojection_error = 0.0;
    point.has_position = false;
    point.has_anchor = true;
    point.candidate = point.track_length < parameters.candidate_min_track_length;
    return point;
}

int32_t count_positioned_map_points(const std::vector<MapPoint>& map_points) {
    int32_t count = 0;
    for (const MapPoint& point : map_points) {
        if (point.has_position) {
            ++count;
        }
    }
    return count;
}

void archive_observation(MapArchive* archive,
                         const MapObservation& observation) {
    archive->observations_by_frame[observation.frame_id].push_back(
        static_cast<int32_t>(archive->observations.size()));
    archive->observations.push_back(observation);
}

void archive_stereo_observation(MapArchive* archive,
                                const StereoObservation& observation) {
    archive->stereo_observations_by_frame[observation.frame_id].push_back(
        static_cast<int32_t>(archive->stereo_observations.size()));
    archive->stereo_observations.push_back(observation);
}

std::vector<cv::Point3f> map_point_positions(
    const std::vector<MapPoint>& map_points) {
    std::vector<cv::Point3f> positions;
    positions.reserve(map_points.size());
    for (const MapPoint& point : map_points) {
        if (point.has_position) {
            positions.push_back(point.position);
        }
    }
    return positions;
}

int32_t add_pending_feature_tracks(const cv::Mat& image,
                                   const Pose& current_pose,
                                   int32_t frame_id,
                                   const MvoParameters& parameters,
                                   bool debug_geometry,
                                   std::vector<cv::Point2f>* current_points,
                                   std::vector<MapPoint>* map_points) {
    int32_t added = 0;
    const int32_t target_points = std::min(
        parameters.mapping.target_tracked_map_points,
        parameters.feature.max_features);
    const int32_t need = target_points -
                         static_cast<int32_t>(current_points->size());
    const int32_t max_new_points = std::min(
        need, parameters.mapping.max_refresh_candidates);
    std::vector<cv::Point2f> new_points;
    if (max_new_points > 0 &&
        detect_refresh_points(image, *current_points, max_new_points,
                              parameters.feature, &new_points)) {
        for (const cv::Point2f& point : new_points) {
            current_points->push_back(point);
            map_points->push_back(make_pending_map_point(
                point, current_pose, frame_id, parameters.mapping));
            ++added;
        }
    }
    if (debug_geometry) {
        std::cout << "track_replenish frame=" << frame_id
                  << " before=" << (current_points->size() - added)
                  << " added=" << added
                  << " after=" << current_points->size()
                  << " target=" << target_points << std::endl;
    }
    return added;
}

int32_t triangulate_pending_map_points(
    const std::vector<cv::Point2f>& current_points,
    const Pose& current_pose,
    const CameraIntrinsics& camera,
    const MvoParameters& parameters,
    int32_t frame_id,
    bool debug_geometry,
    std::vector<MapPoint>* map_points,
    std::vector<cv::Point3f>* all_map_points) {
    int32_t added = 0;
    const int32_t aligned_count = static_cast<int32_t>(
        std::min(current_points.size(), map_points->size()));
    const double triangulation_threshold =
        parameters.mapping.max_triangulation_p90;
    for (int32_t i = 0; i < aligned_count; ++i) {
        MapPoint& point = (*map_points)[static_cast<std::size_t>(i)];
        if (!point.has_position && point.has_anchor &&
            point.track_length >= 2) {
            const std::vector<cv::Point2f> anchor_points = {
                point.anchor_observation};
            const std::vector<cv::Point2f> tracked_points = {
                current_points[static_cast<std::size_t>(i)]};
            cvlib::Matrix p0 = pose_to_projection(point.anchor_pose);
            cvlib::Matrix p1 = pose_to_projection(current_pose);
            cvlib::Matrix norm0 = points2f_to_normalized_matrix(
                anchor_points, camera);
            cvlib::Matrix norm1 = points2f_to_normalized_matrix(
                tracked_points, camera);
            cvlib::Matrix points3d = cvlib::matrix_create(1, 3);
            const cvlib::ErrorCode ec = cvlib::calib3d::triangulate_points(
                &p0, &p1, &norm0, &norm1, &points3d);
            if (ec == cvlib::ErrorCode::kSuccess) {
                const cv::Point3f point3d(
                    static_cast<float>(cvlib::matrix_get(&points3d, 0, 0)),
                    static_cast<float>(cvlib::matrix_get(&points3d, 0, 1)),
                    static_cast<float>(cvlib::matrix_get(&points3d, 0, 2)));
                const double z0 = depth_in_pose(point3d, point.anchor_pose);
                const double z1 = depth_in_pose(point3d, current_pose);
                const double reproj0 = reprojection_residual(
                    point3d, point.anchor_observation, point.anchor_pose,
                    camera);
                const double reproj1 = reprojection_residual(
                    point3d, current_points[static_cast<std::size_t>(i)],
                    current_pose, camera);
                const double parallax = parallax_deg_for_point(
                    point3d, point.anchor_pose, current_pose);
                if (z0 > 1.0e-6 && z1 > 1.0e-6 &&
                    reproj0 <= triangulation_threshold &&
                    reproj1 <= triangulation_threshold &&
                    parallax >= parameters.mapping.min_refresh_parallax_deg) {
                    point.position = point3d;
                    point.has_position = true;
                    point.has_anchor = false;
                    point.last_seen_frame = frame_id;
                    point.last_reprojection_error = reproj1;
                    point.candidate =
                        point.track_length <
                        parameters.mapping.candidate_min_track_length;
                    all_map_points->push_back(point3d);
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
        std::cout << "pending_triangulation frame=" << frame_id
                  << " added=" << added
                  << " active_3d=" << count_positioned_map_points(*map_points)
                  << " active_tracks=" << map_points->size()
                  << std::endl;
    }
    return added;
}

int32_t cull_weak_map_points(int32_t frame_id,
                             const MapCullParameters& parameters,
                             bool debug_geometry,
                             std::vector<cv::Point2f>* current_points,
                             std::vector<MapPoint>* map_points) {
    if (parameters.enabled == 0) {
        return 0;
    }
    std::vector<cv::Point2f> kept_points;
    std::vector<MapPoint> kept_map_points;
    const std::size_t aligned_count = std::min(current_points->size(),
                                               map_points->size());
    kept_points.reserve(aligned_count);
    kept_map_points.reserve(aligned_count);
    int32_t culled = 0;
    for (std::size_t i = 0; i < aligned_count; ++i) {
        MapPoint point = (*map_points)[i];
        bool keep = true;
        if (point.has_position) {
            // Refresh the consecutive-bad-frame streak from this frame's
            // reprojection error, then cull survivors that stayed weak past
            // the grace period.
            if (point.last_reprojection_error >
                parameters.mp_max_reprojection) {
                ++point.consecutive_bad_frames;
            } else {
                point.consecutive_bad_frames = 0;
            }
            const int32_t existed = frame_id - point.created_frame;
            if (existed >= parameters.mp_grace_frames) {
                const bool weak_observations =
                    point.track_length < parameters.mp_min_observations;
                const bool chronically_bad =
                    point.consecutive_bad_frames >=
                    parameters.mp_max_bad_frames;
                if (weak_observations || chronically_bad) {
                    keep = false;
                }
            }
        }
        if (keep) {
            kept_points.push_back((*current_points)[i]);
            kept_map_points.push_back(point);
        } else {
            ++culled;
        }
    }
    // Preserve any trailing tracks beyond the aligned prefix unchanged.
    for (std::size_t i = aligned_count; i < current_points->size(); ++i) {
        kept_points.push_back((*current_points)[i]);
    }
    for (std::size_t i = aligned_count; i < map_points->size(); ++i) {
        kept_map_points.push_back((*map_points)[i]);
    }
    *current_points = kept_points;
    *map_points = kept_map_points;
    if (debug_geometry) {
        std::cout << "map_cull frame=" << frame_id
                  << " culled=" << culled
                  << " active_3d=" << count_positioned_map_points(*map_points)
                  << " active_tracks=" << map_points->size()
                  << std::endl;
    }
    return culled;
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
    TrackState recovered;
    set_identity_pose(&recovered.prev_pose);
    set_identity_pose(&recovered.last_pose);
    if (detect_initial_points(reference_image, parameters.feature,
                              &initial_points)) {
        ok = track_points(reference_image, image, initial_points,
                          parameters.feature,
                          &tracked_reference, &tracked_current, nullptr,
                          debug_geometry,
                          "reinit_" + std::to_string(frame_id), false);
    }
    if (ok) {
        ok = initialize_two_view(tracked_reference, tracked_current,
                                 camera, parameters, run_ba, debug_geometry,
                                 &recovered);
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
        for (const cv::Point3f& world_point : world_points) {
            state->map_points.push_back(make_map_point(
                world_point, frame_id, 2, 0.0, parameters.mapping));
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

}  // namespace mvo
