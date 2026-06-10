#include "app.h"

#include "converter.h"
#include "feature.h"
#include "frame_source.h"
#include "init.h"
#include "loop_closure.h"
#include "map_data.h"
#include "pose_estimation.h"
#include "visualization.h"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace mvo {
namespace {

bool should_use_for_pnp(const MapPoint& point,
                        int32_t frame_id,
                        const MappingParameters& parameters) {
    const int32_t unseen_frames = frame_id - point.last_seen_frame;
    return point.has_position &&
           unseen_frames <= parameters.max_unseen_frames &&
           point.track_length >= parameters.min_pnp_track_length &&
           point.last_reprojection_error <=
               parameters.max_point_reprojection_error;
}

void update_tracked_map_point(MapPoint* point,
                              int32_t frame_id,
                              double reprojection_error,
                              const MappingParameters& parameters) {
    ++point->age;
    ++point->track_length;
    point->last_seen_frame = frame_id;
    point->last_reprojection_error = reprojection_error;
    if (point->track_length >= parameters.candidate_min_track_length) {
        point->candidate = false;
    }
}

void update_pending_map_point(MapPoint* point,
                              int32_t frame_id,
                              const MappingParameters& parameters) {
    ++point->age;
    ++point->track_length;
    point->last_seen_frame = frame_id;
    point->candidate = point->track_length <
                       parameters.candidate_min_track_length;
}

const char* frontend_name(const FeatureParameters& parameters) {
    const char* name = "klt";
    if (parameters.frontend_mode == 1) {
        name = "superpoint_superglue";
    }
    return name;
}

}  // namespace

int32_t run_app(AppConfig config) {
    int32_t exit_code = 1;
    FrameSource source;
    Visualizer visualizer;
    BowDatabase bow_db;
    TrackState state;
    set_identity_pose(&state.prev_pose);
    set_identity_pose(&state.last_pose);

    const bool calib_loaded = load_kitti_calibration(
        config.calib_path, &config.camera);

    const bool source_ok = open_frame_source(config, &source);
    const bool visualizer_ok = initialize_visualizer(config, &visualizer);

    std::cout << "input_type=" << input_type_name(config.input_type)
              << " input_path=" << config.input_path
              << " parameter_dir=" << config.parameter_dir
              << " frames_available=" << source.total_frames
              << " max_frames=" << config.max_frames
              << " no_gui=" << config.no_gui
              << " run_ba=" << config.run_ba << std::endl;
    std::cout << "intrinsics source="
              << (calib_loaded ? config.calib_path : "default")
              << " fx=" << config.camera.fx
              << " fy=" << config.camera.fy
              << " cx=" << config.camera.cx
              << " cy=" << config.camera.cy << std::endl;
    std::cout << "parameters_summary feature_min_init="
              << config.parameters.feature.min_init_tracks
              << " frontend_mode=" << config.parameters.feature.frontend_mode
              << " frontend=" << frontend_name(config.parameters.feature)
              << " pnp_min_tracks=" << config.parameters.pnp.min_tracks
              << " mapping_target="
              << config.parameters.mapping.target_tracked_map_points
              << " map_point_reproj_max="
              << config.parameters.mapping.max_point_reprojection_error
              << " ba_max_points="
              << config.parameters.bundle_adjustment.max_points
              << std::endl;

    if (load_vocabulary(config.vocabulary, &bow_db)) {
        std::cout << "vocabulary=loaded words=" << bow_db.vocabulary.size()
                  << std::endl;
    } else {
        std::cout << "vocabulary=missing_or_empty path=" << config.vocabulary
                  << std::endl;
    }

    if (source_ok && visualizer_ok) {
        cv::Mat image0;
        cv::Mat image1;
        std::vector<cv::Point2f> initial_points;
        std::vector<cv::Point2f> tracked0;
        std::vector<cv::Point2f> tracked1;
        std::vector<cv::Mat> tracked1_descriptors;

        bool ok = read_next_frame(&source, &image0);
        if (ok) {
            ok = read_next_frame(&source, &image1);
        }
        if (ok) {
            ok = detect_initial_points(image0, config.parameters.feature,
                                       &initial_points);
        }
        if (ok) {
            ok = track_points(image0, image1, initial_points,
                              config.parameters.feature, &tracked0,
                              &tracked1, nullptr, config.debug_geometry,
                              "init", false, &tracked1_descriptors);
        }
        if (ok) {
            ok = initialize_two_view(tracked0, tracked1,
                                     &tracked1_descriptors, config.camera,
                                     config.parameters, config.run_ba,
                                     config.debug_geometry, &state);
        }
        if (ok) {
            state.prev_image = image1;
            state.prev_pose = state.last_pose;
            state.frames_processed = 2;
            state.all_map_points = map_point_positions(state.map_points);
            Pose pose0;
            set_identity_pose(&pose0);
            log_visualization(&visualizer, 0, image0, tracked0,
                              map_point_positions(state.map_points),
                              state.all_map_points, pose0);
            log_visualization(&visualizer, 1, image1, state.prev_points,
                              map_point_positions(state.map_points),
                              state.all_map_points,
                              state.last_pose);
            query_and_add_loop(image0, &bow_db, 0,
                               config.parameters.loop_closure);
            query_and_add_loop(image1, &bow_db, 1,
                               config.parameters.loop_closure);
            state.loop_queries = static_cast<int32_t>(bow_db.bows.size());

            const int32_t frame_limit = std::min(
                config.max_frames, source.total_frames);
            for (int32_t frame_id = 2; frame_id < frame_limit; ++frame_id) {
                cv::Mat image;
                std::vector<cv::Point2f> raw_tracked_next;
                std::vector<cv::Point2f> tracked_next;
                std::vector<MapPoint> next_map_points;
                const bool frame_ok = read_next_frame(&source, &image);
                if (state.prev_points.size() != state.map_points.size()) {
                    const std::size_t aligned_count = std::min(
                        state.prev_points.size(), state.map_points.size());
                    state.prev_points.resize(aligned_count);
                    state.map_points.resize(aligned_count);
                }
                if (!frame_ok) {
                    std::cout << "frame_read_failed frame=" << frame_id
                              << std::endl;
                    break;
                } else if (!state.prev_points.empty()) {
                    std::vector<cv::Point2f> tracked_prev;
                    std::vector<int32_t> tracked_indices;
                    const Pose previous_pose = state.prev_pose;
                    bool pnp_ok = false;
                    bool recovered_ok = false;
                    track_points(state.prev_image, image, state.prev_points,
                                 config.parameters.feature, &tracked_prev,
                                 &raw_tracked_next, &tracked_indices,
                                 config.debug_geometry,
                                 "frame_" + std::to_string(frame_id), true);
                    const int32_t tracked_count = std::min(
                        static_cast<int32_t>(raw_tracked_next.size()),
                        static_cast<int32_t>(tracked_indices.size()));
                    for (int32_t i = 0; i < tracked_count; ++i) {
                        tracked_next.push_back(raw_tracked_next[i]);
                        MapPoint point =
                            state.map_points[static_cast<std::size_t>(
                                tracked_indices[static_cast<std::size_t>(i)])];
                        next_map_points.push_back(point);
                    }
                    std::vector<cv::Point3f> pnp_map_points;
                    std::vector<cv::Point2f> pnp_image_points;
                    for (int32_t i = 0;
                         i < static_cast<int32_t>(next_map_points.size());
                         ++i) {
                        const bool use_for_pnp = should_use_for_pnp(
                            next_map_points[static_cast<std::size_t>(i)],
                            frame_id, config.parameters.mapping);
                        if (use_for_pnp) {
                            pnp_map_points.push_back(
                                next_map_points[static_cast<std::size_t>(i)]
                                    .position);
                            pnp_image_points.push_back(
                                tracked_next[static_cast<std::size_t>(i)]);
                        }
                    }
                    if (static_cast<int32_t>(pnp_image_points.size()) >=
                        config.parameters.pnp.min_tracks) {
                        const Pose initial_pose = state.last_pose;
                        if (run_pnp(&pnp_map_points, &pnp_image_points,
                                    config.camera, initial_pose,
                                    config.parameters.pnp,
                                    config.debug_geometry,
                                    &state.last_pose)) {
                            ++state.pnp_success;
                            pnp_ok = true;
                        }
                    } else {
                        std::cout << "pnp_skipped tracks="
                                  << pnp_image_points.size()
                                  << " tracked=" << tracked_next.size()
                                  << std::endl;
                    }
                    if (pnp_ok) {
                        std::vector<cv::Point2f> inlier_points;
                        std::vector<MapPoint> inlier_map_points;
                        inlier_points.reserve(tracked_next.size());
                        inlier_map_points.reserve(next_map_points.size());
                        for (int32_t i = 0;
                             i < static_cast<int32_t>(next_map_points.size());
                             ++i) {
                            MapPoint point =
                                next_map_points[static_cast<std::size_t>(i)];
                            const cv::Point2f& observation =
                                tracked_next[static_cast<std::size_t>(i)];
                            if (point.has_position) {
                                const double residual = reprojection_residual(
                                    point.position, observation,
                                    state.last_pose, config.camera);
                                if (std::isfinite(residual) &&
                                    residual <=
                                        config.parameters.pnp
                                            .reprojection_inlier_threshold) {
                                    update_tracked_map_point(
                                        &point, frame_id, residual,
                                        config.parameters.mapping);
                                    inlier_points.push_back(observation);
                                    inlier_map_points.push_back(point);
                                }
                            } else if (point.has_anchor &&
                                       point.age <
                                           config.parameters.mapping
                                               .max_active_age) {
                                update_pending_map_point(
                                    &point, frame_id,
                                    config.parameters.mapping);
                                inlier_points.push_back(observation);
                                inlier_map_points.push_back(point);
                            }
                        }
                        tracked_next = inlier_points;
                        next_map_points = inlier_map_points;
                        triangulate_pending_map_points(
                            tracked_next, state.last_pose, config.camera,
                            config.parameters, frame_id,
                            config.debug_geometry, &next_map_points,
                            &state.all_map_points);
                        add_pending_feature_tracks(
                            image, state.last_pose, frame_id,
                            config.parameters, config.debug_geometry,
                            &tracked_next, &next_map_points);
                        if (config.debug_geometry) {
                            std::cout << "lifecycle frame=" << frame_id
                                      << " active=" << tracked_next.size()
                                      << " pnp_candidates="
                                      << pnp_image_points.size() << std::endl;
                        }
                    }
                    if (!pnp_ok) {
                        recovered_ok = recover_two_view_from_reference(
                            state.prev_image, image, previous_pose,
                            config.camera, config.parameters, config.run_ba,
                            config.debug_geometry, frame_id, &state);
                    }
                    if (pnp_ok && !recovered_ok) {
                        state.prev_image = image;
                        state.prev_pose = state.last_pose;
                        state.prev_points = tracked_next;
                        state.map_points = next_map_points;
                    } else if (!recovered_ok) {
                        std::cout << "frame_pose_failed frame=" << frame_id
                                  << " tracked=" << tracked_next.size()
                                  << " state_held=1" << std::endl;
                    }
                    query_and_add_loop(image, &bow_db, frame_id,
                                       config.parameters.loop_closure);
                    state.loop_queries =
                        static_cast<int32_t>(bow_db.bows.size());
                    ++state.frames_processed;
                    if (pnp_ok || recovered_ok) {
                        log_visualization(&visualizer, frame_id, image,
                                          state.prev_points,
                                          map_point_positions(state.map_points),
                                          state.all_map_points,
                                          state.last_pose);
                    } else {
                        const std::vector<cv::Point2f> empty_points;
                        const std::vector<cv::Point3f> empty_map_points;
                        log_visualization(&visualizer, frame_id, image,
                                          empty_points, empty_map_points,
                                          state.all_map_points,
                                          state.last_pose);
                    }
                    std::cout << "frame=" << frame_id
                              << " tracks=" << state.prev_points.size()
                              << " map_points="
                              << count_positioned_map_points(state.map_points)
                              << " pending="
                              << state.map_points.size() -
                                     static_cast<std::size_t>(
                                         count_positioned_map_points(
                                             state.map_points))
                              << std::endl;
                } else {
                    bool recovered_ok = false;
                    if (!state.prev_image.empty()) {
                        recovered_ok = recover_two_view_from_reference(
                            state.prev_image, image, state.prev_pose,
                            config.camera, config.parameters, config.run_ba,
                            config.debug_geometry, frame_id, &state);
                    }
                    if (!recovered_ok) {
                        state.prev_image = image;
                        state.prev_pose = state.last_pose;
                        state.prev_points.clear();
                        state.map_points.clear();
                    }
                    query_and_add_loop(image, &bow_db, frame_id,
                                       config.parameters.loop_closure);
                    state.loop_queries =
                        static_cast<int32_t>(bow_db.bows.size());
                    ++state.frames_processed;
                    log_visualization(
                        &visualizer, frame_id, image, state.prev_points,
                        map_point_positions(state.map_points),
                        state.all_map_points, state.last_pose);
                    std::cout << "frame=" << frame_id
                              << " tracks=" << state.prev_points.size()
                              << " map_points="
                              << count_positioned_map_points(state.map_points)
                              << " pending="
                              << state.map_points.size() -
                                     static_cast<std::size_t>(
                                         count_positioned_map_points(
                                             state.map_points))
                              << " recovered=" << recovered_ok << std::endl;
                }
            }
            exit_code = 0;
        }
    } else if (!source_ok) {
        std::cout << "input=open_failed type="
                  << input_type_name(config.input_type)
                  << " path=" << config.input_path << std::endl;
    }

    flush_visualizer(&visualizer);
    std::cout << "summary frames=" << state.frames_processed
              << " keyframes=" << state.keyframes
              << " pnp_success=" << state.pnp_success
              << " loop_queries=" << state.loop_queries
              << " map_points=" << count_positioned_map_points(
                     state.map_points)
              << " pending="
              << state.map_points.size() -
                     static_cast<std::size_t>(
                         count_positioned_map_points(state.map_points))
              << " exit_code=" << exit_code << std::endl;
    return exit_code;
}

}  // namespace mvo
