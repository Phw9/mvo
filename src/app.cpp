#include "mvo/app.h"

#include "mvo/feature.h"
#include "mvo/frame_source.h"
#include "mvo/init.h"
#include "mvo/loop_closure.h"
#include "mvo/map_data.h"
#include "mvo/pose_estimation.h"
#include "mvo/visualization.h"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

namespace mvo {

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
              << " pnp_min_tracks=" << config.parameters.pnp.min_tracks
              << " mapping_min_tracked="
              << config.parameters.mapping.min_tracked_map_points
              << " mapping_target="
              << config.parameters.mapping.target_tracked_map_points
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
                              "init", false);
        }
        if (ok) {
            ok = initialize_two_view(tracked0, tracked1, config.camera,
                                     config.parameters, config.run_ba,
                                     config.debug_geometry, &state);
        }
        if (ok) {
            state.prev_image = image1;
            state.prev_pose = state.last_pose;
            state.frames_processed = 2;
            state.all_map_points = state.map_points;
            Pose pose0;
            set_identity_pose(&pose0);
            log_visualization(&visualizer, 0, image0, tracked0,
                              state.map_points, state.all_map_points, pose0);
            log_visualization(&visualizer, 1, image1, state.prev_points,
                              state.map_points, state.all_map_points,
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
                std::vector<cv::Point3f> next_map_points;
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
                    std::vector<uchar> status;
                    std::vector<float> err;
                    std::vector<cv::Point2f> tracked_prev;
                    std::vector<int32_t> tracked_indices;
                    const Pose previous_pose = state.prev_pose;
                    bool pnp_ok = false;
                    bool recovered_ok = false;
                    track_points(state.prev_image, image, state.prev_points,
                                 config.parameters.feature,
                                 &tracked_prev, &raw_tracked_next,
                                 &tracked_indices,
                                 config.debug_geometry,
                                 "frame_" + std::to_string(frame_id), true);
                    for (int32_t i = 0;
                         i < static_cast<int32_t>(raw_tracked_next.size());
                         ++i) {
                        tracked_next.push_back(raw_tracked_next[i]);
                        next_map_points.push_back(
                            state.map_points[static_cast<std::size_t>(
                                tracked_indices[static_cast<std::size_t>(i)])]);
                    }
                    if (static_cast<int32_t>(tracked_next.size()) >=
                        config.parameters.pnp.min_tracks) {
                        const Pose initial_pose = state.last_pose;
                        if (run_pnp(&next_map_points, &tracked_next,
                                    config.camera, initial_pose,
                                    config.parameters.pnp,
                                    config.debug_geometry,
                                    &state.last_pose)) {
                            ++state.pnp_success;
                            pnp_ok = true;
                        }
                    } else {
                        std::cout << "pnp_skipped tracks="
                                  << tracked_next.size() << std::endl;
                    }
                    if (pnp_ok &&
                        static_cast<int32_t>(tracked_next.size()) <
                            config.parameters.mapping
                                .min_tracked_map_points) {
                        const int32_t added = refresh_map_points(
                            state.prev_image, image, state.prev_points,
                            previous_pose, &state.last_pose, config.camera,
                            config.parameters,
                            frame_id, config.run_ba, config.debug_geometry,
                            &tracked_next, &next_map_points,
                            &state.all_map_points);
                        if (added >=
                            config.parameters.mapping
                                .min_refresh_map_points) {
                            ++state.keyframes;
                        }
                    }
                    if (!pnp_ok) {
                        recovered_ok = recover_two_view_from_reference(
                            state.prev_image, image, previous_pose,
                            config.camera, config.parameters, config.run_ba,
                            config.debug_geometry, frame_id, &state);
                    }
                    if (pnp_ok) {
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
                                          state.prev_points, state.map_points,
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
                              << " map_points=" << state.map_points.size()
                              << std::endl;
                } else {
                    state.prev_image = image;
                    query_and_add_loop(image, &bow_db, frame_id,
                                       config.parameters.loop_closure);
                    state.loop_queries =
                        static_cast<int32_t>(bow_db.bows.size());
                    ++state.frames_processed;
                    log_visualization(&visualizer, frame_id, image,
                                      state.prev_points, state.map_points,
                                      state.all_map_points, state.last_pose);
                    std::cout << "frame=" << frame_id
                              << " tracks=0 map_points=0" << std::endl;
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
              << " map_points=" << state.map_points.size()
              << " exit_code=" << exit_code << std::endl;
    return exit_code;
}

}  // namespace mvo
