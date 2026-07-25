#include "app.h"

#include "bundle_adjustment.h"
#include "converter.h"
#include "feature.h"
#include "frame_source.h"
#include "init.h"
#include "loop_closure.h"
#include "map_data.h"
#include "pose_estimation.h"
#include "pose_graph.h"
#include "stereo.h"
#include "visualization.h"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace mvo {
namespace {

bool map_point_usable_for_pnp(const MapPoint& point,
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

// Extrapolates the next world->camera pose assuming the inter-frame motion is
// constant: predicted = M * last, where M is the motion from before_last to
// last (Rm = R_last * R_before^T, tm = t_last - Rm * t_before).
void predict_constant_velocity_pose(const Pose& before_last, const Pose& last,
                                    Pose* predicted) {
    double rm[9];
    for (int32_t r = 0; r < 3; ++r) {
        for (int32_t c = 0; c < 3; ++c) {
            double sum = 0.0;
            for (int32_t k = 0; k < 3; ++k) {
                sum += last.r[r * 3 + k] * before_last.r[c * 3 + k];
            }
            rm[r * 3 + c] = sum;
        }
    }
    double tm[3];
    for (int32_t r = 0; r < 3; ++r) {
        tm[r] = last.t[r] - (rm[r * 3 + 0] * before_last.t[0] +
                             rm[r * 3 + 1] * before_last.t[1] +
                             rm[r * 3 + 2] * before_last.t[2]);
    }
    for (int32_t r = 0; r < 3; ++r) {
        for (int32_t c = 0; c < 3; ++c) {
            double sum = 0.0;
            for (int32_t k = 0; k < 3; ++k) {
                sum += rm[r * 3 + k] * last.r[k * 3 + c];
            }
            predicted->r[r * 3 + c] = sum;
        }
        predicted->t[r] = rm[r * 3 + 0] * last.t[0] +
                          rm[r * 3 + 1] * last.t[1] +
                          rm[r * 3 + 2] * last.t[2] + tm[r];
    }
}

// Projects a world point with a pose and intrinsics; false if behind camera.
bool project_map_point(const cv::Point3f& world, const Pose& pose,
                       const CameraIntrinsics& camera, cv::Point2f* pixel) {
    const double x = static_cast<double>(world.x);
    const double y = static_cast<double>(world.y);
    const double z = static_cast<double>(world.z);
    const double xc = pose.r[0] * x + pose.r[1] * y + pose.r[2] * z + pose.t[0];
    const double yc = pose.r[3] * x + pose.r[4] * y + pose.r[5] * z + pose.t[1];
    const double zc = pose.r[6] * x + pose.r[7] * y + pose.r[8] * z + pose.t[2];
    if (zc <= 1.0e-6) {
        return false;
    }
    pixel->x = static_cast<float>(camera.fx * xc / zc + camera.cx);
    pixel->y = static_cast<float>(camera.fy * yc / zc + camera.cy);
    return true;
}

// Builds a per-point motion prediction (parallel to state->prev_points) from
// the constant-velocity pose, reprojecting positioned map points and falling
// back to the previous pixel otherwise. Returns null when no motion history.
const std::vector<cv::Point2f>* build_motion_prediction(
    const TrackState& state, const CameraIntrinsics& camera,
    std::vector<cv::Point2f>* prediction) {
    prediction->clear();
    if (!state.has_pose_before_last ||
        state.prev_points.size() != state.map_points.size() ||
        state.prev_points.empty()) {
        return nullptr;
    }
    Pose predicted_pose;
    predict_constant_velocity_pose(state.pose_before_last, state.prev_pose,
                                   &predicted_pose);
    prediction->resize(state.prev_points.size());
    for (std::size_t i = 0; i < state.prev_points.size(); ++i) {
        cv::Point2f pixel;
        if (state.map_points[i].has_position &&
            project_map_point(state.map_points[i].position, predicted_pose,
                              camera, &pixel)) {
            (*prediction)[i] = pixel;
        } else {
            (*prediction)[i] = state.prev_points[i];
        }
    }
    return prediction;
}

std::size_t count_pending_map_points(const std::vector<MapPoint>& map_points) {
    return map_points.size() -
           static_cast<std::size_t>(count_positioned_map_points(map_points));
}

void log_startup(const AppConfig& config, const FrameSource& source,
                 bool calib_loaded) {
    std::cout << "input_type=" << input_type_name(config.input_type)
              << " input_path=" << config.input_path
              << " parameter_dir=" << config.parameter_dir
              << " frames_available=" << source.total_frames
              << " max_frames=" << config.max_frames
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
              << " mapping_target="
              << config.parameters.mapping.target_tracked_map_points
              << " map_point_reproj_max="
              << config.parameters.mapping.max_point_reprojection_error
              << " ba_max_points="
              << config.parameters.bundle_adjustment.max_points
              << std::endl;
}

// PGO aligns the trajectory to the loop constraints; global Schur BA then
// refines poses and structure jointly and provides the published result.
void run_backend_correction(const AppConfig& config, int32_t frame_id,
                            bool force, BowDatabase* bow_db,
                            MapArchive* archive, Visualizer* visualizer,
                            TrackState* state) {
    if (config.parameters.loop_closure.pgo_enabled != 0) {
        std::vector<cv::Point3f> optimized_centers;
        std::vector<Pose> corrected_poses;
        LoopCorrection correction;
        if (run_pose_graph_optimization(
                bow_db, config.parameters.loop_closure, frame_id, force,
                config.debug_geometry, &optimized_centers,
                &corrected_poses, &correction)) {
            if (config.parameters.loop_closure.gba_enabled != 0) {
                std::vector<cv::Point3f> ba_centers;
                if (run_loop_global_ba(*bow_db, corrected_poses, *archive,
                                       config.camera,
                                       config.parameters.loop_closure,
                                       frame_id, config.debug_geometry,
                                       &ba_centers)) {
                    optimized_centers = ba_centers;
                }
            }
            // Tracking continues from the corrected frame, so the scale it
            // triangulates in stays the scale the pose graph published.
            apply_loop_correction(correction, state, archive);
            std::cout << "loop_correction_applied frame=" << frame_id
                      << " scale=" << correction.scale
                      << " map_points=" << state->map_points.size()
                      << std::endl;
            log_optimized_trajectory(visualizer, frame_id,
                                     optimized_centers);
            bow_db->last_corrected_centers = optimized_centers;
        }
    }
}

void run_loop_query(const cv::Mat& image, int32_t frame_id,
                    const Pose& pose, const AppConfig& config,
                    BowDatabase* bow_db, MapArchive* archive,
                    Visualizer* visualizer, TrackState* state) {
    LoopClosureEvent closure;
    if (query_and_add_loop(image, bow_db, frame_id, pose, config.camera,
                           config.parameters.loop_closure,
                           config.debug_geometry, &closure)) {
        log_loop_edge(visualizer, frame_id, closure.match_center,
                      closure.query_center);
    }
    run_backend_correction(config, frame_id, false, bow_db, archive,
                           visualizer, state);
    state->loop_queries = static_cast<int32_t>(bow_db->keyframes.size());
}

void log_frame_state(Visualizer* visualizer, int32_t frame_id,
                     const cv::Mat& image, const TrackState& state) {
    log_visualization(visualizer, frame_id, image, state.prev_points,
                      map_point_positions(state.map_points),
                      state.all_map_points, state.last_pose);
}

void print_frame_stats(int32_t frame_id, const TrackState& state) {
    std::cout << "frame=" << frame_id
              << " tracks=" << state.prev_points.size()
              << " map_points="
              << count_positioned_map_points(state.map_points)
              << " pending=" << count_pending_map_points(state.map_points);
}

void align_tracks_with_map_points(TrackState* state) {
    if (state->prev_points.size() != state->map_points.size()) {
        const std::size_t aligned_count = std::min(
            state->prev_points.size(), state->map_points.size());
        state->prev_points.resize(aligned_count);
        state->map_points.resize(aligned_count);
    }
}

void collect_pnp_candidates(const std::vector<MapPoint>& map_points,
                            const std::vector<cv::Point2f>& observations,
                            int32_t frame_id,
                            const MvoParameters& parameters,
                            bool debug_geometry,
                            std::vector<cv::Point3f>* pnp_map_points,
                            std::vector<cv::Point2f>* pnp_image_points) {
    std::vector<cv::Point3f> fallback_map_points;
    std::vector<cv::Point2f> fallback_image_points;
    const std::size_t aligned_count = std::min(map_points.size(),
                                               observations.size());
    for (std::size_t i = 0; i < aligned_count; ++i) {
        const MapPoint& point = map_points[i];
        if (!map_point_usable_for_pnp(point, frame_id, parameters.mapping)) {
            continue;
        }
        if (!point.candidate) {
            pnp_map_points->push_back(point.position);
            pnp_image_points->push_back(observations[i]);
        } else {
            fallback_map_points.push_back(point.position);
            fallback_image_points.push_back(observations[i]);
        }
    }
    const std::size_t stable_candidates = pnp_image_points->size();
    if (static_cast<int32_t>(stable_candidates) <
        parameters.pnp.min_tracks) {
        pnp_map_points->insert(pnp_map_points->end(),
                               fallback_map_points.begin(),
                               fallback_map_points.end());
        pnp_image_points->insert(pnp_image_points->end(),
                                 fallback_image_points.begin(),
                                 fallback_image_points.end());
    }
    if (debug_geometry) {
        std::cout << "pnp_candidate_pool frame=" << frame_id
                  << " stable=" << stable_candidates
                  << " fallback=" << fallback_image_points.size()
                  << " total=" << pnp_image_points->size()
                  << std::endl;
    }
}

void retain_pnp_inlier_tracks(const AppConfig& config, int32_t frame_id,
                              const Pose& pose,
                              std::vector<cv::Point2f>* tracked_points,
                              std::vector<MapPoint>* map_points,
                              MapArchive* archive) {
    std::vector<cv::Point2f> inlier_points;
    std::vector<MapPoint> inlier_map_points;
    inlier_points.reserve(tracked_points->size());
    inlier_map_points.reserve(map_points->size());
    const std::size_t aligned_count = std::min(tracked_points->size(),
                                               map_points->size());
    for (std::size_t i = 0; i < aligned_count; ++i) {
        MapPoint point = (*map_points)[i];
        const cv::Point2f& observation = (*tracked_points)[i];
        if (point.has_position) {
            const double residual = reprojection_residual(
                point.position, observation, pose, config.camera);
            if (std::isfinite(residual) &&
                residual <=
                    config.parameters.pnp.reprojection_inlier_threshold) {
                update_tracked_map_point(&point, frame_id, residual,
                                         config.parameters.mapping);
                if (point.id >= 0) {
                    archive_observation(archive,
                                        {frame_id, point.id, observation});
                    archive->positions[point.id] = point.position;
                    archive->last_seen[point.id] = frame_id;
                }
                inlier_points.push_back(observation);
                inlier_map_points.push_back(point);
            }
        } else if (point.has_anchor &&
                   point.age < config.parameters.mapping.max_active_age) {
            update_pending_map_point(&point, frame_id,
                                     config.parameters.mapping);
            inlier_points.push_back(observation);
            inlier_map_points.push_back(point);
        }
    }
    *tracked_points = inlier_points;
    *map_points = inlier_map_points;
}

bool initialize_tracking(const AppConfig& config, FrameSource* source,
                         Visualizer* visualizer, BowDatabase* bow_db,
                         MapArchive* archive, TrackState* state) {
    cv::Mat image0;
    cv::Mat image1;
    std::vector<cv::Point2f> initial_points;
    std::vector<cv::Point2f> tracked0;
    std::vector<cv::Point2f> tracked1;

    bool ok = read_next_frame(source, &image0);
    if (ok) {
        ok = read_next_frame(source, &image1);
    }
    if (ok) {
        ok = detect_initial_points(image0, config.parameters.feature,
                                   &initial_points);
    }
    if (ok) {
        ok = track_points(image0, image1, initial_points,
                          config.parameters.feature, &tracked0, &tracked1,
                          nullptr, config.debug_geometry, "init", false);
    }
    if (ok) {
        ok = initialize_two_view(tracked0, tracked1, config.camera,
                                 config.parameters, config.run_ba,
                                 config.debug_geometry, state);
    }
    if (ok) {
        state->prev_image = image1;
        state->prev_pose = state->last_pose;
        state->frames_processed = 2;
        state->all_map_points = map_point_positions(state->map_points);
        Pose pose0;
        set_identity_pose(&pose0);
        log_visualization(visualizer, 0, image0, tracked0,
                          map_point_positions(state->map_points),
                          state->all_map_points, pose0);
        log_visualization(visualizer, 1, image1, state->prev_points,
                          map_point_positions(state->map_points),
                          state->all_map_points, state->last_pose);
        run_loop_query(image0, 0, pose0, config, bow_db, archive,
                       visualizer, state);
        run_loop_query(image1, 1, state->last_pose, config, bow_db,
                       archive, visualizer, state);
    }
    return ok;
}

void process_frame_with_tracks(const AppConfig& config, int32_t frame_id,
                               const cv::Mat& image, BowDatabase* bow_db,
                               MapArchive* archive, Visualizer* visualizer,
                               TrackState* state) {
    std::vector<cv::Point2f> tracked_prev;
    std::vector<cv::Point2f> raw_tracked_next;
    std::vector<int32_t> tracked_indices;
    const Pose previous_pose = state->prev_pose;
    bool pnp_ok = false;
    bool recovered_ok = false;
    std::vector<cv::Point2f> motion_prediction;
    const std::vector<cv::Point2f>* prediction =
        build_motion_prediction(*state, config.camera, &motion_prediction);
    track_points(state->prev_image, image, state->prev_points,
                 config.parameters.feature, &tracked_prev, &raw_tracked_next,
                 &tracked_indices, config.debug_geometry,
                 "frame_" + std::to_string(frame_id), true, prediction);

    std::vector<cv::Point2f> tracked_next;
    std::vector<MapPoint> next_map_points;
    const std::size_t tracked_count = std::min(raw_tracked_next.size(),
                                               tracked_indices.size());
    tracked_next.reserve(tracked_count);
    next_map_points.reserve(tracked_count);
    for (std::size_t i = 0; i < tracked_count; ++i) {
        tracked_next.push_back(raw_tracked_next[i]);
        next_map_points.push_back(
            state->map_points[static_cast<std::size_t>(tracked_indices[i])]);
    }

    std::vector<cv::Point3f> pnp_map_points;
    std::vector<cv::Point2f> pnp_image_points;
    collect_pnp_candidates(next_map_points, tracked_next, frame_id,
                           config.parameters, config.debug_geometry,
                           &pnp_map_points, &pnp_image_points);
    if (static_cast<int32_t>(pnp_image_points.size()) >=
        config.parameters.pnp.min_tracks) {
        const Pose initial_pose = state->last_pose;
        if (run_pnp(&pnp_map_points, &pnp_image_points, config.camera,
                    initial_pose, config.parameters.pnp,
                    config.debug_geometry, &state->last_pose)) {
            ++state->pnp_success;
            pnp_ok = true;
        }
    } else {
        std::cout << "pnp_skipped tracks=" << pnp_image_points.size()
                  << " tracked=" << tracked_next.size() << std::endl;
    }

    if (pnp_ok) {
        retain_pnp_inlier_tracks(config, frame_id, state->last_pose,
                                 &tracked_next, &next_map_points, archive);
        triangulate_pending_map_points(
            tracked_next, state->last_pose, config.camera, config.parameters,
            frame_id, config.debug_geometry, &next_map_points,
            &state->all_map_points);
        add_pending_feature_tracks(image, state->last_pose, frame_id,
                                   config.parameters, config.debug_geometry,
                                   &tracked_next, &next_map_points);
        if (config.debug_geometry) {
            std::cout << "lifecycle frame=" << frame_id
                      << " active=" << tracked_next.size()
                      << " pnp_candidates=" << pnp_image_points.size()
                      << std::endl;
        }
    } else {
        recovered_ok = recover_two_view_from_reference(
            state->prev_image, image, previous_pose, config.camera,
            config.parameters, config.run_ba, config.debug_geometry,
            frame_id, state);
    }
    if (pnp_ok && !recovered_ok) {
        state->prev_image = image;
        state->pose_before_last = state->prev_pose;
        state->has_pose_before_last = true;
        state->prev_pose = state->last_pose;
        state->prev_points = tracked_next;
        state->map_points = next_map_points;
        // Record this frame's pose, then run windowed local BA over the last
        // few frames. Both must happen after the map update so the current
        // frame's archived observations and committed map points are in the
        // window, and before run_loop_query so the keyframe added for this
        // frame carries the refined pose.
        const MonoLocalBaParameters& local_ba =
            config.parameters.mono_local_ba;
        state->pose_window.push_back({frame_id, state->last_pose});
        if (static_cast<int32_t>(state->pose_window.size()) >
            local_ba.window) {
            state->pose_window.erase(
                state->pose_window.begin(),
                state->pose_window.end() - local_ba.window);
        }
        if (config.run_ba && local_ba.enabled != 0 &&
            static_cast<int32_t>(state->pose_window.size()) >=
                local_ba.window &&
            frame_id % local_ba.interval == 0) {
            run_mono_local_ba(config.camera, local_ba, frame_id,
                              config.debug_geometry, archive, bow_db, state);
        }
    } else if (!recovered_ok) {
        std::cout << "frame_pose_failed frame=" << frame_id
                  << " tracked=" << tracked_next.size()
                  << " state_held=1" << std::endl;
    }
    run_loop_query(image, frame_id, state->last_pose, config, bow_db,
                   archive, visualizer, state);
    ++state->frames_processed;
    if (pnp_ok || recovered_ok) {
        log_frame_state(visualizer, frame_id, image, *state);
    } else {
        const std::vector<cv::Point2f> empty_points;
        const std::vector<cv::Point3f> empty_map_points;
        log_visualization(visualizer, frame_id, image, empty_points,
                          empty_map_points, state->all_map_points,
                          state->last_pose);
    }
    print_frame_stats(frame_id, *state);
    std::cout << std::endl;
}

bool initialize_stereo(const AppConfig& config, FrameSource* source,
                       Visualizer* visualizer, MapArchive* archive,
                       StereoBackend* backend, TrackState* state) {
    cv::Mat left;
    cv::Mat right;
    bool ok = read_next_stereo_frame(source, &left, &right);
    if (ok) {
        Pose pose0;
        set_identity_pose(&pose0);
        ok = initialize_stereo_tracking(left, right, pose0, config.camera,
                                        config.stereo_baseline,
                                        config.parameters, 0,
                                        config.debug_geometry, archive,
                                        state);
    }
    if (ok) {
        state->frames_processed = 1;
        backend->trajectory.push_back({0, state->last_pose});
        log_visualization(visualizer, 0, left, state->prev_points,
                          map_point_positions(state->map_points),
                          state->all_map_points, state->last_pose);
    }
    return ok;
}

void process_stereo_frame(const AppConfig& config, int32_t frame_id,
                          const cv::Mat& left, const cv::Mat& right,
                          MapArchive* archive, Visualizer* visualizer,
                          StereoBackend* backend, TrackState* state) {
    bool pnp_ok = false;
    bool recovered_ok = false;
    std::vector<cv::Point2f> tracked_next;
    std::vector<MapPoint> next_map_points;
    if (!state->prev_points.empty()) {
        std::vector<cv::Point2f> tracked_prev;
        std::vector<cv::Point2f> raw_tracked_next;
        std::vector<int32_t> tracked_indices;
        track_points(state->prev_image, left, state->prev_points,
                     config.parameters.feature, &tracked_prev,
                     &raw_tracked_next, &tracked_indices,
                     config.debug_geometry,
                     "frame_" + std::to_string(frame_id), true);
        const std::size_t tracked_count = std::min(raw_tracked_next.size(),
                                                   tracked_indices.size());
        tracked_next.reserve(tracked_count);
        next_map_points.reserve(tracked_count);
        for (std::size_t i = 0; i < tracked_count; ++i) {
            tracked_next.push_back(raw_tracked_next[i]);
            next_map_points.push_back(
                state->map_points[static_cast<std::size_t>(
                    tracked_indices[i])]);
        }

        std::vector<cv::Point3f> pnp_map_points;
        std::vector<cv::Point2f> pnp_image_points;
        collect_pnp_candidates(next_map_points, tracked_next, frame_id,
                               config.parameters, config.debug_geometry,
                               &pnp_map_points, &pnp_image_points);
        if (static_cast<int32_t>(pnp_image_points.size()) >=
            config.parameters.pnp.min_tracks) {
            const Pose initial_pose = state->last_pose;
            if (run_pnp(&pnp_map_points, &pnp_image_points, config.camera,
                        initial_pose, config.parameters.pnp,
                        config.debug_geometry, &state->last_pose)) {
                ++state->pnp_success;
                pnp_ok = true;
            }
        } else {
            std::cout << "pnp_skipped tracks=" << pnp_image_points.size()
                      << " tracked=" << tracked_next.size() << std::endl;
        }
    }

    if (pnp_ok) {
        retain_pnp_inlier_tracks(config, frame_id, state->last_pose,
                                 &tracked_next, &next_map_points, archive);
        add_pending_feature_tracks(left, state->last_pose, frame_id,
                                   config.parameters, config.debug_geometry,
                                   &tracked_next, &next_map_points);
        // Fresh and previously unmatched tracks both get metric depth from
        // the current pair, so new map points are usable immediately.
        triangulate_stereo_map_points(
            left, right, tracked_next, state->last_pose, config.camera,
            config.stereo_baseline, config.parameters, frame_id,
            config.debug_geometry, &next_map_points, &state->all_map_points,
            archive);
        state->prev_image = left;
        state->prev_pose = state->last_pose;
        state->prev_points = tracked_next;
        state->map_points = next_map_points;
    } else {
        if (config.debug_geometry) {
            std::cout << "stereo_recover_pose frame=" << frame_id
                      << " ortho_err="
                      << rotation_orthonormality_error(state->last_pose)
                      << " det=" << rotation_determinant(state->last_pose)
                      << std::endl;
        }
        recovered_ok = initialize_stereo_tracking(
            left, right, state->last_pose, config.camera,
            config.stereo_baseline, config.parameters, frame_id,
            config.debug_geometry, archive, state);
        if (recovered_ok) {
            std::cout << "stereo_reinitialized frame=" << frame_id
                      << " tracks=" << state->prev_points.size()
                      << " map_points="
                      << count_positioned_map_points(state->map_points)
                      << std::endl;
        } else {
            std::cout << "frame_pose_failed frame=" << frame_id
                      << " tracked=" << tracked_next.size()
                      << " state_held=1" << std::endl;
        }
    }
    backend->trajectory.push_back({frame_id, state->last_pose});
    if (config.run_ba &&
        config.parameters.stereo.local_ba_enabled != 0 &&
        frame_id % config.parameters.stereo.local_ba_interval == 0) {
        run_stereo_local_ba(config.camera, config.stereo_baseline,
                            config.parameters.stereo, frame_id,
                            config.debug_geometry, archive, backend, state);
    }
    ++state->frames_processed;
    if (pnp_ok || recovered_ok) {
        log_frame_state(visualizer, frame_id, left, *state);
    } else {
        const std::vector<cv::Point2f> empty_points;
        const std::vector<cv::Point3f> empty_map_points;
        log_visualization(visualizer, frame_id, left, empty_points,
                          empty_map_points, state->all_map_points,
                          state->last_pose);
    }
    print_frame_stats(frame_id, *state);
    std::cout << std::endl;
}

int32_t run_stereo_app(AppConfig config) {
    int32_t exit_code = 1;
    FrameSource source;
    Visualizer visualizer;
    MapArchive archive;
    StereoBackend backend;
    TrackState state;
    set_identity_pose(&state.prev_pose);
    set_identity_pose(&state.last_pose);

    const bool calib_loaded = load_kitti_calibration(
        config.calib_path, &config.camera);
    if (config.stereo_baseline <= 0.0) {
        load_kitti_stereo_baseline(config.calib_path,
                                   &config.stereo_baseline);
    }
    const bool baseline_ok = config.stereo_baseline > 0.0;
    const bool source_ok = open_frame_source(config, &source);
    const bool visualizer_ok = initialize_visualizer(config, &visualizer);
    log_startup(config, source, calib_loaded);
    std::cout << "stereo right_path=" << config.right_input_path
              << " baseline=" << config.stereo_baseline
              << " loop_closure=disabled" << std::endl;
    if (!baseline_ok) {
        std::cout << "stereo_baseline=missing calib=" << config.calib_path
                  << " hint=provide_P1_or_--baseline" << std::endl;
    }

    if (source_ok && visualizer_ok && baseline_ok) {
        if (initialize_stereo(config, &source, &visualizer, &archive,
                              &backend, &state)) {
            const int32_t frame_limit = std::min(
                config.max_frames, source.total_frames);
            for (int32_t frame_id = 1; frame_id < frame_limit; ++frame_id) {
                cv::Mat left;
                cv::Mat right;
                if (!read_next_stereo_frame(&source, &left, &right)) {
                    std::cout << "frame_read_failed frame=" << frame_id
                              << std::endl;
                    break;
                }
                align_tracks_with_map_points(&state);
                process_stereo_frame(config, frame_id, left, right,
                                     &archive, &visualizer, &backend,
                                     &state);
            }
            if (config.run_ba &&
                config.parameters.stereo.full_ba_enabled != 0) {
                if (run_stereo_full_ba(config.camera,
                                       config.stereo_baseline,
                                       config.parameters.stereo,
                                       config.debug_geometry, archive,
                                       &backend)) {
                    std::vector<cv::Point3f> centers;
                    centers.reserve(backend.optimized.size());
                    for (const StereoFramePose& entry : backend.optimized) {
                        centers.push_back(
                            camera_center_from_pose(entry.pose));
                    }
                    log_optimized_trajectory(&visualizer,
                                             state.frames_processed,
                                             centers);
                }
            } else {
                backend.optimized = backend.trajectory;
            }
            // Dump trajectories for offline evaluation against GT.
            std::ofstream raw_out("build/trajectory_raw.txt");
            for (const StereoFramePose& entry : backend.trajectory) {
                const cv::Point3f c = camera_center_from_pose(entry.pose);
                raw_out << entry.frame_id << " " << c.x << " " << c.y
                        << " " << c.z << "\n";
            }
            std::ofstream corrected_out("build/trajectory_corrected.txt");
            for (const StereoFramePose& entry : backend.optimized) {
                const cv::Point3f c = camera_center_from_pose(entry.pose);
                corrected_out << entry.frame_id << " " << c.x << " " << c.y
                              << " " << c.z << "\n";
            }
            exit_code = 0;
        }
    } else if (!source_ok) {
        std::cout << "input=open_failed type="
                  << input_type_name(config.input_type)
                  << " mode=" << sensor_mode_name(config.sensor_mode)
                  << " path=" << config.input_path
                  << " right_path=" << config.right_input_path << std::endl;
    }

    flush_visualizer(&visualizer);
    std::cout << "summary mode=stereo frames=" << state.frames_processed
              << " keyframes=" << state.keyframes
              << " pnp_success=" << state.pnp_success
              << " local_ba_runs=" << backend.local_ba_runs
              << " local_ba_accepted=" << backend.local_ba_accepted
              << " full_ba_accepted=" << backend.full_ba_accepted
              << " map_points=" << count_positioned_map_points(
                     state.map_points)
              << " pending=" << count_pending_map_points(state.map_points)
              << " exit_code=" << exit_code << std::endl;
    return exit_code;
}

void process_frame_without_tracks(const AppConfig& config, int32_t frame_id,
                                  const cv::Mat& image, BowDatabase* bow_db,
                                  MapArchive* archive, Visualizer* visualizer,
                                  TrackState* state) {
    bool recovered_ok = false;
    if (!state->prev_image.empty()) {
        recovered_ok = recover_two_view_from_reference(
            state->prev_image, image, state->prev_pose, config.camera,
            config.parameters, config.run_ba, config.debug_geometry,
            frame_id, state);
    }
    if (!recovered_ok) {
        state->prev_image = image;
        state->prev_pose = state->last_pose;
        state->prev_points.clear();
        state->map_points.clear();
    }
    run_loop_query(image, frame_id, state->last_pose, config, bow_db,
                   archive, visualizer, state);
    ++state->frames_processed;
    log_frame_state(visualizer, frame_id, image, *state);
    print_frame_stats(frame_id, *state);
    std::cout << " recovered=" << recovered_ok << std::endl;
}

}  // namespace

int32_t run_app(AppConfig config) {
    if (config.sensor_mode == SensorMode::kStereo) {
        return run_stereo_app(std::move(config));
    }
    int32_t exit_code = 1;
    FrameSource source;
    Visualizer visualizer;
    BowDatabase bow_db;
    MapArchive archive;
    TrackState state;
    set_identity_pose(&state.prev_pose);
    set_identity_pose(&state.last_pose);

    const bool calib_loaded = load_kitti_calibration(
        config.calib_path, &config.camera);
    const bool source_ok = open_frame_source(config, &source);
    const bool visualizer_ok = initialize_visualizer(config, &visualizer);
    log_startup(config, source, calib_loaded);

    if (load_vocabulary(config.vocabulary, &bow_db)) {
        std::cout << "vocabulary=loaded words=" << bow_db.vocabulary.size()
                  << std::endl;
    } else {
        std::cout << "vocabulary=missing_or_empty path=" << config.vocabulary
                  << std::endl;
    }

    if (source_ok && visualizer_ok) {
        if (initialize_tracking(config, &source, &visualizer, &bow_db,
                                &archive, &state)) {
            const int32_t frame_limit = std::min(
                config.max_frames, source.total_frames);
            for (int32_t frame_id = 2; frame_id < frame_limit; ++frame_id) {
                cv::Mat image;
                if (!read_next_frame(&source, &image)) {
                    std::cout << "frame_read_failed frame=" << frame_id
                              << std::endl;
                    break;
                }
                align_tracks_with_map_points(&state);
                if (!state.prev_points.empty()) {
                    process_frame_with_tracks(config, frame_id, image,
                                              &bow_db, &archive,
                                              &visualizer, &state);
                } else {
                    process_frame_without_tracks(config, frame_id, image,
                                                 &bow_db, &archive,
                                                 &visualizer, &state);
                }
            }
            // Flush closures whose episode was still open at sequence end.
            run_backend_correction(config, state.frames_processed, true,
                                   &bow_db, &archive, &visualizer, &state);
            // Dump trajectories for offline evaluation against GT.
            std::ofstream raw_out("build/trajectory_raw.txt");
            for (const LoopKeyframe& keyframe : bow_db.keyframes) {
                const cv::Point3f c =
                    camera_center_from_pose(keyframe.vo_pose);
                raw_out << keyframe.frame_id << " " << c.x << " " << c.y
                        << " " << c.z << "\n";
            }
            // Keyframe poses live in the corrected frame once a pose graph
            // run is applied, so they cover the whole sequence instead of
            // stopping at the last optimization.
            std::ofstream corrected_out("build/trajectory_corrected.txt");
            for (const LoopKeyframe& keyframe : bow_db.keyframes) {
                const cv::Point3f c = camera_center_from_pose(keyframe.pose);
                corrected_out << keyframe.frame_id << " " << c.x << " " << c.y
                              << " " << c.z << "\n";
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
              << " loop_closures=" << bow_db.closures.size()
              << " pgo_runs=" << bow_db.pgo_runs
              << " map_points=" << count_positioned_map_points(
                     state.map_points)
              << " pending=" << count_pending_map_points(state.map_points)
              << " exit_code=" << exit_code << std::endl;
    return exit_code;
}

}  // namespace mvo
