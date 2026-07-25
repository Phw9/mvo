#include "parameters.h"

#include <opencv2/core/persistence.hpp>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>

namespace mvo {
namespace {

void read_int_parameter(const cv::FileNode& node,
                        const std::string& key,
                        int32_t* value) {
    const cv::FileNode child = node[key];
    if (!child.empty()) {
        *value = static_cast<int32_t>(child);
    }
}

void read_double_parameter(const cv::FileNode& node,
                           const std::string& key,
                           double* value) {
    const cv::FileNode child = node[key];
    if (!child.empty()) {
        *value = static_cast<double>(child);
    }
}

void read_float_parameter(const cv::FileNode& node,
                          const std::string& key,
                          float* value) {
    const cv::FileNode child = node[key];
    if (!child.empty()) {
        *value = static_cast<float>(static_cast<double>(child));
    }
}

cv::FileNode module_node(cv::FileStorage* fs, const std::string& module_name) {
    cv::FileNode node = fs->root();
    const cv::FileNode child = node[module_name];
    if (!child.empty()) {
        node = child;
    }
    return node;
}

bool load_feature_parameters(const std::filesystem::path& path,
                             FeatureParameters* parameters) {
    bool ok = false;
    cv::FileStorage fs(path.string(), cv::FileStorage::READ |
                                          cv::FileStorage::FORMAT_JSON);
    if (fs.isOpened()) {
        const cv::FileNode node = module_node(&fs, "feature");
        read_int_parameter(node, "min_init_tracks",
                           &parameters->min_init_tracks);
        read_int_parameter(node, "max_features", &parameters->max_features);
        read_int_parameter(node, "max_init_tracks",
                           &parameters->max_init_tracks);
        read_int_parameter(node, "refresh_grid_rows",
                           &parameters->refresh_grid_rows);
        read_int_parameter(node, "refresh_grid_cols",
                           &parameters->refresh_grid_cols);
        read_int_parameter(node, "klt_init_window_size",
                           &parameters->klt_init_window_size);
        read_int_parameter(node, "klt_init_pyramid_levels",
                           &parameters->klt_init_pyramid_levels);
        read_int_parameter(node, "klt_window_size",
                           &parameters->klt_window_size);
        read_int_parameter(node, "klt_pyramid_levels",
                           &parameters->klt_pyramid_levels);
        read_int_parameter(node, "klt_max_iterations",
                           &parameters->klt_max_iterations);
        read_double_parameter(node, "klt_quality",
                              &parameters->klt_quality);
        read_double_parameter(node, "klt_min_distance",
                              &parameters->klt_min_distance);
        read_double_parameter(node, "klt_epsilon",
                              &parameters->klt_epsilon);
        read_double_parameter(node, "klt_min_eig_threshold",
                              &parameters->klt_min_eig_threshold);
        read_double_parameter(node, "max_forward_backward_error",
                              &parameters->max_forward_backward_error);
        read_double_parameter(node, "forward_backward_motion_ratio",
                              &parameters->forward_backward_motion_ratio);
        read_double_parameter(
            node, "max_adaptive_forward_backward_error",
            &parameters->max_adaptive_forward_backward_error);
        ok = true;
    }
    return ok;
}

bool load_pnp_parameters(const std::filesystem::path& path,
                         PnpParameters* parameters) {
    bool ok = false;
    cv::FileStorage fs(path.string(), cv::FileStorage::READ |
                                          cv::FileStorage::FORMAT_JSON);
    if (fs.isOpened()) {
        const cv::FileNode node = module_node(&fs, "pnp");
        read_int_parameter(node, "min_tracks", &parameters->min_tracks);
        read_int_parameter(node, "min_stable_inliers",
                           &parameters->min_stable_inliers);
        read_int_parameter(node, "ransac_iterations",
                           &parameters->ransac_iterations);
        read_int_parameter(node, "ransac_sample_size",
                           &parameters->ransac_sample_size);
        read_double_parameter(node, "reprojection_inlier_threshold",
                              &parameters->reprojection_inlier_threshold);
        read_double_parameter(node, "max_reprojection_p90",
                              &parameters->max_reprojection_p90);
        ok = true;
    }
    return ok;
}

bool load_initializer_parameters(const std::filesystem::path& path,
                                 InitializerParameters* parameters) {
    bool ok = false;
    cv::FileStorage fs(path.string(), cv::FileStorage::READ |
                                          cv::FileStorage::FORMAT_JSON);
    if (fs.isOpened()) {
        const cv::FileNode node = module_node(&fs, "initializer");
        read_int_parameter(node, "min_tracks", &parameters->min_tracks);
        read_int_parameter(node, "min_map_points",
                           &parameters->min_map_points);
        read_int_parameter(node, "ransac_max_iters",
                           &parameters->ransac_max_iters);
        read_double_parameter(node, "fundamental_inlier_threshold",
                              &parameters->fundamental_inlier_threshold);
        read_double_parameter(node, "homography_inlier_threshold",
                              &parameters->homography_inlier_threshold);
        read_double_parameter(node, "homography_model_ratio",
                              &parameters->homography_model_ratio);
        read_double_parameter(node, "min_parallax_deg",
                              &parameters->min_parallax_deg);
        read_double_parameter(node, "max_triangulation_p90",
                              &parameters->max_triangulation_p90);
        ok = true;
    }
    return ok;
}

bool load_mapping_parameters(const std::filesystem::path& path,
                             MappingParameters* parameters) {
    bool ok = false;
    cv::FileStorage fs(path.string(), cv::FileStorage::READ |
                                          cv::FileStorage::FORMAT_JSON);
    if (fs.isOpened()) {
        const cv::FileNode node = module_node(&fs, "mapping");
        read_int_parameter(node, "target_tracked_map_points",
                           &parameters->target_tracked_map_points);
        read_int_parameter(node, "max_refresh_candidates",
                           &parameters->max_refresh_candidates);
        read_int_parameter(node, "max_active_age",
                           &parameters->max_active_age);
        read_int_parameter(node, "max_unseen_frames",
                           &parameters->max_unseen_frames);
        read_int_parameter(node, "min_pnp_track_length",
                           &parameters->min_pnp_track_length);
        read_int_parameter(node, "candidate_min_track_length",
                           &parameters->candidate_min_track_length);
        read_double_parameter(node, "max_point_reprojection_error",
                              &parameters->max_point_reprojection_error);
        read_double_parameter(node, "min_refresh_parallax_deg",
                              &parameters->min_refresh_parallax_deg);
        read_double_parameter(node, "max_triangulation_p90",
                              &parameters->max_triangulation_p90);
        ok = true;
    }
    return ok;
}

bool load_bundle_adjustment_parameters(
    const std::filesystem::path& path,
    BundleAdjustmentParameters* parameters) {
    bool ok = false;
    cv::FileStorage fs(path.string(), cv::FileStorage::READ |
                                          cv::FileStorage::FORMAT_JSON);
    if (fs.isOpened()) {
        const cv::FileNode node = module_node(&fs, "bundle_adjustment");
        read_int_parameter(node, "max_points", &parameters->max_points);
        read_int_parameter(node, "min_points", &parameters->min_points);
        read_int_parameter(node, "max_iterations",
                           &parameters->max_iterations);
        read_int_parameter(node, "solver", &parameters->solver);
        read_double_parameter(node, "loss_scale", &parameters->loss_scale);
        read_double_parameter(node, "min_baseline",
                              &parameters->min_baseline);
        read_double_parameter(node, "max_anchor_scale_change",
                              &parameters->max_anchor_scale_change);
        read_double_parameter(node, "max_cost_growth",
                              &parameters->max_cost_growth);
        read_double_parameter(node, "max_reprojection_p90",
                              &parameters->max_reprojection_p90);
        ok = true;
    }
    return ok;
}

bool load_loop_closure_parameters(const std::filesystem::path& path,
                                  LoopClosureParameters* parameters) {
    bool ok = false;
    cv::FileStorage fs(path.string(), cv::FileStorage::READ |
                                          cv::FileStorage::FORMAT_JSON);
    if (fs.isOpened()) {
        const cv::FileNode node = module_node(&fs, "loop_closure");
        read_int_parameter(node, "orb_features", &parameters->orb_features);
        read_int_parameter(node, "recent_exclusion",
                           &parameters->recent_exclusion);
        read_int_parameter(node, "min_matches", &parameters->min_matches);
        read_int_parameter(node, "min_inliers", &parameters->min_inliers);
        read_int_parameter(node, "ransac_max_iters",
                           &parameters->ransac_max_iters);
        read_int_parameter(node, "min_consecutive_detections",
                           &parameters->min_consecutive_detections);
        read_int_parameter(node, "consistency_window",
                           &parameters->consistency_window);
        read_int_parameter(node, "metric_neighbor_gap",
                           &parameters->metric_neighbor_gap);
        read_int_parameter(node, "metric_min_inliers",
                           &parameters->metric_min_inliers);
        read_int_parameter(node, "metric_ransac_iters",
                           &parameters->metric_ransac_iters);
        read_double_parameter(node, "metric_inlier_ratio",
                              &parameters->metric_inlier_ratio);
        read_double_parameter(node, "metric_max_scale_ratio",
                              &parameters->metric_max_scale_ratio);
        read_int_parameter(node, "metric_required",
                           &parameters->metric_required);
        read_double_parameter(node, "metric_min_parallax_deg",
                              &parameters->metric_min_parallax_deg);
        read_double_parameter(node, "metric_max_reprojection_error",
                              &parameters->metric_max_reprojection_error);
        read_int_parameter(node, "duplicate_frame_gap",
                           &parameters->duplicate_frame_gap);
        read_int_parameter(node, "duplicate_match_window",
                           &parameters->duplicate_match_window);
        read_double_parameter(node, "duplicate_distance",
                              &parameters->duplicate_distance);
        read_double_parameter(node, "min_score", &parameters->min_score);
        read_double_parameter(node, "match_ratio",
                              &parameters->match_ratio);
        read_double_parameter(node, "inlier_threshold",
                              &parameters->inlier_threshold);
        read_double_parameter(node, "max_rotation_error",
                              &parameters->max_rotation_error);
        read_int_parameter(node, "pgo_enabled", &parameters->pgo_enabled);
        read_int_parameter(node, "pgo_max_graph_poses",
                           &parameters->pgo_max_graph_poses);
        read_int_parameter(node, "pgo_max_iterations",
                           &parameters->pgo_max_iterations);
        read_int_parameter(node, "pgo_episode_end_gap",
                           &parameters->pgo_episode_end_gap);
        read_int_parameter(node, "pgo_pending_trigger",
                           &parameters->pgo_pending_trigger);
        read_int_parameter(node, "pgo_loss_type",
                           &parameters->pgo_loss_type);
        read_double_parameter(node, "pgo_loss_scale",
                              &parameters->pgo_loss_scale);
        read_double_parameter(node, "pgo_loop_translation_weight",
                              &parameters->pgo_loop_translation_weight);
        read_double_parameter(node, "pgo_loop_rotation_weight",
                              &parameters->pgo_loop_rotation_weight);
        read_double_parameter(node, "pgo_loop_scale_weight",
                              &parameters->pgo_loop_scale_weight);
        read_double_parameter(node, "pgo_scale_weight",
                              &parameters->pgo_scale_weight);
        read_double_parameter(node, "pgo_max_scale_change",
                              &parameters->pgo_max_scale_change);
        read_int_parameter(node, "gba_enabled", &parameters->gba_enabled);
        read_int_parameter(node, "gba_max_cameras",
                           &parameters->gba_max_cameras);
        read_int_parameter(node, "gba_max_points",
                           &parameters->gba_max_points);
        read_int_parameter(node, "gba_min_observations",
                           &parameters->gba_min_observations);
        read_int_parameter(node, "gba_max_loop_points",
                           &parameters->gba_max_loop_points);
        read_int_parameter(node, "gba_max_iterations",
                           &parameters->gba_max_iterations);
        read_double_parameter(node, "gba_loss_scale",
                              &parameters->gba_loss_scale);
        ok = true;
    }
    return ok;
}

bool load_stereo_parameters(const std::filesystem::path& path,
                            StereoParameters* parameters) {
    bool ok = false;
    cv::FileStorage fs(path.string(), cv::FileStorage::READ |
                                          cv::FileStorage::FORMAT_JSON);
    if (fs.isOpened()) {
        const cv::FileNode node = module_node(&fs, "stereo");
        read_double_parameter(node, "max_epipolar_error",
                              &parameters->max_epipolar_error);
        read_double_parameter(node, "min_disparity",
                              &parameters->min_disparity);
        read_double_parameter(node, "max_disparity",
                              &parameters->max_disparity);
        read_double_parameter(node, "min_depth", &parameters->min_depth);
        read_double_parameter(node, "max_depth", &parameters->max_depth);
        read_int_parameter(node, "min_init_points",
                           &parameters->min_init_points);
        read_double_parameter(node, "max_rotation_error",
                              &parameters->max_rotation_error);
        read_int_parameter(node, "ba_jacobian_mode",
                           &parameters->ba_jacobian_mode);
        read_int_parameter(node, "local_ba_enabled",
                           &parameters->local_ba_enabled);
        read_int_parameter(node, "local_ba_interval",
                           &parameters->local_ba_interval);
        read_int_parameter(node, "local_ba_window",
                           &parameters->local_ba_window);
        read_int_parameter(node, "local_ba_max_points",
                           &parameters->local_ba_max_points);
        read_int_parameter(node, "local_ba_min_observations",
                           &parameters->local_ba_min_observations);
        read_int_parameter(node, "local_ba_min_camera_observations",
                           &parameters->local_ba_min_camera_observations);
        read_int_parameter(node, "local_ba_max_iterations",
                           &parameters->local_ba_max_iterations);
        read_int_parameter(node, "local_ba_stereo_rows",
                           &parameters->local_ba_stereo_rows);
        read_double_parameter(node, "local_ba_loss_scale",
                              &parameters->local_ba_loss_scale);
        read_int_parameter(node, "full_ba_enabled",
                           &parameters->full_ba_enabled);
        read_int_parameter(node, "full_ba_max_cameras",
                           &parameters->full_ba_max_cameras);
        read_int_parameter(node, "full_ba_max_points",
                           &parameters->full_ba_max_points);
        read_int_parameter(node, "full_ba_min_observations",
                           &parameters->full_ba_min_observations);
        read_int_parameter(node, "full_ba_min_camera_observations",
                           &parameters->full_ba_min_camera_observations);
        read_int_parameter(node, "full_ba_max_iterations",
                           &parameters->full_ba_max_iterations);
        read_int_parameter(node, "full_ba_stereo_rows",
                           &parameters->full_ba_stereo_rows);
        read_double_parameter(node, "full_ba_loss_scale",
                              &parameters->full_ba_loss_scale);
        ok = true;
    }
    return ok;
}

bool load_mono_local_ba_parameters(const std::filesystem::path& path,
                                   MonoLocalBaParameters* parameters) {
    bool ok = false;
    cv::FileStorage fs(path.string(), cv::FileStorage::READ |
                                          cv::FileStorage::FORMAT_JSON);
    if (fs.isOpened()) {
        const cv::FileNode node = module_node(&fs, "mono_local_ba");
        read_int_parameter(node, "enabled", &parameters->enabled);
        read_int_parameter(node, "window", &parameters->window);
        read_int_parameter(node, "interval", &parameters->interval);
        read_int_parameter(node, "max_points", &parameters->max_points);
        read_int_parameter(node, "min_observations",
                           &parameters->min_observations);
        read_int_parameter(node, "min_camera_observations",
                           &parameters->min_camera_observations);
        read_int_parameter(node, "max_iterations",
                           &parameters->max_iterations);
        read_double_parameter(node, "loss_scale", &parameters->loss_scale);
        read_int_parameter(node, "solver", &parameters->solver);
        ok = true;
    }
    return ok;
}

bool load_visualization_parameters(const std::filesystem::path& path,
                                   VisualizationParameters* parameters) {
    bool ok = false;
    cv::FileStorage fs(path.string(), cv::FileStorage::READ |
                                          cv::FileStorage::FORMAT_JSON);
    if (fs.isOpened()) {
        const cv::FileNode node = module_node(&fs, "visualization");
        read_float_parameter(node, "previous_map_point_radius",
                             &parameters->previous_map_point_radius);
        read_float_parameter(node, "current_map_point_radius",
                             &parameters->current_map_point_radius);
        read_float_parameter(node, "trajectory_radius",
                             &parameters->trajectory_radius);
        read_float_parameter(node, "klt_track_radius",
                             &parameters->klt_track_radius);
        read_float_parameter(node, "loop_edge_radius",
                             &parameters->loop_edge_radius);
        read_float_parameter(node, "optimized_trajectory_radius",
                             &parameters->optimized_trajectory_radius);
        read_int_parameter(node, "log_chunk_size",
                           &parameters->log_chunk_size);
        ok = true;
    }
    return ok;
}

void sanitize_parameters(MvoParameters* parameters) {
    parameters->feature.min_init_tracks =
        std::max(2, parameters->feature.min_init_tracks);
    parameters->feature.max_features =
        std::max(parameters->feature.min_init_tracks,
                 parameters->feature.max_features);
    parameters->feature.max_init_tracks =
        std::max(2, parameters->feature.max_init_tracks);
    parameters->feature.refresh_grid_rows =
        std::max(1, parameters->feature.refresh_grid_rows);
    parameters->feature.refresh_grid_cols =
        std::max(1, parameters->feature.refresh_grid_cols);
    parameters->feature.klt_init_window_size =
        std::max(3, parameters->feature.klt_init_window_size);
    parameters->feature.klt_window_size =
        std::max(3, parameters->feature.klt_window_size);
    parameters->pnp.min_tracks = std::max(4, parameters->pnp.min_tracks);
    parameters->pnp.min_stable_inliers =
        std::max(parameters->pnp.min_tracks,
                 parameters->pnp.min_stable_inliers);
    parameters->pnp.ransac_iterations =
        std::max(0, parameters->pnp.ransac_iterations);
    parameters->pnp.ransac_sample_size =
        std::max(parameters->pnp.min_tracks,
                 parameters->pnp.ransac_sample_size);
    parameters->initializer.min_tracks =
        std::max(8, parameters->initializer.min_tracks);
    parameters->initializer.min_map_points =
        std::max(4, parameters->initializer.min_map_points);
    parameters->mapping.target_tracked_map_points =
        std::max(1, parameters->mapping.target_tracked_map_points);
    parameters->mapping.max_refresh_candidates =
        std::max(1, parameters->mapping.max_refresh_candidates);
    parameters->mapping.max_active_age =
        std::max(1, parameters->mapping.max_active_age);
    parameters->mapping.max_unseen_frames =
        std::max(0, parameters->mapping.max_unseen_frames);
    parameters->mapping.min_pnp_track_length =
        std::max(1, parameters->mapping.min_pnp_track_length);
    parameters->mapping.candidate_min_track_length =
        std::max(1, parameters->mapping.candidate_min_track_length);
    parameters->mapping.max_point_reprojection_error =
        std::max(0.1, parameters->mapping.max_point_reprojection_error);
    parameters->mapping.max_triangulation_p90 =
        std::max(0.1, parameters->mapping.max_triangulation_p90);
    parameters->bundle_adjustment.max_points =
        std::max(1, parameters->bundle_adjustment.max_points);
    parameters->bundle_adjustment.min_points =
        std::max(1, parameters->bundle_adjustment.min_points);
    parameters->bundle_adjustment.max_iterations =
        std::max(1, parameters->bundle_adjustment.max_iterations);
    parameters->bundle_adjustment.solver = std::min(
        1, std::max(0, parameters->bundle_adjustment.solver));
    parameters->bundle_adjustment.loss_scale =
        std::max(0.1, parameters->bundle_adjustment.loss_scale);
    parameters->bundle_adjustment.min_baseline =
        std::max(1.0e-12, parameters->bundle_adjustment.min_baseline);
    parameters->bundle_adjustment.max_anchor_scale_change =
        std::max(1.0, parameters->bundle_adjustment.max_anchor_scale_change);
    parameters->bundle_adjustment.max_cost_growth =
        std::max(1.0, parameters->bundle_adjustment.max_cost_growth);
    parameters->bundle_adjustment.max_reprojection_p90 =
        std::max(0.1, parameters->bundle_adjustment.max_reprojection_p90);
    parameters->loop_closure.orb_features =
        std::max(1, parameters->loop_closure.orb_features);
    parameters->loop_closure.recent_exclusion =
        std::max(0, parameters->loop_closure.recent_exclusion);
    parameters->loop_closure.min_matches =
        std::max(8, parameters->loop_closure.min_matches);
    parameters->loop_closure.min_inliers =
        std::max(8, parameters->loop_closure.min_inliers);
    parameters->loop_closure.ransac_max_iters =
        std::max(1, parameters->loop_closure.ransac_max_iters);
    parameters->loop_closure.min_consecutive_detections =
        std::max(1, parameters->loop_closure.min_consecutive_detections);
    parameters->loop_closure.consistency_window =
        std::max(0, parameters->loop_closure.consistency_window);
    parameters->loop_closure.metric_neighbor_gap =
        std::max(1, parameters->loop_closure.metric_neighbor_gap);
    parameters->loop_closure.metric_min_inliers =
        std::max(4, parameters->loop_closure.metric_min_inliers);
    parameters->loop_closure.metric_ransac_iters =
        std::max(1, parameters->loop_closure.metric_ransac_iters);
    parameters->loop_closure.metric_inlier_ratio =
        std::max(1.0e-3, parameters->loop_closure.metric_inlier_ratio);
    parameters->loop_closure.metric_max_scale_ratio =
        std::max(1.0, parameters->loop_closure.metric_max_scale_ratio);
    parameters->loop_closure.metric_min_parallax_deg =
        std::max(0.0, parameters->loop_closure.metric_min_parallax_deg);
    parameters->loop_closure.metric_max_reprojection_error =
        std::max(0.1, parameters->loop_closure.metric_max_reprojection_error);
    parameters->loop_closure.pgo_loop_scale_weight =
        std::max(0.0, parameters->loop_closure.pgo_loop_scale_weight);
    parameters->loop_closure.duplicate_frame_gap =
        std::max(0, parameters->loop_closure.duplicate_frame_gap);
    parameters->loop_closure.duplicate_match_window =
        std::max(0, parameters->loop_closure.duplicate_match_window);
    parameters->loop_closure.duplicate_distance =
        std::max(0.0, parameters->loop_closure.duplicate_distance);
    parameters->loop_closure.min_score =
        std::max(0.0, parameters->loop_closure.min_score);
    parameters->loop_closure.match_ratio = std::min(
        1.0, std::max(0.1, parameters->loop_closure.match_ratio));
    parameters->loop_closure.inlier_threshold =
        std::max(0.1, parameters->loop_closure.inlier_threshold);
    parameters->loop_closure.max_rotation_error =
        std::max(1.0e-12, parameters->loop_closure.max_rotation_error);
    parameters->loop_closure.pgo_max_graph_poses =
        std::max(2, parameters->loop_closure.pgo_max_graph_poses);
    parameters->loop_closure.pgo_max_iterations =
        std::max(1, parameters->loop_closure.pgo_max_iterations);
    parameters->loop_closure.pgo_episode_end_gap =
        std::max(0, parameters->loop_closure.pgo_episode_end_gap);
    parameters->loop_closure.pgo_pending_trigger =
        std::max(1, parameters->loop_closure.pgo_pending_trigger);
    parameters->loop_closure.pgo_loss_type = std::min(
        2, std::max(0, parameters->loop_closure.pgo_loss_type));
    parameters->loop_closure.pgo_loss_scale =
        std::max(0.1, parameters->loop_closure.pgo_loss_scale);
    parameters->loop_closure.pgo_scale_weight =
        std::max(0.0, parameters->loop_closure.pgo_scale_weight);
    parameters->loop_closure.pgo_max_scale_change =
        std::max(1.0, parameters->loop_closure.pgo_max_scale_change);
    parameters->loop_closure.pgo_loop_translation_weight =
        std::max(0.0, parameters->loop_closure.pgo_loop_translation_weight);
    parameters->loop_closure.pgo_loop_rotation_weight =
        std::max(0.0, parameters->loop_closure.pgo_loop_rotation_weight);
    parameters->loop_closure.gba_max_cameras =
        std::max(2, parameters->loop_closure.gba_max_cameras);
    parameters->loop_closure.gba_max_points =
        std::max(1, parameters->loop_closure.gba_max_points);
    parameters->loop_closure.gba_min_observations =
        std::max(2, parameters->loop_closure.gba_min_observations);
    parameters->loop_closure.gba_max_loop_points =
        std::max(0, parameters->loop_closure.gba_max_loop_points);
    parameters->loop_closure.gba_max_iterations =
        std::max(1, parameters->loop_closure.gba_max_iterations);
    parameters->loop_closure.gba_loss_scale =
        std::max(0.1, parameters->loop_closure.gba_loss_scale);
    parameters->stereo.max_epipolar_error =
        std::max(0.1, parameters->stereo.max_epipolar_error);
    parameters->stereo.min_disparity =
        std::max(1.0e-3, parameters->stereo.min_disparity);
    parameters->stereo.max_disparity =
        std::max(parameters->stereo.min_disparity,
                 parameters->stereo.max_disparity);
    parameters->stereo.min_depth =
        std::max(1.0e-3, parameters->stereo.min_depth);
    parameters->stereo.max_depth =
        std::max(parameters->stereo.min_depth,
                 parameters->stereo.max_depth);
    parameters->stereo.min_init_points =
        std::max(4, parameters->stereo.min_init_points);
    parameters->stereo.max_rotation_error =
        std::max(1.0e-12, parameters->stereo.max_rotation_error);
    parameters->stereo.ba_jacobian_mode = std::min(
        1, std::max(0, parameters->stereo.ba_jacobian_mode));
    parameters->stereo.local_ba_interval =
        std::max(1, parameters->stereo.local_ba_interval);
    parameters->stereo.local_ba_window =
        std::max(2, parameters->stereo.local_ba_window);
    parameters->stereo.local_ba_max_points =
        std::max(1, parameters->stereo.local_ba_max_points);
    parameters->stereo.local_ba_min_observations =
        std::max(2, parameters->stereo.local_ba_min_observations);
    parameters->stereo.local_ba_min_camera_observations =
        std::max(1, parameters->stereo.local_ba_min_camera_observations);
    parameters->stereo.local_ba_max_iterations =
        std::max(1, parameters->stereo.local_ba_max_iterations);
    parameters->stereo.local_ba_loss_scale =
        std::max(0.1, parameters->stereo.local_ba_loss_scale);
    parameters->stereo.full_ba_max_cameras =
        std::max(2, parameters->stereo.full_ba_max_cameras);
    parameters->stereo.full_ba_max_points =
        std::max(1, parameters->stereo.full_ba_max_points);
    parameters->stereo.full_ba_min_observations =
        std::max(2, parameters->stereo.full_ba_min_observations);
    parameters->stereo.full_ba_min_camera_observations =
        std::max(1, parameters->stereo.full_ba_min_camera_observations);
    parameters->stereo.full_ba_max_iterations =
        std::max(1, parameters->stereo.full_ba_max_iterations);
    parameters->stereo.full_ba_loss_scale =
        std::max(0.1, parameters->stereo.full_ba_loss_scale);
    parameters->mono_local_ba.window =
        std::max(2, parameters->mono_local_ba.window);
    parameters->mono_local_ba.interval =
        std::max(1, parameters->mono_local_ba.interval);
    parameters->mono_local_ba.max_points =
        std::max(1, parameters->mono_local_ba.max_points);
    parameters->mono_local_ba.min_observations =
        std::max(2, parameters->mono_local_ba.min_observations);
    parameters->mono_local_ba.min_camera_observations =
        std::max(1, parameters->mono_local_ba.min_camera_observations);
    parameters->mono_local_ba.max_iterations =
        std::max(1, parameters->mono_local_ba.max_iterations);
    parameters->mono_local_ba.loss_scale =
        std::max(0.1, parameters->mono_local_ba.loss_scale);
    parameters->mono_local_ba.solver = std::min(
        1, std::max(0, parameters->mono_local_ba.solver));
    parameters->visualization.log_chunk_size =
        std::max(16, parameters->visualization.log_chunk_size);
}

}  // namespace

bool load_parameter_configs(const std::string& directory,
                            MvoParameters* parameters) {
    bool ok = false;
    if (!directory.empty()) {
        const std::filesystem::path root(directory);
        if (std::filesystem::exists(root)) {
            load_feature_parameters(root / "feature.json",
                                    &parameters->feature);
            load_pnp_parameters(root / "pnp.json", &parameters->pnp);
            load_initializer_parameters(root / "initializer.json",
                                        &parameters->initializer);
            load_mapping_parameters(root / "mapping.json",
                                    &parameters->mapping);
            load_bundle_adjustment_parameters(
                root / "bundle_adjustment.json",
                &parameters->bundle_adjustment);
            load_loop_closure_parameters(root / "loop_closure.json",
                                         &parameters->loop_closure);
            load_stereo_parameters(root / "stereo.json",
                                   &parameters->stereo);
            load_mono_local_ba_parameters(root / "local_ba.json",
                                          &parameters->mono_local_ba);
            load_visualization_parameters(root / "visualization.json",
                                          &parameters->visualization);
            sanitize_parameters(parameters);
            std::cout << "parameters=loaded dir=" << directory << std::endl;
            ok = true;
        } else {
            std::cout << "parameters=missing dir=" << directory << std::endl;
        }
    }
    return ok;
}

}  // namespace mvo
