#pragma once

#include <cstdint>
#include <string>

namespace mvo {

struct FeatureParameters {
    int32_t min_init_tracks = 24;
    int32_t max_features = 800;
    int32_t max_init_tracks = 120;
    int32_t refresh_grid_rows = 4;
    int32_t refresh_grid_cols = 6;
    int32_t klt_init_window_size = 21;
    int32_t klt_init_pyramid_levels = 3;
    int32_t klt_window_size = 31;
    int32_t klt_pyramid_levels = 4;
    int32_t klt_max_iterations = 30;
    double klt_quality = 0.01;
    double klt_min_distance = 12.0;
    double klt_epsilon = 0.01;
    double klt_min_eig_threshold = 1.0e-4;
    double max_forward_backward_error = 1.0;
    double forward_backward_motion_ratio = 0.05;
    double max_adaptive_forward_backward_error = 2.0;
};

struct PnpParameters {
    int32_t min_tracks = 6;
    int32_t min_stable_inliers = 20;
    int32_t ransac_iterations = 80;
    int32_t ransac_sample_size = 8;
    double reprojection_inlier_threshold = 5.0;
    double max_reprojection_p90 = 5.0;
};

struct InitializerParameters {
    int32_t min_tracks = 24;
    int32_t min_map_points = 6;
    int32_t ransac_max_iters = 1000;
    double fundamental_inlier_threshold = 0.1;
    double homography_inlier_threshold = 5.0;
    double homography_model_ratio = 0.45;
    double min_parallax_deg = 0.3;
    double max_triangulation_p90 = 3.0;
};

struct MappingParameters {
    int32_t target_tracked_map_points = 100;
    int32_t max_refresh_candidates = 400;
    int32_t max_active_age = 60;
    int32_t max_unseen_frames = 2;
    int32_t min_pnp_track_length = 2;
    int32_t candidate_min_track_length = 3;
    double max_point_reprojection_error = 5.0;
    double min_refresh_parallax_deg = 0.2;
    double max_triangulation_p90 = 3.0;
};

struct BundleAdjustmentParameters {
    int32_t max_points = 120;
    int32_t min_points = 12;
    int32_t max_iterations = 20;
    int32_t solver = 0;
    double loss_scale = 3.0;
    double min_baseline = 1.0e-6;
    double max_anchor_scale_change = 2.0;
    double max_cost_growth = 1.05;
    double max_reprojection_p90 = 3.0;
};

struct LoopClosureParameters {
    int32_t orb_features = 1000;
    int32_t recent_exclusion = 100;
    int32_t min_matches = 30;
    int32_t min_inliers = 25;
    int32_t ransac_max_iters = 500;
    int32_t min_consecutive_detections = 3;
    int32_t consistency_window = 30;
    int32_t metric_neighbor_gap = 8;
    int32_t metric_min_inliers = 12;
    int32_t metric_ransac_iters = 200;
    double metric_inlier_ratio = 0.05;
    double metric_max_scale_ratio = 3.0;
    int32_t metric_required = 1;
    double metric_min_parallax_deg = 0.3;
    double metric_max_reprojection_error = 3.0;
    int32_t duplicate_frame_gap = 50;
    int32_t duplicate_match_window = 20;
    double duplicate_distance = 2.0;
    double min_score = 0.05;
    double match_ratio = 0.75;
    double inlier_threshold = 2.0;
    double max_rotation_error = 1.0e-6;
    int32_t pgo_enabled = 1;
    int32_t pgo_max_graph_poses = 150;
    int32_t pgo_max_iterations = 50;
    int32_t pgo_episode_end_gap = 30;
    int32_t pgo_pending_trigger = 3;
    int32_t pgo_loss_type = 0;
    double pgo_loss_scale = 1.0;
    double pgo_loop_translation_weight = 1.0;
    double pgo_loop_rotation_weight = 1.0;
    double pgo_loop_scale_weight = 1.0;
    double pgo_scale_weight = 1.0;
    double pgo_max_scale_change = 3.0;
    int32_t gba_enabled = 1;
    int32_t gba_max_cameras = 150;
    int32_t gba_max_points = 3000;
    int32_t gba_min_observations = 2;
    int32_t gba_max_loop_points = 100;
    int32_t gba_max_iterations = 60;
    double gba_loss_scale = 3.0;
};

struct StereoParameters {
    double max_epipolar_error = 1.5;
    double min_disparity = 1.0;
    double max_disparity = 150.0;
    double min_depth = 1.0;
    double max_depth = 90.0;
    int32_t min_init_points = 40;
    double max_rotation_error = 1.0e-3;
    int32_t ba_jacobian_mode = 0;
    int32_t local_ba_enabled = 1;
    int32_t local_ba_interval = 10;
    int32_t local_ba_window = 10;
    int32_t local_ba_max_points = 350;
    int32_t local_ba_min_observations = 4;
    int32_t local_ba_min_camera_observations = 6;
    int32_t local_ba_max_iterations = 15;
    int32_t local_ba_stereo_rows = 0;
    double local_ba_loss_scale = 3.0;
    int32_t full_ba_enabled = 1;
    int32_t full_ba_max_cameras = 200;
    int32_t full_ba_max_points = 5000;
    int32_t full_ba_min_observations = 3;
    int32_t full_ba_min_camera_observations = 6;
    int32_t full_ba_max_iterations = 40;
    int32_t full_ba_stereo_rows = 1;
    double full_ba_loss_scale = 3.0;
};

// Windowed local bundle adjustment for the monocular path. The window holds
// the most recent poses; the first (oldest) surviving camera is fixed as the
// gauge anchor so the refinement stays continuous with the published
// trajectory. No stereo rows and no metric scale are involved.
struct MonoLocalBaParameters {
    int32_t enabled = 1;
    int32_t window = 10;
    int32_t interval = 5;
    int32_t max_points = 250;
    int32_t min_observations = 3;
    int32_t min_camera_observations = 3;
    int32_t max_iterations = 10;
    double loss_scale = 3.0;
    int32_t solver = 1;
};

struct VisualizationParameters {
    float previous_map_point_radius = 0.08F;
    float current_map_point_radius = 0.18F;
    float trajectory_radius = 0.22F;
    float klt_track_radius = 3.0F;
    float loop_edge_radius = 0.3F;
    float optimized_trajectory_radius = 0.22F;
    // Growing point sets (trajectory, historical map points) are logged as
    // closed chunks of this size plus a re-logged head, so per-frame logging
    // cost stays bounded instead of growing with the sequence.
    int32_t log_chunk_size = 2048;
};

struct MvoParameters {
    FeatureParameters feature;
    PnpParameters pnp;
    InitializerParameters initializer;
    MappingParameters mapping;
    BundleAdjustmentParameters bundle_adjustment;
    LoopClosureParameters loop_closure;
    StereoParameters stereo;
    MonoLocalBaParameters mono_local_ba;
    VisualizationParameters visualization;
};

bool load_parameter_configs(const std::string& directory,
                            MvoParameters* parameters);

}  // namespace mvo
