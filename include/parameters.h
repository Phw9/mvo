#pragma once

#include <cstdint>
#include <string>

namespace mvo {

struct FeatureParameters {
    int32_t frontend_mode = 0;
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
    int32_t orb_features = 1000;
    double orb_scale_factor = 1.2;
    int32_t orb_levels = 8;
    int32_t orb_edge_threshold = 31;
    int32_t orb_patch_size = 31;
    int32_t orb_fast_threshold = 20;
    double orb_min_distance = 8.0;
    double orb_match_ratio = 0.8;
    double orb_max_match_distance = 80.0;
    double orb_ransac_threshold = 2.0;
    double orb_projection_radius = 35.0;
    double orb_projection_fallback_radius = 90.0;
    std::string superpoint_model = "";
    std::string superglue_model = "";
};

struct PnpParameters {
    int32_t min_tracks = 6;
    int32_t min_stable_inliers = 20;
    int32_t orb_min_stable_inliers = 6;
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
    double orb_min_parallax_deg = 0.5;
    double orb_max_triangulation_p90 = 8.0;
};

struct MappingParameters {
    int32_t min_tracked_map_points = 45;
    int32_t target_tracked_map_points = 100;
    int32_t max_refresh_candidates = 400;
    int32_t min_refresh_map_points = 10;
    int32_t orb_min_refresh_map_points = 24;
    int32_t aggressive_target_tracked_map_points = 180;
    int32_t aggressive_max_refresh_candidates = 700;
    int32_t aggressive_min_refresh_map_points = 80;
    int32_t max_active_age = 60;
    int32_t max_unseen_frames = 2;
    int32_t min_pnp_track_length = 2;
    int32_t candidate_min_track_length = 3;
    double max_point_reprojection_error = 5.0;
    double min_refresh_parallax_deg = 0.2;
    double max_triangulation_p90 = 3.0;
    double orb_max_triangulation_p90 = 6.0;
    double aggressive_translation = 1.5;
    double aggressive_rotation_deg = 3.0;
};

struct BundleAdjustmentParameters {
    int32_t max_points = 120;
    int32_t min_points = 12;
    int32_t max_iterations = 20;
    double loss_scale = 3.0;
    double min_baseline = 1.0e-6;
    double max_cost_growth = 1.05;
    double max_reprojection_p90 = 3.0;
};

struct LoopClosureParameters {
    int32_t orb_features = 1000;
    int32_t recent_exclusion = 5;
};

struct VisualizationParameters {
    float previous_map_point_radius = 0.08F;
    float current_map_point_radius = 0.18F;
    float trajectory_radius = 0.22F;
    float klt_track_radius = 3.0F;
};

struct MvoParameters {
    FeatureParameters feature;
    PnpParameters pnp;
    InitializerParameters initializer;
    MappingParameters mapping;
    BundleAdjustmentParameters bundle_adjustment;
    LoopClosureParameters loop_closure;
    VisualizationParameters visualization;
};

bool load_parameter_configs(const std::string& directory,
                            MvoParameters* parameters);

}  // namespace mvo
