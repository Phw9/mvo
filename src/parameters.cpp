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
        read_int_parameter(node, "min_tracked_map_points",
                           &parameters->min_tracked_map_points);
        read_int_parameter(node, "target_tracked_map_points",
                           &parameters->target_tracked_map_points);
        read_int_parameter(node, "max_refresh_candidates",
                           &parameters->max_refresh_candidates);
        read_int_parameter(node, "min_refresh_map_points",
                           &parameters->min_refresh_map_points);
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
        read_double_parameter(node, "loss_scale", &parameters->loss_scale);
        read_double_parameter(node, "min_baseline",
                              &parameters->min_baseline);
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
    parameters->initializer.min_tracks =
        std::max(8, parameters->initializer.min_tracks);
    parameters->initializer.min_map_points =
        std::max(4, parameters->initializer.min_map_points);
    parameters->mapping.min_tracked_map_points =
        std::max(1, parameters->mapping.min_tracked_map_points);
    parameters->mapping.target_tracked_map_points =
        std::max(parameters->mapping.min_tracked_map_points,
                 parameters->mapping.target_tracked_map_points);
    parameters->mapping.min_refresh_map_points =
        std::max(1, parameters->mapping.min_refresh_map_points);
    parameters->bundle_adjustment.max_points =
        std::max(1, parameters->bundle_adjustment.max_points);
    parameters->bundle_adjustment.min_points =
        std::max(1, parameters->bundle_adjustment.min_points);
    parameters->bundle_adjustment.max_iterations =
        std::max(1, parameters->bundle_adjustment.max_iterations);
    parameters->loop_closure.orb_features =
        std::max(1, parameters->loop_closure.orb_features);
    parameters->loop_closure.recent_exclusion =
        std::max(0, parameters->loop_closure.recent_exclusion);
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
