#pragma once

#include "config.h"

#include <opencv2/core.hpp>

#include <cstdint>
#include <memory>
#include <vector>

#ifndef MVO_HAS_RERUN
#define MVO_HAS_RERUN 0
#endif

#if MVO_HAS_RERUN
#include <rerun.hpp>
#endif

namespace mvo {

struct Visualizer {
    bool enabled = false;
    VisualizationParameters parameters;
    std::vector<cv::Point3f> trajectory;
#if MVO_HAS_RERUN
    std::unique_ptr<rerun::RecordingStream> rec;
#endif
};

bool initialize_visualizer(const AppConfig& config, Visualizer* visualizer);
void log_visualization(Visualizer* visualizer, int32_t frame_id,
                       const cv::Mat& image,
                       const std::vector<cv::Point2f>& current_points,
                       const std::vector<cv::Point3f>& current_map_points,
                       const std::vector<cv::Point3f>& all_map_points,
                       const Pose& pose);
void flush_visualizer(Visualizer* visualizer);
cv::Point3f camera_center_from_pose(const Pose& pose);
double median_parallax_deg(const std::vector<cv::Point3f>& map_points,
                           const Pose& pose0,
                           const Pose& pose1);
double parallax_deg_for_point(const cv::Point3f& point,
                              const Pose& pose0,
                              const Pose& pose1);

}  // namespace mvo
