#include "visualization.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>

namespace mvo {

bool visualizer_requested(const AppConfig& config) {
    bool requested = false;
    if (config.rerun_spawn || !config.rerun_save_path.empty()) {
        requested = true;
    }
    return requested;
}

bool initialize_visualizer(const AppConfig& config, Visualizer* visualizer) {
    bool ok = true;
    visualizer->enabled = false;
    visualizer->parameters = config.parameters.visualization;
    visualizer->trajectory.clear();
    if (visualizer_requested(config)) {
#if MVO_HAS_RERUN
        visualizer->rec = std::make_unique<rerun::RecordingStream>("mvo");
        if (!config.rerun_save_path.empty()) {
            visualizer->rec->save(config.rerun_save_path).exit_on_failure();
        } else {
            visualizer->rec->spawn().exit_on_failure();
        }
        visualizer->enabled = true;
        std::cout << "rerun=enabled mode="
                  << (config.rerun_save_path.empty() ? "spawn" : "save")
                  << " path=" << config.rerun_save_path << std::endl;
#else
        std::cout << "rerun=unavailable build_without_rerun_sdk"
                  << std::endl;
        ok = false;
#endif
    }
    return ok;
}

cv::Point3f camera_center_from_pose(const Pose& pose) {
    const float x = static_cast<float>(
        -(pose.r[0] * pose.t[0] + pose.r[3] * pose.t[1] +
          pose.r[6] * pose.t[2]));
    const float y = static_cast<float>(
        -(pose.r[1] * pose.t[0] + pose.r[4] * pose.t[1] +
          pose.r[7] * pose.t[2]));
    const float z = static_cast<float>(
        -(pose.r[2] * pose.t[0] + pose.r[5] * pose.t[1] +
          pose.r[8] * pose.t[2]));
    cv::Point3f center(x, y, z);
    return center;
}

double median_parallax_deg(const std::vector<cv::Point3f>& map_points,
                           const Pose& pose0,
                           const Pose& pose1) {
    double median = 0.0;
    std::vector<double> angles;
    const cv::Point3f c0 = camera_center_from_pose(pose0);
    const cv::Point3f c1 = camera_center_from_pose(pose1);
    angles.reserve(map_points.size());
    for (const cv::Point3f& point : map_points) {
        const double v0x = static_cast<double>(point.x - c0.x);
        const double v0y = static_cast<double>(point.y - c0.y);
        const double v0z = static_cast<double>(point.z - c0.z);
        const double v1x = static_cast<double>(point.x - c1.x);
        const double v1y = static_cast<double>(point.y - c1.y);
        const double v1z = static_cast<double>(point.z - c1.z);
        const double n0 = std::sqrt(v0x * v0x + v0y * v0y + v0z * v0z);
        const double n1 = std::sqrt(v1x * v1x + v1y * v1y + v1z * v1z);
        if (n0 > 1.0e-9 && n1 > 1.0e-9) {
            double cos_angle = (v0x * v1x + v0y * v1y + v0z * v1z) /
                               (n0 * n1);
            cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
            angles.push_back(std::acos(cos_angle) * 180.0 / kPi);
        }
    }
    if (!angles.empty()) {
        std::sort(angles.begin(), angles.end());
        median = angles[angles.size() / 2];
    }
    return median;
}

double parallax_deg_for_point(const cv::Point3f& point,
                              const Pose& pose0,
                              const Pose& pose1) {
    double angle = 0.0;
    const cv::Point3f c0 = camera_center_from_pose(pose0);
    const cv::Point3f c1 = camera_center_from_pose(pose1);
    const double v0x = static_cast<double>(point.x - c0.x);
    const double v0y = static_cast<double>(point.y - c0.y);
    const double v0z = static_cast<double>(point.z - c0.z);
    const double v1x = static_cast<double>(point.x - c1.x);
    const double v1y = static_cast<double>(point.y - c1.y);
    const double v1z = static_cast<double>(point.z - c1.z);
    const double n0 = std::sqrt(v0x * v0x + v0y * v0y + v0z * v0z);
    const double n1 = std::sqrt(v1x * v1x + v1y * v1y + v1z * v1z);
    if (n0 > 1.0e-9 && n1 > 1.0e-9) {
        double cos_angle = (v0x * v1x + v0y * v1y + v0z * v1z) / (n0 * n1);
        cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
        angle = std::acos(cos_angle) * 180.0 / kPi;
    }
    return angle;
}

#if MVO_HAS_RERUN
std::vector<uint8_t> image_bytes(const cv::Mat& image) {
    cv::Mat continuous = image;
    if (!image.isContinuous()) {
        continuous = image.clone();
    }
    std::vector<uint8_t> bytes(
        continuous.data,
        continuous.data + continuous.total() * continuous.elemSize());
    return bytes;
}

std::vector<rerun::Position2D> points2d_for_rerun(
    const std::vector<cv::Point2f>& points) {
    std::vector<rerun::Position2D> rerun_points;
    rerun_points.reserve(points.size());
    for (const cv::Point2f& point : points) {
        rerun_points.emplace_back(point.x, point.y);
    }
    return rerun_points;
}

std::vector<rerun::Position3D> points3d_for_rerun(
    const std::vector<cv::Point3f>& points) {
    std::vector<rerun::Position3D> rerun_points;
    rerun_points.reserve(points.size());
    for (const cv::Point3f& point : points) {
        rerun_points.emplace_back(point.x, point.y, point.z);
    }
    return rerun_points;
}
#endif

bool point3f_near(const cv::Point3f& a, const cv::Point3f& b) {
    const double dx = static_cast<double>(a.x - b.x);
    const double dy = static_cast<double>(a.y - b.y);
    const double dz = static_cast<double>(a.z - b.z);
    const bool near = dx * dx + dy * dy + dz * dz <= 1.0e-10;
    return near;
}

bool contains_point3f(const std::vector<cv::Point3f>& points,
                      const cv::Point3f& query) {
    bool contains = false;
    for (const cv::Point3f& point : points) {
        if (point3f_near(point, query)) {
            contains = true;
            break;
        }
    }
    return contains;
}

void collect_previous_map_points(
    const std::vector<cv::Point3f>& all_map_points,
    const std::vector<cv::Point3f>& current_map_points,
    std::vector<cv::Point3f>* previous_map_points) {
    previous_map_points->clear();
    previous_map_points->reserve(all_map_points.size());
    for (const cv::Point3f& point : all_map_points) {
        if (!contains_point3f(current_map_points, point)) {
            previous_map_points->push_back(point);
        }
    }
}

void log_visualization(Visualizer* visualizer, int32_t frame_id,
                       const cv::Mat& image,
                       const std::vector<cv::Point2f>& tracked_points,
                       const std::vector<cv::Point3f>& current_map_points,
                       const std::vector<cv::Point3f>& all_map_points,
                       const Pose& pose) {
    const cv::Point3f camera_center = camera_center_from_pose(pose);
    if (visualizer->enabled) {
        visualizer->trajectory.push_back(camera_center);
#if MVO_HAS_RERUN
        visualizer->rec->set_time_sequence("frame", frame_id);
        if (!image.empty()) {
            const std::vector<uint8_t> bytes = image_bytes(image);
            visualizer->rec->log(
                "camera/image",
                rerun::Image::from_grayscale8(
                    bytes, {static_cast<uint32_t>(image.cols),
                            static_cast<uint32_t>(image.rows)}));
        }
        if (!tracked_points.empty()) {
            visualizer->rec->log("camera/klt_tracks",
                                 rerun::Points2D(
                                     points2d_for_rerun(tracked_points))
                                     .with_radii(
                                         rerun::Radius::ui_points(
                                             visualizer->parameters
                                                 .klt_track_radius))
                                     .with_colors(rerun::Color(0, 255, 255)));
        }
        std::vector<cv::Point3f> previous_map_points;
        collect_previous_map_points(all_map_points, current_map_points,
                                    &previous_map_points);
        if (!previous_map_points.empty()) {
            visualizer->rec->log("world/previous_map_points",
                                 rerun::Points3D(
                                     points3d_for_rerun(previous_map_points))
                                     .with_radii(
                                         visualizer->parameters
                                             .previous_map_point_radius)
                                     .with_colors(rerun::Color(0, 0, 0)));
        }
        if (!current_map_points.empty()) {
            visualizer->rec->log("world/current_3d_keypoints",
                                 rerun::Points3D(
                                     points3d_for_rerun(current_map_points))
                                     .with_radii(
                                         visualizer->parameters
                                             .current_map_point_radius)
                                     .with_colors(rerun::Color(255, 0, 0)));
        }
        if (!visualizer->trajectory.empty()) {
            visualizer->rec->log("world/camera_trajectory",
                                 rerun::Points3D(
                                     points3d_for_rerun(
                                         visualizer->trajectory))
                                     .with_radii(
                                         visualizer->parameters
                                             .trajectory_radius)
                                     .with_colors(rerun::Color(0, 128, 255)));
        }
#else
        (void)frame_id;
        (void)image;
        (void)tracked_points;
        (void)current_map_points;
        (void)all_map_points;
#endif
    }
}

void flush_visualizer(Visualizer* visualizer) {
#if MVO_HAS_RERUN
    if (visualizer->enabled) {
        visualizer->rec->flush_blocking(5.0f).exit_on_failure();
    }
#else
    (void)visualizer;
#endif
}

}  // namespace mvo
