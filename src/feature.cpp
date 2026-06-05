#include "feature.h"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace mvo {

bool detect_initial_points(const cv::Mat& image,
                           const FeatureParameters& parameters,
                           std::vector<cv::Point2f>* points) {
    bool ok = false;
    cv::goodFeaturesToTrack(image, *points, parameters.max_features,
                            parameters.klt_quality,
                            parameters.klt_min_distance);
    ok = static_cast<int32_t>(points->size()) >=
         parameters.min_init_tracks;
    return ok;
}

bool point_is_far_from_existing(const cv::Point2f& point,
                                const std::vector<cv::Point2f>& existing,
                                double min_distance) {
    bool far = true;
    const double min_dist_sq = min_distance * min_distance;
    for (const cv::Point2f& other : existing) {
        const double dx = static_cast<double>(point.x - other.x);
        const double dy = static_cast<double>(point.y - other.y);
        if (dx * dx + dy * dy < min_dist_sq) {
            far = false;
            break;
        }
    }
    return far;
}

bool detect_refresh_points(const cv::Mat& image,
                           const std::vector<cv::Point2f>& existing,
                           int32_t max_points,
                           const FeatureParameters& parameters,
                           std::vector<cv::Point2f>* points) {
    bool ok = false;
    points->clear();
    cv::Mat mask(image.size(), CV_8UC1, cv::Scalar(255));
    std::vector<cv::Point2f> occupied = existing;
    for (const cv::Point2f& point : existing) {
        cv::circle(mask, point,
                   static_cast<int32_t>(parameters.klt_min_distance),
                   cv::Scalar(0), cv::FILLED);
    }

    const int32_t grid_cells = parameters.refresh_grid_rows *
                               parameters.refresh_grid_cols;
    const int32_t max_per_cell = std::max(
        1, (max_points + grid_cells - 1) / grid_cells);
    for (int32_t row = 0; row < parameters.refresh_grid_rows &&
                          static_cast<int32_t>(points->size()) < max_points;
         ++row) {
        for (int32_t col = 0; col < parameters.refresh_grid_cols &&
                              static_cast<int32_t>(points->size()) <
                                  max_points;
             ++col) {
            const int32_t x0 = col * image.cols /
                               parameters.refresh_grid_cols;
            const int32_t x1 = (col + 1) * image.cols /
                               parameters.refresh_grid_cols;
            const int32_t y0 = row * image.rows /
                               parameters.refresh_grid_rows;
            const int32_t y1 = (row + 1) * image.rows /
                               parameters.refresh_grid_rows;
            const cv::Rect cell_rect(x0, y0, x1 - x0, y1 - y0);
            if (cell_rect.width > 0 && cell_rect.height > 0) {
                std::vector<cv::Point2f> cell_points;
                cv::goodFeaturesToTrack(image(cell_rect), cell_points,
                                        max_per_cell, parameters.klt_quality,
                                        parameters.klt_min_distance,
                                        mask(cell_rect));
                for (const cv::Point2f& cell_point : cell_points) {
                    const cv::Point2f point(
                        cell_point.x + static_cast<float>(x0),
                        cell_point.y + static_cast<float>(y0));
                    if (static_cast<int32_t>(points->size()) < max_points &&
                        point_is_far_from_existing(point, occupied,
                                                   parameters.klt_min_distance)) {
                        points->push_back(point);
                        occupied.push_back(point);
                        cv::circle(mask, point,
                                   static_cast<int32_t>(
                                       parameters.klt_min_distance),
                                   cv::Scalar(0), cv::FILLED);
                    }
                }
            }
        }
    }

    if (static_cast<int32_t>(points->size()) < max_points) {
        std::vector<cv::Point2f> extra_points;
        cv::goodFeaturesToTrack(image, extra_points,
                                max_points -
                                    static_cast<int32_t>(points->size()),
                                parameters.klt_quality,
                                parameters.klt_min_distance, mask);
        for (const cv::Point2f& point : extra_points) {
            if (static_cast<int32_t>(points->size()) < max_points &&
                point_is_far_from_existing(point, occupied,
                                           parameters.klt_min_distance)) {
                points->push_back(point);
                occupied.push_back(point);
            }
        }
    }

    ok = !points->empty();
    return ok;
}

double adaptive_forward_backward_threshold(
    const std::vector<cv::Point2f>& prev_points,
    const std::vector<cv::Point2f>& next_points,
    const std::vector<uchar>& status,
    const FeatureParameters& parameters,
    const cv::Size& image_size) {
    double threshold = parameters.max_forward_backward_error;
    std::vector<double> motion_pixels;
    for (int32_t i = 0; i < static_cast<int32_t>(status.size()); ++i) {
        const bool in_bounds =
            next_points[i].x >= 0.0F &&
            next_points[i].x < static_cast<float>(image_size.width) &&
            next_points[i].y >= 0.0F &&
            next_points[i].y < static_cast<float>(image_size.height);
        if (status[i] != 0 && in_bounds) {
            const double dx = static_cast<double>(
                next_points[i].x - prev_points[i].x);
            const double dy = static_cast<double>(
                next_points[i].y - prev_points[i].y);
            motion_pixels.push_back(std::sqrt(dx * dx + dy * dy));
        }
    }
    if (!motion_pixels.empty()) {
        const size_t median_index = motion_pixels.size() / 2U;
        std::nth_element(motion_pixels.begin(),
                         motion_pixels.begin() + median_index,
                         motion_pixels.end());
        const double motion_threshold =
            motion_pixels[median_index] *
            parameters.forward_backward_motion_ratio;
        threshold = std::max(parameters.max_forward_backward_error,
                             std::min(
                                      parameters.max_adaptive_forward_backward_error,
                                      motion_threshold));
    }
    return threshold;
}

struct KltPassResult {
    std::vector<cv::Point2f> tracked_prev;
    std::vector<cv::Point2f> tracked_next;
    std::vector<int32_t> tracked_indices;
    int32_t raw_kept = 0;
    double fb_threshold = 1.0;
    int32_t window_size = 21;
    int32_t pyramid_levels = 3;
    bool used_wide_search = false;
};

void run_klt_pass(const cv::Mat& prev_image,
                  const cv::Mat& image,
                  const std::vector<cv::Point2f>& prev_points,
                  const FeatureParameters& parameters,
                  int32_t window_size,
                  int32_t pyramid_levels,
                  bool adaptive_fb,
                  KltPassResult* result) {
    result->tracked_prev.clear();
    result->tracked_next.clear();
    result->tracked_indices.clear();
    result->raw_kept = 0;
    result->fb_threshold = parameters.max_forward_backward_error;
    result->window_size = window_size;
    result->pyramid_levels = pyramid_levels;
    result->used_wide_search = adaptive_fb;

    std::vector<uchar> status;
    std::vector<uchar> backward_status;
    std::vector<float> err;
    std::vector<float> backward_err;
    std::vector<cv::Point2f> next_points;
    std::vector<cv::Point2f> backward_points;
    if (!prev_points.empty()) {
        const cv::Size lk_window(window_size, window_size);
        const cv::TermCriteria lk_criteria(
            cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
            parameters.klt_max_iterations, parameters.klt_epsilon);
        cv::calcOpticalFlowPyrLK(prev_image, image, prev_points, next_points,
                                 status, err, lk_window, pyramid_levels,
                                 lk_criteria, 0,
                                 parameters.klt_min_eig_threshold);
        cv::calcOpticalFlowPyrLK(image, prev_image, next_points,
                                 backward_points, backward_status,
                                 backward_err, lk_window, pyramid_levels,
                                 lk_criteria, 0,
                                 parameters.klt_min_eig_threshold);
        if (adaptive_fb) {
            result->fb_threshold = adaptive_forward_backward_threshold(
                prev_points, next_points, status, parameters, image.size());
        }
        for (int32_t i = 0; i < static_cast<int32_t>(status.size()); ++i) {
            if (status[i] != 0) {
                ++result->raw_kept;
            }
            const bool in_bounds =
                next_points[i].x >= 0.0F &&
                next_points[i].x < static_cast<float>(image.cols) &&
                next_points[i].y >= 0.0F &&
                next_points[i].y < static_cast<float>(image.rows);
            const double dx = static_cast<double>(
                backward_points[i].x - prev_points[i].x);
            const double dy = static_cast<double>(
                backward_points[i].y - prev_points[i].y);
            const double fb_error = std::sqrt(dx * dx + dy * dy);
            if (status[i] != 0 && backward_status[i] != 0 && in_bounds &&
                fb_error <= result->fb_threshold) {
                result->tracked_prev.push_back(prev_points[i]);
                result->tracked_next.push_back(next_points[i]);
                result->tracked_indices.push_back(i);
            }
        }
        if (static_cast<int32_t>(result->tracked_next.size()) >
            parameters.max_init_tracks) {
            result->tracked_prev.resize(parameters.max_init_tracks);
            result->tracked_next.resize(parameters.max_init_tracks);
            result->tracked_indices.resize(parameters.max_init_tracks);
        }
    }
}

bool track_points(const cv::Mat& prev_image, const cv::Mat& image,
                  const std::vector<cv::Point2f>& prev_points,
                  const FeatureParameters& parameters,
                  std::vector<cv::Point2f>* tracked_prev,
                  std::vector<cv::Point2f>* tracked_next,
                  std::vector<int32_t>* tracked_indices,
                  bool debug_geometry,
                  const std::string& tag,
                  bool wide_search) {
    bool ok = false;
    KltPassResult result;
    run_klt_pass(prev_image, image, prev_points, parameters,
                 parameters.klt_init_window_size,
                 parameters.klt_init_pyramid_levels, false, &result);
    const int32_t retry_threshold = std::min(
        parameters.min_init_tracks, static_cast<int32_t>(prev_points.size()));
    if (wide_search &&
        static_cast<int32_t>(result.tracked_next.size()) < retry_threshold) {
        KltPassResult retry_result;
        run_klt_pass(prev_image, image, prev_points, parameters,
                     parameters.klt_window_size,
                     parameters.klt_pyramid_levels, true, &retry_result);
        if (retry_result.tracked_next.size() > result.tracked_next.size()) {
            result = retry_result;
        }
    }

    *tracked_prev = result.tracked_prev;
    *tracked_next = result.tracked_next;
    if (tracked_indices != nullptr) {
        *tracked_indices = result.tracked_indices;
    }
    if (debug_geometry) {
        std::cout << "klt_debug tag=" << tag
                  << " input=" << prev_points.size()
                  << " status=" << result.raw_kept
                  << " fb_kept=" << tracked_next->size()
                  << " fb_thresh=" << result.fb_threshold
                  << " lk_window=" << result.window_size
                  << " lk_levels=" << result.pyramid_levels
                  << " lk_retry=" << result.used_wide_search
                  << std::endl;
    }
    ok = static_cast<int32_t>(tracked_next->size()) >=
         parameters.min_init_tracks;
    return ok;
}

}  // namespace mvo
