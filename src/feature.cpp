#include "feature.h"

#include "feature2d/features.h"
#include "feature2d/klt.h"
#include "image/image.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace mvo {
namespace {

cv::Mat to_grayscale(const cv::Mat& image) {
    cv::Mat gray;
    if (image.channels() == 1) {
        gray = image;
    } else {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }
    return gray;
}

cvlib::image::ImageView make_cvlib_feature_image_view(
    const cv::Mat& image,
    std::vector<cvlib::float64_t>* pixels) {
    cvlib::image::ImageView view;
    view.type = cvlib::k64FC1;
    pixels->clear();
    if (!image.empty()) {
        cv::Mat gray64;
        to_grayscale(image).convertTo(gray64, CV_64F);
        const double* data = gray64.ptr<double>(0);
        pixels->assign(data, data + gray64.total());
        view.data = pixels->data();
        view.rows = gray64.rows;
        view.cols = gray64.cols;
        view.stride = gray64.cols;
    }
    return view;
}

cvlib::image::ImageView make_cvlib_feature_mask_view(
    const cv::Mat& mask) {
    cvlib::image::ImageView view;
    view.type = cvlib::k8UC1;
    if (!mask.empty()) {
        view.data = mask.ptr<uint8_t>(0);
        view.rows = mask.rows;
        view.cols = mask.cols;
        view.stride = static_cast<int32_t>(mask.step1());
    }
    return view;
}

bool detect_good_features_with_cvlib(const cv::Mat& image,
                                     int32_t max_points,
                                     double quality,
                                     double min_distance,
                                     const cv::Mat* mask,
                                     std::vector<cv::Point2f>* points) {
    bool ok = false;
    if (points != nullptr) {
        points->clear();
    }
    if (points != nullptr && max_points > 0 && !image.empty()) {
        std::vector<cvlib::float64_t> image_pixels;
        const cvlib::image::ImageView image_view =
            make_cvlib_feature_image_view(image, &image_pixels);
        cvlib::image::ImageView mask_view;
        const cvlib::image::ImageView* mask_view_ptr = nullptr;
        if (mask != nullptr && !mask->empty()) {
            mask_view = make_cvlib_feature_mask_view(*mask);
            mask_view_ptr = &mask_view;
        }
        cvlib::feature2d::GoodFeaturesToTrackParameters cvlib_parameters =
            cvlib::feature2d::good_features_to_track_default_parameters();
        cvlib_parameters.max_corners = max_points;
        cvlib_parameters.quality_level = quality;
        cvlib_parameters.min_distance = min_distance;
        std::vector<cvlib::feature2d::Keypoint> keypoints;
        const cvlib::ErrorCode ec =
            cvlib::feature2d::good_features_to_track(
                &image_view, mask_view_ptr, &cvlib_parameters, &keypoints);
        if (ec == cvlib::ErrorCode::kSuccess) {
            points->reserve(keypoints.size());
            for (const cvlib::feature2d::Keypoint& keypoint : keypoints) {
                points->push_back(cv::Point2f(
                    static_cast<float>(keypoint.x),
                    static_cast<float>(keypoint.y)));
            }
            ok = !points->empty();
        }
    }
    return ok;
}

void make_existing_mask(const cv::Mat& image,
                        const std::vector<cv::Point2f>& existing,
                        double min_distance,
                        cv::Mat* mask) {
    *mask = cv::Mat(image.size(), CV_8UC1, cv::Scalar(255));
    for (const cv::Point2f& point : existing) {
        cv::circle(*mask, point, static_cast<int32_t>(min_distance),
                   cv::Scalar(0), cv::FILLED);
    }
}

cvlib::image::ImageView make_cvlib_klt_image_view(
    const cv::Mat& image,
    std::vector<cvlib::float32_t>* pixels) {
    cvlib::image::ImageView view;
    view.type = cvlib::k32FC1;
    pixels->clear();
    if (!image.empty()) {
        cv::Mat gray32;
        to_grayscale(image).convertTo(gray32, CV_32F);
        const float* data = gray32.ptr<float>(0);
        pixels->assign(data, data + gray32.total());
        view.data = pixels->data();
        view.rows = gray32.rows;
        view.cols = gray32.cols;
        view.stride = gray32.cols;
    }
    return view;
}

cvlib::feature2d::KltParameters make_cvlib_klt_parameters(
    const FeatureParameters& parameters,
    int32_t window_size,
    int32_t pyramid_levels) {
    cvlib::feature2d::KltParameters klt_parameters =
        cvlib::feature2d::klt_default_parameters();
    klt_parameters.window_width = window_size;
    klt_parameters.window_height = window_size;
    klt_parameters.max_level = pyramid_levels;
    klt_parameters.max_iterations = parameters.klt_max_iterations;
    klt_parameters.epsilon = parameters.klt_epsilon;
    klt_parameters.min_eig_threshold = parameters.klt_min_eig_threshold;
    return klt_parameters;
}

cvlib::ErrorCode track_points_with_cvlib_klt(
    const cvlib::image::ImageView& prev_view,
    const cvlib::image::ImageView& next_view,
    const std::vector<cv::Point2f>& prev_points,
    const cvlib::feature2d::KltParameters& klt_parameters,
    std::vector<cv::Point2f>* next_points,
    std::vector<uint8_t>* status) {
    cvlib::ErrorCode ec = cvlib::ErrorCode::kSuccess;
    next_points->clear();
    status->clear();
    if (!prev_points.empty()) {
        const int32_t point_count =
            static_cast<int32_t>(prev_points.size());
        cvlib::Matrix input_points = cvlib::matrix_create(point_count, 2);
        cvlib::Matrix output_points = cvlib::matrix_create(point_count, 2);
        std::vector<cvlib::float64_t> errors(prev_points.size());
        status->resize(prev_points.size(), 0U);
        if (input_points.data != nullptr && output_points.data != nullptr) {
            for (int32_t i = 0; i < point_count; ++i) {
                input_points(i, 0) = static_cast<double>(prev_points[i].x);
                input_points(i, 1) = static_cast<double>(prev_points[i].y);
            }
            ec = cvlib::feature2d::klt_track(
                &prev_view, &next_view, &input_points, &klt_parameters,
                &output_points, status->data(), errors.data());
            if (ec == cvlib::ErrorCode::kSuccess) {
                next_points->resize(prev_points.size());
                for (int32_t i = 0; i < point_count; ++i) {
                    (*next_points)[i] = cv::Point2f(
                        static_cast<float>(output_points(i, 0)),
                        static_cast<float>(output_points(i, 1)));
                }
            } else {
                status->clear();
            }
        } else {
            ec = cvlib::ErrorCode::kNullPointer;
            status->clear();
        }
        cvlib::matrix_destroy(&input_points);
        cvlib::matrix_destroy(&output_points);
    }
    return ec;
}

double adaptive_forward_backward_threshold(
    const std::vector<cv::Point2f>& prev_points,
    const std::vector<cv::Point2f>& next_points,
    const std::vector<uint8_t>& status,
    const FeatureParameters& parameters,
    const cv::Size& image_size) {
    double threshold = parameters.max_forward_backward_error;
    std::vector<double> motion_pixels;
    for (std::size_t i = 0; i < status.size(); ++i) {
        const bool in_bounds =
            next_points[i].x >= 0.0F &&
            next_points[i].x < static_cast<float>(image_size.width) &&
            next_points[i].y >= 0.0F &&
            next_points[i].y < static_cast<float>(image_size.height);
        if (status[i] != 0U && in_bounds) {
            const double dx = static_cast<double>(
                next_points[i].x - prev_points[i].x);
            const double dy = static_cast<double>(
                next_points[i].y - prev_points[i].y);
            motion_pixels.push_back(std::sqrt(dx * dx + dy * dy));
        }
    }
    if (!motion_pixels.empty()) {
        const std::size_t median_index = motion_pixels.size() / 2U;
        std::nth_element(motion_pixels.begin(),
                         motion_pixels.begin() + median_index,
                         motion_pixels.end());
        const double motion_threshold =
            motion_pixels[median_index] *
            parameters.forward_backward_motion_ratio;
        threshold = std::max(
            parameters.max_forward_backward_error,
            std::min(parameters.max_adaptive_forward_backward_error,
                     motion_threshold));
    }
    return threshold;
}

struct KltPassResult {
    std::vector<cv::Point2f> tracked_prev;
    std::vector<cv::Point2f> tracked_next;
    std::vector<int32_t> tracked_indices;
    int32_t raw_kept = 0;
    int32_t cvlib_error = 0;
    double fb_threshold = 1.0;
    int32_t window_size = 21;
    int32_t pyramid_levels = 3;
    bool used_wide_search = false;
};

void reset_klt_pass_result(const FeatureParameters& parameters,
                           int32_t window_size,
                           int32_t pyramid_levels,
                           bool adaptive_fb,
                           KltPassResult* result) {
    result->tracked_prev.clear();
    result->tracked_next.clear();
    result->tracked_indices.clear();
    result->raw_kept = 0;
    result->cvlib_error = 0;
    result->fb_threshold = parameters.max_forward_backward_error;
    result->window_size = window_size;
    result->pyramid_levels = pyramid_levels;
    result->used_wide_search = adaptive_fb;
}

void run_klt_pass(const cv::Mat& prev_image,
                  const cv::Mat& image,
                  const std::vector<cv::Point2f>& prev_points,
                  const FeatureParameters& parameters,
                  int32_t window_size,
                  int32_t pyramid_levels,
                  bool adaptive_fb,
                  KltPassResult* result) {
    reset_klt_pass_result(parameters, window_size, pyramid_levels, adaptive_fb,
                          result);
    if (prev_points.empty()) {
        return;
    }

    std::vector<cvlib::float32_t> prev_pixels;
    std::vector<cvlib::float32_t> next_pixels;
    const cvlib::image::ImageView prev_view =
        make_cvlib_klt_image_view(prev_image, &prev_pixels);
    const cvlib::image::ImageView next_view =
        make_cvlib_klt_image_view(image, &next_pixels);
    const cvlib::feature2d::KltParameters klt_parameters =
        make_cvlib_klt_parameters(parameters, window_size, pyramid_levels);

    std::vector<uint8_t> status;
    std::vector<uint8_t> backward_status;
    std::vector<cv::Point2f> next_points;
    std::vector<cv::Point2f> backward_points;
    cvlib::ErrorCode ec = track_points_with_cvlib_klt(
        prev_view, next_view, prev_points, klt_parameters, &next_points,
        &status);
    if (ec == cvlib::ErrorCode::kSuccess) {
        ec = track_points_with_cvlib_klt(
            next_view, prev_view, next_points, klt_parameters,
            &backward_points, &backward_status);
    }
    result->cvlib_error = static_cast<int32_t>(ec);
    if (ec == cvlib::ErrorCode::kSuccess) {
        if (adaptive_fb) {
            result->fb_threshold = adaptive_forward_backward_threshold(
                prev_points, next_points, status, parameters, image.size());
        }
        const std::size_t track_count =
            std::min(std::min(status.size(), backward_status.size()),
                     std::min(next_points.size(), backward_points.size()));
        for (std::size_t i = 0; i < track_count; ++i) {
            if (status[i] != 0U) {
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
            if (status[i] != 0U && backward_status[i] != 0U && in_bounds &&
                fb_error <= result->fb_threshold) {
                result->tracked_prev.push_back(prev_points[i]);
                result->tracked_next.push_back(next_points[i]);
                result->tracked_indices.push_back(static_cast<int32_t>(i));
            }
        }
    }
    if (static_cast<int32_t>(result->tracked_next.size()) >
        parameters.max_features) {
        result->tracked_prev.resize(parameters.max_features);
        result->tracked_next.resize(parameters.max_features);
        result->tracked_indices.resize(parameters.max_features);
    }
}

}  // namespace

bool detect_initial_points(const cv::Mat& image,
                           const FeatureParameters& parameters,
                           std::vector<cv::Point2f>* points) {
    bool ok = false;
    if (points != nullptr) {
        points->clear();
        ok = detect_good_features_with_cvlib(
            image, parameters.max_init_tracks, parameters.klt_quality,
            parameters.klt_min_distance, nullptr, points);
        ok = ok && static_cast<int32_t>(points->size()) >=
                       parameters.min_init_tracks;
    }
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
    if (points != nullptr && max_points > 0 && !image.empty()) {
        points->clear();
        cv::Mat mask;
        make_existing_mask(image, existing, parameters.klt_min_distance, &mask);
        std::vector<cv::Point2f> occupied = existing;
        const int32_t grid_cells =
            parameters.refresh_grid_rows * parameters.refresh_grid_cols;
        const int32_t max_per_cell =
            std::max(1, (max_points + grid_cells - 1) / grid_cells);
        for (int32_t row = 0;
             row < parameters.refresh_grid_rows &&
             static_cast<int32_t>(points->size()) < max_points;
             ++row) {
            for (int32_t col = 0;
                 col < parameters.refresh_grid_cols &&
                 static_cast<int32_t>(points->size()) < max_points;
                 ++col) {
                const int32_t x0 =
                    col * image.cols / parameters.refresh_grid_cols;
                const int32_t x1 =
                    (col + 1) * image.cols / parameters.refresh_grid_cols;
                const int32_t y0 =
                    row * image.rows / parameters.refresh_grid_rows;
                const int32_t y1 =
                    (row + 1) * image.rows / parameters.refresh_grid_rows;
                const cv::Rect cell_rect(x0, y0, x1 - x0, y1 - y0);
                if (cell_rect.width > 0 && cell_rect.height > 0) {
                    std::vector<cv::Point2f> cell_points;
                    const cv::Mat cell_image = image(cell_rect);
                    const cv::Mat cell_mask = mask(cell_rect);
                    detect_good_features_with_cvlib(
                        cell_image, max_per_cell, parameters.klt_quality,
                        parameters.klt_min_distance, &cell_mask,
                        &cell_points);
                    for (const cv::Point2f& cell_point : cell_points) {
                        const cv::Point2f point(
                            cell_point.x + static_cast<float>(x0),
                            cell_point.y + static_cast<float>(y0));
                        if (static_cast<int32_t>(points->size()) <
                                max_points &&
                            point_is_far_from_existing(
                                point, occupied,
                                parameters.klt_min_distance)) {
                            points->push_back(point);
                            occupied.push_back(point);
                            cv::circle(
                                mask, point,
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
            detect_good_features_with_cvlib(
                image, max_points - static_cast<int32_t>(points->size()),
                parameters.klt_quality, parameters.klt_min_distance, &mask,
                &extra_points);
            for (const cv::Point2f& point : extra_points) {
                if (static_cast<int32_t>(points->size()) < max_points &&
                    point_is_far_from_existing(
                        point, occupied, parameters.klt_min_distance)) {
                    points->push_back(point);
                    occupied.push_back(point);
                }
            }
        }
        ok = !points->empty();
    } else if (points != nullptr) {
        points->clear();
    }
    return ok;
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
    if (tracked_prev != nullptr) {
        tracked_prev->clear();
    }
    if (tracked_next != nullptr) {
        tracked_next->clear();
    }
    if (tracked_indices != nullptr) {
        tracked_indices->clear();
    }
    if (tracked_prev != nullptr && tracked_next != nullptr) {
        KltPassResult result;
        const int32_t window_size =
            wide_search ? parameters.klt_window_size
                        : parameters.klt_init_window_size;
        const int32_t pyramid_levels =
            wide_search ? parameters.klt_pyramid_levels
                        : parameters.klt_init_pyramid_levels;
        run_klt_pass(prev_image, image, prev_points, parameters, window_size,
                     pyramid_levels, wide_search, &result);

        *tracked_prev = std::move(result.tracked_prev);
        *tracked_next = std::move(result.tracked_next);
        if (tracked_indices != nullptr) {
            *tracked_indices = std::move(result.tracked_indices);
        }
        if (debug_geometry) {
            std::cout << "klt_debug tag=" << tag
                      << " input=" << prev_points.size()
                      << " status=" << result.raw_kept
                      << " fb_kept=" << tracked_next->size()
                      << " cvlib_error=" << result.cvlib_error
                      << " fb_thresh=" << result.fb_threshold
                      << " lk_window=" << result.window_size
                      << " lk_levels=" << result.pyramid_levels
                      << " lk_wide=" << result.used_wide_search
                      << std::endl;
        }
        ok = static_cast<int32_t>(tracked_next->size()) >=
             parameters.min_init_tracks;
    }
    return ok;
}

}  // namespace mvo
