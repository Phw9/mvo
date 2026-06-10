#include "feature.h"

#include "feature2d/features.h"
#include "feature2d/klt.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace mvo {
namespace {

bool frontend_is_superpoint(const FeatureParameters& parameters) {
    bool is_superpoint = false;
    if (parameters.frontend_mode == 1) {
        is_superpoint = true;
    }
    return is_superpoint;
}

const char* frontend_name(const FeatureParameters& parameters) {
    const char* name = "klt";
    if (parameters.frontend_mode == 1) {
        name = "superpoint_superglue";
    }
    return name;
}

bool superpoint_unavailable(const FeatureParameters& parameters,
                            const std::string& tag) {
    bool ok = false;
    std::cout << "frontend_unavailable tag=" << tag
              << " frontend_mode=" << parameters.frontend_mode
              << " frontend=" << frontend_name(parameters)
              << " reason=superpoint_superglue_runtime_not_configured"
              << " superpoint_model=" << parameters.superpoint_model
              << " superglue_model=" << parameters.superglue_model
              << std::endl;
    return ok;
}

cvlib::feature2d::FeatureImageView make_cvlib_feature_image_view(
    const cv::Mat& image,
    std::vector<cvlib::float64_t>* pixels) {
    cvlib::feature2d::FeatureImageView view;
    pixels->clear();
    if (!image.empty()) {
        cv::Mat gray;
        if (image.channels() == 1) {
            gray = image;
        } else {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        }
        cv::Mat gray64;
        gray.convertTo(gray64, CV_64F);
        pixels->resize(static_cast<std::size_t>(gray64.rows * gray64.cols));
        for (int32_t row = 0; row < gray64.rows; ++row) {
            const double* src = gray64.ptr<double>(row);
            for (int32_t col = 0; col < gray64.cols; ++col) {
                (*pixels)[static_cast<std::size_t>(
                    row * gray64.cols + col)] = src[col];
            }
        }
        view.data = pixels->data();
        view.rows = gray64.rows;
        view.cols = gray64.cols;
        view.stride = gray64.cols;
    }
    return view;
}

cvlib::feature2d::FeatureMaskView make_cvlib_feature_mask_view(
    const cv::Mat& mask) {
    cvlib::feature2d::FeatureMaskView view;
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
        const cvlib::feature2d::FeatureImageView image_view =
            make_cvlib_feature_image_view(image, &image_pixels);
        cvlib::feature2d::FeatureMaskView mask_view;
        const cvlib::feature2d::FeatureMaskView* mask_view_ptr = nullptr;
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

cvlib::feature2d::KltImageViewF32 make_cvlib_klt_image_view(
    const cv::Mat& image,
    std::vector<cvlib::float32_t>* pixels) {
    cvlib::feature2d::KltImageViewF32 view;
    pixels->clear();
    if (!image.empty()) {
        cv::Mat gray;
        if (image.channels() == 1) {
            gray = image;
        } else {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        }
        cv::Mat gray32;
        gray.convertTo(gray32, CV_32F);
        pixels->resize(static_cast<std::size_t>(gray32.rows * gray32.cols));
        for (int32_t row = 0; row < gray32.rows; ++row) {
            const float* src = gray32.ptr<float>(row);
            for (int32_t col = 0; col < gray32.cols; ++col) {
                (*pixels)[static_cast<std::size_t>(
                    row * gray32.cols + col)] = src[col];
            }
        }
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
    const cv::Mat& prev_image,
    const cv::Mat& image,
    const std::vector<cv::Point2f>& prev_points,
    const FeatureParameters& parameters,
    int32_t window_size,
    int32_t pyramid_levels,
    std::vector<cv::Point2f>* next_points,
    std::vector<uint8_t>* status) {
    cvlib::ErrorCode ec = cvlib::ErrorCode::kSuccess;
    next_points->clear();
    status->clear();
    if (!prev_points.empty()) {
        std::vector<cvlib::float32_t> prev_pixels;
        std::vector<cvlib::float32_t> image_pixels;
        const cvlib::feature2d::KltImageViewF32 prev_view =
            make_cvlib_klt_image_view(prev_image, &prev_pixels);
        const cvlib::feature2d::KltImageViewF32 image_view =
            make_cvlib_klt_image_view(image, &image_pixels);
        std::vector<cvlib::feature2d::KltPoint> input_points(
            prev_points.size());
        std::vector<cvlib::feature2d::KltPoint> output_points(
            prev_points.size());
        std::vector<cvlib::float64_t> errors(prev_points.size());
        status->resize(prev_points.size(), 0U);
        for (int32_t i = 0; i < static_cast<int32_t>(prev_points.size());
             ++i) {
            input_points[static_cast<std::size_t>(i)].x =
                static_cast<double>(prev_points[static_cast<std::size_t>(i)].x);
            input_points[static_cast<std::size_t>(i)].y =
                static_cast<double>(prev_points[static_cast<std::size_t>(i)].y);
        }
        const cvlib::feature2d::KltParameters klt_parameters =
            make_cvlib_klt_parameters(parameters, window_size,
                                      pyramid_levels);
        ec = cvlib::feature2d::klt_track_f32(
            &prev_view, &image_view, input_points.data(),
            static_cast<int32_t>(input_points.size()), &klt_parameters,
            output_points.data(), status->data(), errors.data());
        if (ec == cvlib::ErrorCode::kSuccess) {
            next_points->resize(output_points.size());
            for (int32_t i = 0;
                 i < static_cast<int32_t>(output_points.size()); ++i) {
                (*next_points)[static_cast<std::size_t>(i)] = cv::Point2f(
                    static_cast<float>(
                        output_points[static_cast<std::size_t>(i)].x),
                    static_cast<float>(
                        output_points[static_cast<std::size_t>(i)].y));
            }
        } else {
            status->clear();
        }
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
    for (int32_t i = 0; i < static_cast<int32_t>(status.size()); ++i) {
        const bool in_bounds =
            next_points[static_cast<std::size_t>(i)].x >= 0.0F &&
            next_points[static_cast<std::size_t>(i)].x <
                static_cast<float>(image_size.width) &&
            next_points[static_cast<std::size_t>(i)].y >= 0.0F &&
            next_points[static_cast<std::size_t>(i)].y <
                static_cast<float>(image_size.height);
        if (status[static_cast<std::size_t>(i)] != 0U && in_bounds) {
            const double dx = static_cast<double>(
                next_points[static_cast<std::size_t>(i)].x -
                prev_points[static_cast<std::size_t>(i)].x);
            const double dy = static_cast<double>(
                next_points[static_cast<std::size_t>(i)].y -
                prev_points[static_cast<std::size_t>(i)].y);
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

void make_direct_klt_inputs(
    const std::vector<cv::Point2f>& prev_points,
    std::vector<cv::Point2f>* tracking_prev_points,
    std::vector<int32_t>* original_indices) {
    tracking_prev_points->clear();
    original_indices->clear();
    tracking_prev_points->reserve(prev_points.size());
    original_indices->reserve(prev_points.size());
    for (int32_t i = 0; i < static_cast<int32_t>(prev_points.size()); ++i) {
        tracking_prev_points->push_back(
            prev_points[static_cast<std::size_t>(i)]);
        original_indices->push_back(i);
    }
}

void run_klt_from_tracking_points(
    const cv::Mat& tracking_prev_image,
    const cv::Mat& image,
    const std::vector<cv::Point2f>& original_prev_points,
    const std::vector<cv::Point2f>& tracking_prev_points,
    const std::vector<int32_t>& original_indices,
    const FeatureParameters& parameters,
    int32_t window_size,
    int32_t pyramid_levels,
    bool adaptive_fb,
    KltPassResult* result) {
    reset_klt_pass_result(parameters, window_size, pyramid_levels, adaptive_fb,
                          result);

    std::vector<uint8_t> status;
    std::vector<uint8_t> backward_status;
    std::vector<cv::Point2f> next_points;
    std::vector<cv::Point2f> backward_points;
    if (!tracking_prev_points.empty()) {
        cvlib::ErrorCode ec = track_points_with_cvlib_klt(
            tracking_prev_image, image, tracking_prev_points, parameters,
            window_size, pyramid_levels, &next_points, &status);
        if (ec == cvlib::ErrorCode::kSuccess) {
            ec = track_points_with_cvlib_klt(
                image, tracking_prev_image, next_points, parameters,
                window_size, pyramid_levels, &backward_points,
                &backward_status);
        }
        result->cvlib_error = static_cast<int32_t>(ec);
        if (ec == cvlib::ErrorCode::kSuccess) {
            if (adaptive_fb) {
                result->fb_threshold = adaptive_forward_backward_threshold(
                    tracking_prev_points, next_points, status, parameters,
                    image.size());
            }
            const int32_t track_count = std::min(
                std::min(static_cast<int32_t>(status.size()),
                         static_cast<int32_t>(backward_status.size())),
                std::min(static_cast<int32_t>(next_points.size()),
                         static_cast<int32_t>(backward_points.size())));
            for (int32_t i = 0; i < track_count; ++i) {
                const std::size_t index = static_cast<std::size_t>(i);
                if (status[index] != 0U) {
                    ++result->raw_kept;
                }
                const bool in_bounds =
                    next_points[index].x >= 0.0F &&
                    next_points[index].x < static_cast<float>(image.cols) &&
                    next_points[index].y >= 0.0F &&
                    next_points[index].y < static_cast<float>(image.rows);
                const double dx = static_cast<double>(
                    backward_points[index].x - tracking_prev_points[index].x);
                const double dy = static_cast<double>(
                    backward_points[index].y - tracking_prev_points[index].y);
                const double fb_error = std::sqrt(dx * dx + dy * dy);
                const bool original_index_ok =
                    original_indices[index] >= 0 &&
                    original_indices[index] <
                        static_cast<int32_t>(original_prev_points.size());
                if (status[index] != 0U && backward_status[index] != 0U &&
                    in_bounds && original_index_ok &&
                    fb_error <= result->fb_threshold) {
                    const int32_t original_index = original_indices[index];
                    result->tracked_prev.push_back(
                        original_prev_points[static_cast<std::size_t>(
                            original_index)]);
                    result->tracked_next.push_back(next_points[index]);
                    result->tracked_indices.push_back(original_index);
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
}

void run_klt_pass(const cv::Mat& prev_image,
                  const cv::Mat& image,
                  const std::vector<cv::Point2f>& prev_points,
                  const FeatureParameters& parameters,
                  int32_t window_size,
                  int32_t pyramid_levels,
                  bool adaptive_fb,
                  KltPassResult* result) {
    std::vector<cv::Point2f> direct_points;
    std::vector<int32_t> direct_indices;
    make_direct_klt_inputs(prev_points, &direct_points, &direct_indices);
    run_klt_from_tracking_points(prev_image, image, prev_points, direct_points,
                                 direct_indices, parameters, window_size,
                                 pyramid_levels, adaptive_fb, result);
}

}  // namespace

bool detect_initial_points(const cv::Mat& image,
                           const FeatureParameters& parameters,
                           std::vector<cv::Point2f>* points) {
    bool ok = false;
    if (frontend_is_superpoint(parameters)) {
        ok = superpoint_unavailable(parameters, "detect_initial");
    } else if (points != nullptr) {
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
    if (frontend_is_superpoint(parameters)) {
        ok = superpoint_unavailable(parameters, "detect_refresh");
    } else if (points != nullptr && max_points > 0 && !image.empty()) {
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
                  bool wide_search,
                  std::vector<cv::Mat>* tracked_descriptors) {
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
    if (tracked_descriptors != nullptr) {
        tracked_descriptors->clear();
    }
    if (frontend_is_superpoint(parameters)) {
        ok = superpoint_unavailable(parameters, tag);
    } else if (tracked_prev != nullptr && tracked_next != nullptr) {
        KltPassResult result;
        const int32_t window_size =
            wide_search ? parameters.klt_window_size
                        : parameters.klt_init_window_size;
        const int32_t pyramid_levels =
            wide_search ? parameters.klt_pyramid_levels
                        : parameters.klt_init_pyramid_levels;
        run_klt_pass(prev_image, image, prev_points, parameters, window_size,
                     pyramid_levels, wide_search, &result);

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
