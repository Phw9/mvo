// Native iterative Lucas-Kanade point tracking.

#ifndef CVLIB_FEATURE2D_KLT_H_
#define CVLIB_FEATURE2D_KLT_H_

#include "defs.h"
#include "error_codes.h"

#include <cstdint>

namespace cvlib {
namespace feature2d {

/*
Read-only grayscale image view.

@param data Row-major grayscale samples.
@param rows Image row count.
@param cols Image column count.
@param stride Elements between consecutive rows.
*/
struct KltImageView {
    const float64_t* data = nullptr;
    int32_t rows = 0;
    int32_t cols = 0;
    int32_t stride = 0;
};

/*
Read-only float32 grayscale image view.

@param data Row-major grayscale samples.
@param rows Image row count.
@param cols Image column count.
@param stride Elements between consecutive rows.
*/
struct KltImageViewF32 {
    const float32_t* data = nullptr;
    int32_t rows = 0;
    int32_t cols = 0;
    int32_t stride = 0;
};

/*
Two-dimensional feature point.

@param x Horizontal pixel coordinate.
@param y Vertical pixel coordinate.
*/
struct KltPoint {
    float64_t x = 0.0;
    float64_t y = 0.0;
};

/*
KLT tracker parameters.

@param window_width Tracking window width in pixels.
@param window_height Tracking window height in pixels.
@param max_level Pyramid level hint for API compatibility.
@param max_iterations Maximum Gauss-Newton iterations.
@param epsilon Convergence threshold in pixels.
@param min_eig_threshold Minimum structure-tensor eigenvalue.
@param fallback_search_radius Integer fallback search radius.
@param fallback_error_threshold Maximum fallback patch error.
*/
struct KltParameters {
    int32_t window_width = 21;
    int32_t window_height = 21;
    int32_t max_level = 3;
    int32_t max_iterations = 30;
    float64_t epsilon = 0.01;
    float64_t min_eig_threshold = 1.0e-4;
    int32_t fallback_search_radius = 0;
    float64_t fallback_error_threshold = 5.0;
};

/*
Returns OpenCV-compatible default tracker parameters.

@returns KltParameters.
*/
KltParameters klt_default_parameters();

/*
Tracks points from prev_image into next_image.

@param prev_image Previous grayscale image.
@param next_image Current grayscale image.
@param prev_points Input points in prev_image.
@param point_count Number of points to track.
@param parameters Tracker parameters.
@param next_points Output points in next_image.
@param status Output track status values, 1 for success.
@param errors Optional output mean absolute patch errors.
@returns ErrorCode.
*/
ErrorCode klt_track(const KltImageView* prev_image,
                    const KltImageView* next_image,
                    const KltPoint* prev_points,
                    int32_t point_count,
                    const KltParameters* parameters,
                    KltPoint* next_points,
                    uint8_t* status,
                    float64_t* errors = nullptr);

/*
Tracks points from float32 prev_image into float32 next_image.

@param prev_image Previous grayscale image.
@param next_image Current grayscale image.
@param prev_points Input points in prev_image.
@param point_count Number of points to track.
@param parameters Tracker parameters.
@param next_points Output points in next_image.
@param status Output track status values, 1 for success.
@param errors Optional output mean absolute patch errors.
@returns ErrorCode.
*/
ErrorCode klt_track_f32(const KltImageViewF32* prev_image,
                        const KltImageViewF32* next_image,
                        const KltPoint* prev_points,
                        int32_t point_count,
                        const KltParameters* parameters,
                        KltPoint* next_points,
                        uint8_t* status,
                        float64_t* errors = nullptr);

}  // namespace feature2d
}  // namespace cvlib

#endif  // CVLIB_FEATURE2D_KLT_H_
