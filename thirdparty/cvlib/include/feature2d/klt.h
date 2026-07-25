// Native iterative Lucas-Kanade point tracking.

#ifndef CVLIB_FEATURE2D_KLT_H_
#define CVLIB_FEATURE2D_KLT_H_

#include "../types.h"
#include "../error_codes.h"
#include "../image/image.h"

#include <cstdint>

namespace cvlib {
namespace feature2d {

/*
KLT tracker parameters.

@param window_width Tracking window width in pixels.
@param window_height Tracking window height in pixels.
@param max_level Pyramid level count hint.
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
Returns default tracker parameters.

@returns KltParameters.
*/
KltParameters klt_default_parameters();

/*
Tracks points from prev_image into next_image. The two images must be
single-channel and share the same element depth (k64FC1 or k32FC1); the
tracker runs its f64 or f32 path accordingly.

@param prev_image Previous grayscale image (k64FC1 or k32FC1).
@param next_image Current grayscale image, same type and shape.
@param prev_points Input points, N-by-2 (k64FC1).
@param parameters Tracker parameters.
@param next_points Output points, N-by-2 (k64FC1); pre-created.
@param status Output track status values, length N, 1 for success.
@param errors Optional output mean absolute patch errors, length N.
@returns ErrorCode.
*/
ErrorCode klt_track(const image::ImageView* prev_image,
                    const image::ImageView* next_image,
                    const Matrix* prev_points,
                    const KltParameters* parameters,
                    Matrix* next_points,
                    uint8_t* status,
                    float64_t* errors = nullptr);

}  // namespace feature2d
}  // namespace cvlib

#endif  // CVLIB_FEATURE2D_KLT_H_
