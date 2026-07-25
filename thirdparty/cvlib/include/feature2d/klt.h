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
@param use_initial_flow When true, each search starts from the caller's
       next_points entry (a motion prediction) instead of the previous
       position; next_points is then read as input as well as written.
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
    bool use_initial_flow = false;
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

/*
Owning, self-contained image pyramid for reuse across frames. Build once
with klt_build_pyramid and release with klt_pyramid_destroy; a video loop
can then reuse the previous frame's pyramid instead of rebuilding it.
*/
struct KltPyramid {
    void* impl = nullptr;
};

/*
Builds a multi-resolution pyramid from an image, copying the base level so
the result is independent of the source buffer's lifetime.

@param image Grayscale image (k64FC1 or k32FC1).
@param parameters Tracker parameters (window sizes and max_level set the
       level count).
@param out Output pyramid; overwritten on success (release with
       klt_pyramid_destroy).
@returns ErrorCode.
*/
ErrorCode klt_build_pyramid(const image::ImageView* image,
                            const KltParameters* parameters,
                            KltPyramid* out);

// Releases pyramid storage and resets the handle; safe on an empty handle.
void klt_pyramid_destroy(KltPyramid* pyramid);

/*
Tracks points using two prebuilt pyramids instead of images, avoiding a
per-call pyramid rebuild. The pyramids must share the element depth and
have been built with the same window/level parameters used here.

@param prev_pyramid Pyramid of the previous frame.
@param next_pyramid Pyramid of the current frame.
@param prev_points Input points, N-by-2 (k64FC1).
@param parameters Tracker parameters.
@param next_points Output points, N-by-2 (k64FC1); pre-created (also read
       when parameters->use_initial_flow is set).
@param status Output track status, length N.
@param errors Optional output mean absolute patch errors, length N.
@returns ErrorCode.
*/
ErrorCode klt_track(const KltPyramid* prev_pyramid,
                    const KltPyramid* next_pyramid,
                    const Matrix* prev_points,
                    const KltParameters* parameters,
                    Matrix* next_points,
                    uint8_t* status,
                    float64_t* errors = nullptr);

}  // namespace feature2d
}  // namespace cvlib

#endif  // CVLIB_FEATURE2D_KLT_H_
