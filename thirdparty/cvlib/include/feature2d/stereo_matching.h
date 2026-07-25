// Rectified-stereo image matching: row-constrained descriptor matching
// and 1-D subpixel disparity refinement. The geometric half of the
// stereo pipeline (rectification, triangulation, reprojection) lives in
// calib3d/stereo.h.

#ifndef CVLIB_FEATURE2D_STEREO_MATCHING_H_
#define CVLIB_FEATURE2D_STEREO_MATCHING_H_

#include "../types.h"
#include "../error_codes.h"
#include "../feature2d/klt.h"
#include "../feature2d/matching.h"

#include <cstdint>
#include <vector>

namespace cvlib {
namespace feature2d {

/*
Row-constrained Hamming matching for a rectified stereo pair.

Candidates for a left keypoint are restricted to right keypoints whose
row differs by at most max_row_diff and whose disparity
d = u_left - u_right lies in [min_disparity, max_disparity]; the
nearest gated descriptor wins. The ratio, cross-check, and max-distance
filters from DescriptorMatchOptions apply within the gated candidate
sets (cross-check uses the symmetric gates). Ties resolve
deterministically to the lowest right index.

@param left_keypoints Left keypoints, N-by-2 rectified pixels.
@param left_desc Left descriptors, num_left * descriptor_bytes bytes.
@param num_left Number of left descriptors (== N).
@param right_keypoints Right keypoints, M-by-2 rectified pixels.
@param right_desc Right descriptors, num_right * descriptor_bytes bytes.
@param num_right Number of right descriptors (== M).
@param descriptor_bytes Bytes per descriptor (> 0).
@param max_row_diff Row-difference gate in pixels (>= 0).
@param min_disparity Lower disparity gate in pixels (>= 0).
@param max_disparity Upper disparity gate (>= min_disparity).
@param options Optional filters; null uses defaults.
@param matches Output matches ordered by ascending left index;
       query_idx is the left index, train_idx the right index.
@returns ErrorCode.
*/
ErrorCode match_rectified_stereo(
    const Matrix* left_keypoints,
    const uint8_t* left_desc, int32_t num_left,
    const Matrix* right_keypoints,
    const uint8_t* right_desc, int32_t num_right,
    int32_t descriptor_bytes,
    float64_t max_row_diff, float64_t min_disparity,
    float64_t max_disparity,
    const DescriptorMatchOptions* options,
    std::vector<DescriptorMatch>* matches);

/*
Refines stereo disparities to subpixel accuracy with a one-dimensional
Lucas-Kanade iteration along the rectified row.

Each left point (u, v) with initial disparity d searches the right
image at (u - d, v), moving in x only; the row stays fixed, so the
vertical drift a 2-D tracker can leak into disparity is excluded by
construction. Uses the KLT window/iteration/threshold parameters
(pyramid and fallback fields are ignored). Per-point status is 1 on
success and 0 when the point leaves the image, the patch is
ill-conditioned, or the inputs are non-finite; failed points keep
their initial disparity.

The two images must be single-channel and share the same element depth
(k64FC1 or k32FC1); the refinement runs its f64 or f32 path accordingly.

@param left_image Left rectified grayscale image (k64FC1 or k32FC1).
@param right_image Right rectified grayscale image, same type and shape.
@param left_points Left-image points, N-by-2 (k64FC1).
@param initial_disparity Initial disparities, length N.
@param parameters Tracker parameters; null uses klt_default_parameters.
@param disparity_out Output refined disparities, length N.
@param status Output per-point status, length N.
@param errors Optional output mean absolute patch errors, length N.
@returns ErrorCode.
*/
ErrorCode klt_refine_disparity(const image::ImageView* left_image,
                               const image::ImageView* right_image,
                               const Matrix* left_points,
                               const float64_t* initial_disparity,
                               const KltParameters* parameters,
                               float64_t* disparity_out,
                               uint8_t* status,
                               float64_t* errors = nullptr);

/*
Dense block-matching parameters.

@param block_radius Half-width of the square SAD window (>= 1).
@param min_disparity Lower disparity bound in pixels (>= 0).
@param max_disparity Upper disparity bound (>= min_disparity).
@param uniqueness_ratio A winner is kept only when its cost is below
  ratio times the best cost more than one disparity step away; rejects
  ambiguous (flat or repetitive) pixels. In [0, 1); 0 disables.
*/
struct StereoBlockMatchParams {
    int32_t   block_radius = 4;
    int32_t   min_disparity = 0;
    int32_t   max_disparity = 64;
    float64_t uniqueness_ratio = 0.0;
};

/*
Returns pass-through dense block-matching defaults.

@returns StereoBlockMatchParams.
*/
StereoBlockMatchParams default_stereo_block_match_params();

/*
Dense disparity for a rectified pair by winner-take-all SAD block
matching along the row, with parabolic subpixel interpolation of the
cost minimum. Pixels whose window leaves either image, whose disparity
range is empty, or that fail the uniqueness check get the invalid
sentinel -1; valid pixels get disparity d with
u_right = u_left - d.

The two images must be single-channel and share the same element depth
(k64FC1 or k32FC1); block matching runs its f64 or f32 path accordingly.

@param left_image Left rectified grayscale image (k64FC1 or k32FC1).
@param right_image Right rectified grayscale image, same type and size.
@param params Optional parameters; null uses defaults.
@param disparity Output rows-by-cols disparity map; pre-allocated.
@returns ErrorCode.
*/
ErrorCode compute_disparity(const image::ImageView* left_image,
                            const image::ImageView* right_image,
                            const StereoBlockMatchParams* params,
                            Matrix* disparity);

}  // namespace feature2d
}  // namespace cvlib

#endif  // CVLIB_FEATURE2D_STEREO_MATCHING_H_
