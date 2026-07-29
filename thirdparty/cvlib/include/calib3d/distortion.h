// Public Brown-Conrady distortion API.

#ifndef CVLIB_CALIB3D_DISTORTION_H_
#define CVLIB_CALIB3D_DISTORTION_H_

#include "../types.h"
#include "../error_codes.h"

namespace cvlib {
namespace calib3d {

/*
Applies Brown-Conrady distortion to N normalized image points.
Supports radial (k1..k3 or rational k1..k6) and tangential (p1, p2) terms.

@param normalized N-by-2 matrix of normalized (x, y) coordinates.
@param dist_coeff Distortion coefficients of length 4, 5, 6, or 8.
@param distorted Output N-by-2; must be pre-allocated.
@returns ErrorCode.

*/

ErrorCode distort_points(const Matrix* normalized, const Vector* dist_coeff,
                         Matrix* distorted);

/*
Removes Brown-Conrady distortion using fixed-point iteration.

@param distorted N-by-2 matrix of distorted normalized coordinates.
@param dist_coeff Distortion coefficients of length 4, 5, 6, or 8.
@param normalized Output N-by-2; must be pre-allocated.
@returns ErrorCode.

*/

ErrorCode undistort_points(const Matrix* distorted, const Vector* dist_coeff,
                           Matrix* normalized);

/*
Maps distorted pixels to rectified pixels (sparse rectification): each
pixel is normalized by K, undistorted (Brown-Conrady), rotated by the
rectification R, then projected by the new projection P. This connects a
stereo_rectify result to the sparse stereo pipeline the way the dense
rectify map does for images.

@param points Distorted pixels, N-by-2.
@param k Original 3-by-3 camera matrix.
@param dist_coeff Distortion coefficients of length 4, 5, 6, or 8.
@param r Rectification rotation, 3-by-3.
@param p New projection (rectified 3-by-3 camera matrix, e.g. the left
       3x3 block of stereo_rectify's P).
@param out Output rectified pixels, N-by-2; pre-allocated. A point mapping
       behind the rectified camera (w <= 0) yields a NaN row.
@returns ErrorCode.
*/
ErrorCode undistort_rectify_points(const Matrix* points, const Matrix* k,
                                   const Vector* dist_coeff, const Matrix* r,
                                   const Matrix* p, Matrix* out);

}  // namespace calib3d
}  // namespace cvlib

#endif  // CVLIB_CALIB3D_DISTORTION_H_
