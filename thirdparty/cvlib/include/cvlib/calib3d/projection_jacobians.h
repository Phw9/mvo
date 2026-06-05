// Pinhole projection chain-rule Jacobians.

#ifndef CVLIB_CALIB3D_PROJECTION_JACOBIANS_H_
#define CVLIB_CALIB3D_PROJECTION_JACOBIANS_H_

#include "cvlib/types.h"
#include "cvlib/error_codes.h"

#include <cstdint>

namespace cvlib {
namespace calib3d {

static constexpr int32_t kSe3PerturbLeft  = 0;
static constexpr int32_t kSe3PerturbRight = 1;

/*
2x3 Jacobian of normalized projection (x, y) = (X/Z, Y/Z) w.r.t. (X, Y, Z).
Output row-major.

@param p_cam Camera-frame point of length 3.
@param j_2x3 Output buffer of length 6.

*/

void normalize_jac(const float64_t* p_cam, float64_t* j_2x3);

/*
2x2 Jacobian of distorted normalized point w.r.t. normalized point.

@param x Normalized x.
@param y Normalized y.
@param dist_coeff Distortion coefficients of length >= 4.
@param j_2x2 Output buffer of length 4.

*/

void distort_point_jac(float64_t x, float64_t y, const Vector* dist_coeff,
                       float64_t* j_2x2);

/*
2x2 Jacobian of (u, v) = (fx*xd + cx, fy*yd + cy) w.r.t. (xd, yd).
Equivalent to diag(fx, fy).

@param k Camera intrinsics (3x3).
@param j_2x2 Output buffer of length 4.

*/

void intrinsics_point_jac(const Matrix* k, float64_t* j_2x2);

/*
2x3 Jacobian d(u, v) / d(P_cam) composing intrinsics, distortion, normalize.

@param k Camera intrinsics (3x3).
@param p_cam Camera-frame point of length 3.
@param dist_coeff Optional distortion coefficients; null skips distortion.
@param j_2x3 Output buffer of length 6.

*/

void uv_pcam_jac(const Matrix* k, const float64_t* p_cam,
                 const Vector* dist_coeff, float64_t* j_2x3);

/*
3x6 Jacobian d(P_cam) / d(delta) at delta = 0 for SE(3) perturbation.
Left  : T_new = exp(delta) * T; J = [I, -hat(P_cam)].
Right : T_new = T * exp(delta); J = [R, -R*hat(X_world)].

@param p_cam Camera-frame point (used in left mode).
@param j_3x6 Output buffer of length 18.
@param mode kSe3PerturbLeft (default) or kSe3PerturbRight.
@param r_3x3 Rotation R (used in right mode; null otherwise).
@param x_world World-frame point (used in right mode; null otherwise).
@returns ErrorCode.

*/

ErrorCode pose_point_jac(const float64_t* p_cam, float64_t* j_3x6,
                         int32_t mode = kSe3PerturbLeft,
                         const Matrix* r_3x3 = nullptr,
                         const float64_t* x_world = nullptr);

/*
2x6 reprojection Jacobian d(u, v) / d(delta) for one observation.
Composes normalize_jac, distort_point_jac, intrinsics_point_jac and pose_point_jac.
Requires positive camera-frame depth.

@param k Camera intrinsics (3x3).
@param p_cam Camera-frame point (always used).
@param j_2x6 Output buffer of length 12.
@param dist_coeff Optional distortion coefficients; null skips distortion.
@param mode kSe3PerturbLeft (default) or kSe3PerturbRight.
@param r_3x3 Rotation (used in right mode; null otherwise).
@param x_world World-frame point (used in right mode; null otherwise).
@returns ErrorCode.

*/

ErrorCode reproj_jac(const Matrix* k, const float64_t* p_cam,
                     float64_t* j_2x6,
                     const Vector* dist_coeff = nullptr,
                     int32_t mode = kSe3PerturbLeft,
                     const Matrix* r_3x3 = nullptr,
                     const float64_t* x_world = nullptr);

}  // namespace calib3d
}  // namespace cvlib

#endif  // CVLIB_CALIB3D_PROJECTION_JACOBIANS_H_
