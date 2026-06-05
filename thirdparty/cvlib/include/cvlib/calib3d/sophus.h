// SO(3) and SE(3) Lie group operations.

#ifndef CVLIB_CALIB3D_SOPHUS_H_
#define CVLIB_CALIB3D_SOPHUS_H_

#include "cvlib/types.h"
#include "cvlib/error_codes.h"

#include <cstdint>

namespace cvlib {
namespace calib3d {

/*
Maps R^3 to the 3-by-3 skew-symmetric matrix [v]_x.

@param vector Input, length 3.
@param result Output 3-by-3 skew matrix; must be pre-allocated.
@returns ErrorCode.
*/
ErrorCode hat(const Vector* vector, Matrix* result);

/*
Extracts v from a skew-symmetric 3-by-3 matrix [v]_x.

@param matrix Input 3-by-3 skew-symmetric.
@param result Output vector, length 3.
@returns ErrorCode.
*/
ErrorCode vee(const Matrix* matrix, Vector* result);

/*
Computes the SO(3) logarithm: rotation matrix to axis-angle (angle times unit axis).

@param rotation Input 3-by-3 rotation.
@param result Output axis-angle, length 3.
@returns ErrorCode.
*/
ErrorCode so3_log(const Matrix* rotation, Vector* result);

/*
Computes the SO(3) exponential (Rodrigues): axis-angle to rotation matrix.

@param axis_angle Input axis-angle, length 3.
@param result Output 3-by-3 rotation; must be pre-allocated.
@returns ErrorCode.
*/
ErrorCode so3_exp(const Vector* axis_angle, Matrix* result);

/*
Computes the left-trivialized Jacobian J_l(xi) for SE(3), xi = [rho; phi].
Use for global / world-frame perturbation: T_new = exp(delta) * T.

@param xi Input twist, length 6.
@param result Output 6-by-6 Jacobian; must be pre-allocated.
@returns ErrorCode.
*/
ErrorCode left_jac_se3(const Vector* xi, Matrix* result);

/*
Computes the right-trivialized Jacobian J_r(xi) for SE(3), xi = [rho; phi].
Use for local / body-frame perturbation: T_new = T * exp(delta).
Identity J_r(xi) = J_l(-xi) is used internally.

@param xi Input twist, length 6.
@param result Output 6-by-6 Jacobian; must be pre-allocated.
@returns ErrorCode.
*/
ErrorCode right_jac_se3(const Vector* xi, Matrix* result);

/*
Computes the SE(3) logarithm: homogeneous transform to twist [rho; phi].

@param transformation Input 4-by-4 homogeneous transform.
@param result Output twist, length 6.
@returns ErrorCode.
*/
ErrorCode se3_log(const Matrix* transformation, Vector* result);

/*
Computes the SE(3) exponential: twist to homogeneous transform.

@param xi Input twist, length 6.
@param result Output 4-by-4 transform; must be pre-allocated.
@returns ErrorCode.
*/
ErrorCode se3_exp(const Vector* xi, Matrix* result);

/*
Computes the adjoint representation Ad(T) (6-by-6) for T in SE(3).

@param transformation Input 4-by-4 rigid transform.
@param result Output 6-by-6 adjoint; must be pre-allocated.
@returns ErrorCode.
*/
ErrorCode adj_se3(const Matrix* transformation, Matrix* result);

/*
Interpolates two SO(3) rotations with geodesic interpolation.

@param r1 Start rotation, 3-by-3.
@param r2 End rotation, 3-by-3.
@param t Interpolation factor in [0, 1].
@param result Output interpolated rotation, 3-by-3.
@returns ErrorCode.
*/
ErrorCode so3_interp(const Matrix* r1, const Matrix* r2,
                     float64_t t, Matrix* result);

/*
Interpolates two SE(3) transforms with Lie algebra interpolation.

@param t1 Start transform, 4-by-4.
@param t2 End transform, 4-by-4.
@param t Interpolation factor in [0, 1].
@param result Output interpolated transform, 4-by-4.
@returns ErrorCode.
*/
ErrorCode se3_interp(const Matrix* t1, const Matrix* t2,
                     float64_t t, Matrix* result);

}  // namespace calib3d
}  // namespace cvlib

#endif  // CVLIB_CALIB3D_SOPHUS_H_
