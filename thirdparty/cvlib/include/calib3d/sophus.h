// SO(3), SE(3), and Sim(3) Lie group operations and manifold updates.

#ifndef CVLIB_CALIB3D_SOPHUS_H_
#define CVLIB_CALIB3D_SOPHUS_H_

#include "../types.h"
#include "../error_codes.h"

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
Computes the left-trivialized Jacobian J_l(phi) for SO(3). Use for global /
world-frame rotation perturbation: R_new = exp(delta) * R.

@param phi Input axis-angle, length 3.
@param result Output 3-by-3 Jacobian; must be pre-allocated.
@returns ErrorCode.
*/
ErrorCode left_jac_so3(const Vector* phi, Matrix* result);

/*
Computes the right-trivialized Jacobian J_r(phi) for SO(3). Use for local /
body-frame rotation perturbation: R_new = R * exp(delta). Identity
J_r(phi) = J_l(-phi) is used internally.

@param phi Input axis-angle, length 3.
@param result Output 3-by-3 Jacobian; must be pre-allocated.
@returns ErrorCode.
*/
ErrorCode right_jac_so3(const Vector* phi, Matrix* result);

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

// Flat SE(3) pose parameterization used by the optimizer manifold
// callbacks: row-major R (9) followed by t (3), 6-DOF tangent.
static constexpr int32_t kSe3ParamSize  = 12;
static constexpr int32_t kSe3LocalSize  = 6;

/*
Left perturbation update on a single pose: T_new = exp(delta) * T.
Signature matches optimize::ManifoldPlusFn for direct use as an
optimizer callback. params/x_plus layout: row-major R (9) followed by
t (3).

@param x Input pose params (length 12).
@param n_params Must be 12.
@param delta Tangent perturbation (length 6, [rho; phi]).
@param n_local Must be 6.
@param x_plus Output pose params (length 12).
@param user_data Unused.

*/

void se3_plus_left(const float64_t* x, int32_t n_params,
                   const float64_t* delta, int32_t n_local,
                   float64_t* x_plus, void* user_data);

/*
Right perturbation update on a single pose: T_new = T * exp(delta).
Signature matches optimize::ManifoldPlusFn for direct use as an
optimizer callback. params/x_plus layout: row-major R (9) followed by
t (3).

@param x Input pose params (length 12).
@param n_params Must be 12.
@param delta Tangent perturbation (length 6, [rho; phi]).
@param n_local Must be 6.
@param x_plus Output pose params (length 12).
@param user_data Unused.

*/

void se3_plus_right(const float64_t* x, int32_t n_params,
                    const float64_t* delta, int32_t n_local,
                    float64_t* x_plus, void* user_data);

/*
Computes the Sim(3) exponential: 7-vector [rho; phi; sigma] to the
4-by-4 similarity transform [[s R, t], [0, 1]] with s = exp(sigma),
R = so3_exp(phi), and t coupling rho to both rotation and scale.

@param xi Input tangent vector, length 7.
@param result Output 4-by-4 similarity transform; must be pre-allocated.
@returns ErrorCode.
*/
ErrorCode sim3_exp(const Vector* xi, Matrix* result);

/*
Computes the Sim(3) logarithm: 4-by-4 similarity transform to the
7-vector [rho; phi; sigma]. The scale is recovered from the determinant
of the upper-left block, which must be positive.

@param transformation Input 4-by-4 similarity transform.
@param result Output tangent vector, length 7.
@returns ErrorCode.
*/
ErrorCode sim3_log(const Matrix* transformation, Vector* result);

/*
Left (world-frame) 7-DOF perturbation: T_new = sim3_exp(delta) * T.
result may alias transformation.

@param transformation Input 4-by-4 similarity transform.
@param delta Tangent perturbation, length 7.
@param result Output 4-by-4 similarity transform; must be pre-allocated.
@returns ErrorCode.
*/
ErrorCode sim3_plus_left(const Matrix* transformation, const Vector* delta,
                         Matrix* result);

/*
Right (body-frame) 7-DOF perturbation: T_new = T * sim3_exp(delta).
result may alias transformation.

@param transformation Input 4-by-4 similarity transform.
@param delta Tangent perturbation, length 7.
@param result Output 4-by-4 similarity transform; must be pre-allocated.
@returns ErrorCode.
*/
ErrorCode sim3_plus_right(const Matrix* transformation, const Vector* delta,
                          Matrix* result);

/*
Inverts a Sim(3) similarity transform in closed form: for
T = [[s R, t], [0, 1]], the inverse is [[(1/s) R^T, -(1/s) R^T t], [0, 1]].
Unlike a rigid inv_transform (which assumes s = 1), this restores the 1/s
scale. The upper-left block determinant must be positive.

@param transformation Input 4-by-4 similarity transform.
@param result Output 4-by-4 inverse; pre-allocated, must not alias input.
@returns ErrorCode.
*/
ErrorCode sim3_inverse(const Matrix* transformation, Matrix* result);

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
