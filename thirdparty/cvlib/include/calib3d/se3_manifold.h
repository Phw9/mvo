// SE(3) manifold updates compatible with optimize::ManifoldPlusFn.

#ifndef CVLIB_CALIB3D_SE3_MANIFOLD_H_
#define CVLIB_CALIB3D_SE3_MANIFOLD_H_

#include "../types.h"

#include <cstdint>

namespace cvlib {
namespace calib3d {

static constexpr int32_t kSe3ParamSize  = 12;
static constexpr int32_t kSe3LocalSize  = 6;

/*
Left perturbation update on a single pose: T_new = exp(delta) * T.
params/x_plus layout: row-major R (9) followed by t (3).

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
params/x_plus layout: row-major R (9) followed by t (3).

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

}  // namespace calib3d
}  // namespace cvlib

#endif  // CVLIB_CALIB3D_SE3_MANIFOLD_H_
