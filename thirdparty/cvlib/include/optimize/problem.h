// Generic nonlinear least-squares problem definition.

#ifndef CVLIB_OPTIMIZE_PROBLEM_H_
#define CVLIB_OPTIMIZE_PROBLEM_H_

#include "types.h"
#include "error_codes.h"

#include <cstdint>

namespace cvlib {
namespace optimize {

/*
Residual callback. Fill residuals from current parameters.

@returns true on success, false to abort the iteration.
*/

using ResidualFn = bool (*)(const float64_t* params, int32_t n_params,
                            float64_t* residuals, int32_t n_residuals,
                            void* user_data);

/*
Jacobian callback. Fill row-major Jacobian (n_residuals by n_local).
If null on Problem, the solver computes a numerical Jacobian instead.
*/

using JacobianFn = bool (*)(const float64_t* params, int32_t n_params,
                            float64_t* jacobian_row_major,
                            int32_t n_residuals, int32_t n_local,
                            void* user_data);

/*
Manifold update: x_plus = x boxplus delta.
If null on Problem, Euclidean addition is used (n_local must equal n_params).
*/

using ManifoldPlusFn = void (*)(const float64_t* x, int32_t n_params,
                                const float64_t* delta, int32_t n_local,
                                float64_t* x_plus_delta, void* user_data);

/*
Problem definition consumed by gauss_newton and levenberg_marquardt.

@param n_params Length of the parameter vector x.
@param n_local Tangent-space dimension used by jacobian/plus (defaults to n_params).
@param n_residuals Number of residual entries the solver expects.
@param residual_fn Required cost evaluator.
@param jacobian_fn Optional analytical Jacobian; null triggers numerical diff.
@param plus_fn Optional manifold update; null falls back to Euclidean addition.
@param user_data Opaque pointer forwarded to every callback.
*/

struct Problem {
    int32_t n_params;
    int32_t n_local;
    int32_t n_residuals;
    ResidualFn     residual_fn;
    JacobianFn     jacobian_fn;
    ManifoldPlusFn plus_fn;
    void*          user_data;
};

}  // namespace optimize
}  // namespace cvlib

#endif  // CVLIB_OPTIMIZE_PROBLEM_H_
