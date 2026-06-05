// Forward-difference Jacobian for problems without an analytical Jacobian.

#ifndef CVLIB_OPTIMIZE_NUMERICAL_DIFF_H_
#define CVLIB_OPTIMIZE_NUMERICAL_DIFF_H_

#include "../types.h"
#include "../error_codes.h"
#include "problem.h"

#include <cstdint>

namespace cvlib {
namespace optimize {

/*
Computes a forward-difference Jacobian using the problem's plus_fn for
manifold-aware perturbations. Workspace buffers must be sized to the
problem dimensions; the function does not allocate.

@param problem Problem definition (residual_fn and plus_fn used).
@param params Current parameter vector of length n_params.
@param step Finite-difference step in tangent space.
@param jacobian Output row-major (n_residuals by n_local) buffer.
@returns ErrorCode.

*/

ErrorCode compute_jacobian_forward_diff(const Problem* problem,
                                        const float64_t* params,
                                        float64_t step,
                                        float64_t* jacobian);

}  // namespace optimize
}  // namespace cvlib

#endif  // CVLIB_OPTIMIZE_NUMERICAL_DIFF_H_
