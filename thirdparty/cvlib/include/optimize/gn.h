// Gauss-Newton solver (no damping).

#ifndef CVLIB_OPTIMIZE_GN_H_
#define CVLIB_OPTIMIZE_GN_H_

#include "../types.h"
#include "../error_codes.h"
#include "problem.h"
#include "lm.h"

#include <cstdint>

namespace cvlib {
namespace optimize {

// Gauss-Newton options (subset of LMOptions).

struct GNOptions {
    int32_t   max_iter;
    float64_t ftol;
    float64_t xtol;
    float64_t gtol;
    float64_t numerical_diff_step;
    LossInfo  loss;
};

GNOptions default_gn_options();

/*
Runs Gauss-Newton iterations on the given problem.

@param problem Problem definition.
@param params_inout In/out parameter vector.
@param options Optional solver options; null uses default_gn_options().
@param report Optional output report; may be null.
@returns ErrorCode.

*/

ErrorCode gauss_newton(const Problem* problem,
                       Vector* params_inout,
                       const GNOptions* options = nullptr,
                       OptimizeReport* report = nullptr);

}  // namespace optimize
}  // namespace cvlib

#endif  // CVLIB_OPTIMIZE_GN_H_
