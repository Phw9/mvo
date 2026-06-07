// Levenberg-Marquardt solver.

#ifndef CVLIB_OPTIMIZE_LM_H_
#define CVLIB_OPTIMIZE_LM_H_

#include "types.h"
#include "error_codes.h"
#include "optimize/problem.h"
#include "optimize/loss.h"

#include <cstdint>

namespace cvlib {
namespace optimize {

static constexpr int32_t kTermNotConverged    = 0;
static constexpr int32_t kTermFtol            = 1;
static constexpr int32_t kTermGtol            = 2;
static constexpr int32_t kTermXtol            = 3;
static constexpr int32_t kTermMaxIter         = 4;
static constexpr int32_t kTermLambdaTooLarge  = 5;
static constexpr int32_t kTermUserAbort       = 6;

// Solver options for Levenberg-Marquardt.
struct LMOptions {
    float64_t init_lambda;
    float64_t lambda_up;
    float64_t lambda_down;
    float64_t max_lambda;
    int32_t   max_iter;
    float64_t ftol;
    float64_t xtol;
    float64_t gtol;
    float64_t numerical_diff_step;
    LossInfo  loss;
};

// Solver report populated on return.
struct OptimizeReport {
    int32_t   iterations;
    float64_t initial_cost;
    float64_t final_cost;
    bool      converged;
    int32_t   termination;
};

// Default LM options matching common Ceres-like presets.
LMOptions default_lm_options();

/*
Runs Levenberg-Marquardt on the given problem.
Updates params_inout in place with the optimized parameters.

@param problem Problem definition (residual_fn required).
@param params_inout In/out parameter vector of length problem->n_params.
@param options Optional solver options; null uses default_lm_options().
@param report Optional output report; may be null.
@returns ErrorCode.
*/
ErrorCode levenberg_marquardt(const Problem* problem,
                              Vector* params_inout,
                              const LMOptions* options = nullptr,
                              OptimizeReport* report = nullptr);

}  // namespace optimize
}  // namespace cvlib

#endif  // CVLIB_OPTIMIZE_LM_H_
