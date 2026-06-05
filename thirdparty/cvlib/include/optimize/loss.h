// Robust loss functions for nonlinear least squares.

#ifndef CVLIB_OPTIMIZE_LOSS_H_
#define CVLIB_OPTIMIZE_LOSS_H_

#include "../types.h"

#include <cstdint>

namespace cvlib {
namespace optimize {

static constexpr int32_t kLossTrivial = 0;
static constexpr int32_t kLossHuber   = 1;
static constexpr int32_t kLossCauchy  = 2;

// Loss configuration: type tag plus scale parameter (e.g. Huber threshold).

struct LossInfo {
    int32_t   type;
    float64_t scale;
};

/*
Evaluates loss rho(s) and its derivative drho_ds at squared residual s.
For trivial loss, rho(s) = s and drho_ds = 1. For Huber and Cauchy use scale.

@param info Loss configuration.
@param s Squared residual norm.
@param rho Output rho(s).
@param drho Output drho/ds, used to scale the residual block.

*/

void apply_loss(const LossInfo* info, float64_t s,
                float64_t* rho, float64_t* drho);

}  // namespace optimize
}  // namespace cvlib

#endif  // CVLIB_OPTIMIZE_LOSS_H_
