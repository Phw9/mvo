// Built-in manifold update helpers.

#ifndef CVLIB_OPTIMIZE_MANIFOLD_H_
#define CVLIB_OPTIMIZE_MANIFOLD_H_

#include "../types.h"

#include <cstdint>

namespace cvlib {
namespace optimize {

/*
Euclidean update: x_plus = x + delta. n_local must equal n_params.

@param x Input parameter vector.
@param delta Tangent perturbation.
@param x_plus_delta Output of the same size as x.

*/

void euclidean_plus(const float64_t* x, int32_t n_params,
                    const float64_t* delta, int32_t n_local,
                    float64_t* x_plus_delta, void* user_data);

}  // namespace optimize
}  // namespace cvlib

#endif  // CVLIB_OPTIMIZE_MANIFOLD_H_
