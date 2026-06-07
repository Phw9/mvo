// Forward-mode autodiff helpers using Jet<N> (header-only).

#ifndef CVLIB_OPTIMIZE_AUTODIFF_H_
#define CVLIB_OPTIMIZE_AUTODIFF_H_

#include "optimize/jet.h"
#include "types.h"
#include "error_codes.h"

#include <cstdint>

namespace cvlib {
namespace optimize {

/*
Evaluates residuals and Jacobian for a templated cost functor via autodiff.

Functor concept:
    template <typename T> bool operator()(const T* x, T* r) const;
where x has length N_LOCAL and r has length N_RES.

@param functor User-defined cost.
@param x Input parameters of length N_LOCAL.
@param residuals Output residuals of length N_RES (may be null).
@param jacobian_row_major Output Jacobian (N_RES x N_LOCAL) (may be null).
@returns true on functor success, false otherwise.

*/

template <int32_t N_RES, int32_t N_LOCAL, typename Functor>
bool autodiff_evaluate(const Functor& functor, const float64_t* x,
                       float64_t* residuals, float64_t* jacobian_row_major) {
    using JetT = Jet<N_LOCAL>;
    JetT jet_x[N_LOCAL];
    JetT jet_r[N_RES];
    for (int32_t i = 0; i < N_LOCAL; ++i) {
        jet_x[i] = JetT(x[i], i);
    }
    bool ok = functor(jet_x, jet_r);
    if (ok) {
        if (residuals != nullptr) {
            for (int32_t i = 0; i < N_RES; ++i) {
                residuals[i] = jet_r[i].a;
            }
        }
        if (jacobian_row_major != nullptr) {
            for (int32_t i = 0; i < N_RES; ++i) {
                for (int32_t j = 0; j < N_LOCAL; ++j) {
                    jacobian_row_major[i * N_LOCAL + j] = jet_r[i].v[j];
                }
            }
        }
    }
    return ok;
}

}  // namespace optimize
}  // namespace cvlib

#endif  // CVLIB_OPTIMIZE_AUTODIFF_H_
