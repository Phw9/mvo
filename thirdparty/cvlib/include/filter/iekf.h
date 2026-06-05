// Iterated Error-State Kalman filter (IEKF) update on a manifold state.

#ifndef CVLIB_FILTER_IEKF_H_
#define CVLIB_FILTER_IEKF_H_

#include "eskf.h"
#include "../types.h"
#include "../error_codes.h"

#include <cstdint>

namespace cvlib {
namespace filter {

// Iteration controls for the iterated update.
struct IekfOptions {
    int32_t   max_iter;
    float64_t tol;
};

/*
Iterated ESKF measurement update. Repeats the ESKF linearization at the
latest nominal state until ||delta|| < opts.tol or opts.max_iter is reached.

@param nominal Opaque nominal-state pointer (e.g. PoseState*).
@param P In-out tangent-dim covariance.
@param mani Manifold descriptor.
@param obs Observation descriptor.
@param z Measurement vector.
@param R Measurement noise covariance.
@param opts Iteration controls.
@returns ErrorCode.
*/

ErrorCode iekf_update(void* nominal, Matrix* P,
                      EskfManifold mani, EskfObs obs,
                      const Vector* z, const Matrix* R,
                      IekfOptions opts);

}  // namespace filter
}  // namespace cvlib

#endif  // CVLIB_FILTER_IEKF_H_
