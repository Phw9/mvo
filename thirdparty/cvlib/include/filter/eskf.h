// Generic error-state Kalman filter (ESKF) primitives for manifold states.
//
// The nominal state lives on the manifold (e.g. SE(3) for a pose, or 15-DOF
// IMU). The filter tracks a Gaussian on the additive error in tangent space
// and the caller supplies a "boxplus" injection that maps tangent deltas
// back into the nominal state.

#ifndef CVLIB_FILTER_ESKF_H_
#define CVLIB_FILTER_ESKF_H_

#include "../types.h"
#include "../error_codes.h"

#include <cstdint>

namespace cvlib {
namespace filter {

// Inject the additive tangent delta into the nominal state in place.
typedef ErrorCode (*BoxplusFn)(void* nominal, const Vector* delta);

// Compute the residual r = z (-) h(nominal) on the measurement manifold.
typedef ErrorCode (*ResidualFn)(const void* nominal, const Vector* z,
                                void* user, Vector* residual);

// Compute the m-by-tangent_dim Jacobian H = dh/d(delta) at the nominal.
typedef ErrorCode (*ObsJacOnManFn)(const void* nominal, void* user,
                                   Matrix* H);

// Bundles the manifold injection + tangent dimension.
struct EskfManifold {
    int32_t   tangent_dim;
    BoxplusFn inject;
    void*     user;
};

// Bundles the measurement model on the manifold.
struct EskfObs {
    int32_t        meas_dim;
    ResidualFn     residual;
    ObsJacOnManFn  jac;
    void*          user;
};

/*
Propagates the error-state covariance: P <- F P F^T + Q.

@param P In-out tangent-dim covariance.
@param F Tangent-dim transition matrix (set to I for stationary nominal).
@param Q Tangent-dim process noise.
@returns ErrorCode.
*/

ErrorCode eskf_propagate_cov(Matrix* P, const Matrix* F, const Matrix* Q);

/*
ESKF measurement update: compute Kalman gain on tangent space, inject the
correction into the nominal via mani.inject, and refresh P with Joseph form.

@param nominal Opaque pointer to the nominal state (e.g. PoseState*).
@param P In-out tangent-dim covariance.
@param mani Manifold descriptor (tangent dim + injection callback).
@param obs Measurement descriptor (residual + Jacobian).
@param z Measurement vector.
@param R Measurement noise covariance.
@returns ErrorCode.
*/

ErrorCode eskf_update(void* nominal, Matrix* P,
                      EskfManifold mani, EskfObs obs,
                      const Vector* z, const Matrix* R);

}  // namespace filter
}  // namespace cvlib

#endif  // CVLIB_FILTER_ESKF_H_
