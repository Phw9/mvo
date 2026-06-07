// Extended Kalman filter (EKF) primitives operating on a flat GaussianState.

#ifndef CVLIB_FILTER_EKF_H_
#define CVLIB_FILTER_EKF_H_

#include "filter/kalman.h"
#include "types.h"
#include "error_codes.h"

#include <cstdint>

namespace cvlib {
namespace filter {

// Nonlinear motion model: x_next = f(x).
typedef ErrorCode (*MotionFn)(const Vector* x, void* user, Vector* x_next);

// Jacobian of f with respect to the state at x.
typedef ErrorCode (*MotionJacFn)(const Vector* x, void* user, Matrix* F);

// Nonlinear observation model: z_pred = h(x).
typedef ErrorCode (*ObsFn)(const Vector* x, void* user, Vector* z_pred);

// Jacobian of h with respect to the state at x.
typedef ErrorCode (*ObsJacFn)(const Vector* x, void* user, Matrix* H);

// Bundle of motion model + Jacobian + opaque user pointer.
struct EkfMotion {
    MotionFn    f;
    MotionJacFn jac;
    void*       user;
};

// Bundle of observation model + Jacobian + opaque user pointer.
struct EkfObservation {
    ObsFn       h;
    ObsJacFn    jac;
    void*       user;
};

/*
EKF predict: nominal x <- f(x) and P <- F P F^T + Q with F = df/dx.

@param state In-out Gaussian state.
@param motion Motion model bundle (f, jac, user).
@param Q Process noise covariance (n-by-n).
@returns ErrorCode.
*/

ErrorCode ekf_predict(GaussianState* state, EkfMotion motion,
                      const Matrix* Q);

/*
EKF update: linearize h at x and apply the standard Kalman gain with Joseph form.

@param state In-out Gaussian state.
@param obs Observation model bundle (h, jac, user).
@param z Measurement vector (m).
@param R Measurement noise covariance (m-by-m).
@returns ErrorCode.
*/

ErrorCode ekf_update(GaussianState* state, EkfObservation obs,
                     const Vector* z, const Matrix* R);

}  // namespace filter
}  // namespace cvlib

#endif  // CVLIB_FILTER_EKF_H_
