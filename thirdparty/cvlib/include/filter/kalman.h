// Linear Kalman filter primitives operating on a flat GaussianState.

#ifndef CVLIB_FILTER_KALMAN_H_
#define CVLIB_FILTER_KALMAN_H_

#include "types.h"
#include "error_codes.h"

#include <cstdint>

namespace cvlib {
namespace filter {

// Gaussian state container shared by the linear KF and EKF.
struct GaussianState {
    Vector mean;
    Matrix cov;
};

/*
Performs the linear Kalman predict step in place: x <- F*x, P <- F*P*F^T + Q.

@param state In-out Gaussian state with matching mean / cov dimensions.
@param F State transition matrix (n-by-n).
@param Q Process noise covariance (n-by-n).
@returns ErrorCode.
*/

ErrorCode kalman_predict(GaussianState* state,
                         const Matrix* F, const Matrix* Q);

/*
Performs the linear Kalman update step in place using Joseph form for P.

@param state In-out Gaussian state.
@param z Measurement vector (m).
@param H Observation matrix (m-by-n).
@param R Measurement noise covariance (m-by-m).
@returns ErrorCode.
*/

ErrorCode kalman_update(GaussianState* state, const Vector* z,
                        const Matrix* H, const Matrix* R);

/*
Runs a forward Kalman filter over a measurement sequence.

@param measurements Stacked measurements; rows index time, cols match z.
@param F State transition matrix.
@param H Observation matrix.
@param Q Process noise covariance.
@param R Measurement noise covariance.
@param state Initial Gaussian state; updated in place to the final posterior.
@param state_history Optional output, n_meas-by-n; null disables history.
@returns ErrorCode.
*/

ErrorCode kalman_filter(const Matrix* measurements,
                        const Matrix* F, const Matrix* H,
                        const Matrix* Q, const Matrix* R,
                        GaussianState* state,
                        Matrix* state_history);

}  // namespace filter
}  // namespace cvlib

#endif  // CVLIB_FILTER_KALMAN_H_
