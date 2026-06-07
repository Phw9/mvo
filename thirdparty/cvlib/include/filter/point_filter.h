// Linear Kalman filter helpers for the PointState (3D position with optional
// velocity / acceleration).

#ifndef CVLIB_FILTER_POINT_FILTER_H_
#define CVLIB_FILTER_POINT_FILTER_H_

#include "filter/states.h"
#include "types.h"
#include "error_codes.h"

namespace cvlib {
namespace filter {

/*
Constant-position predict step. F = I; covariance grows by Q.

@param state In-out PointState.
@param Q Process noise covariance (dim-by-dim).
@returns ErrorCode.
*/
ErrorCode point_predict_cp(PointState* state, const Matrix* Q);

/*
Constant-velocity predict step (requires order >= 2).

@param state In-out PointState.
@param dt Positive timestep in seconds.
@param Q Process noise covariance (dim-by-dim).
@returns ErrorCode.
*/
ErrorCode point_predict_cv(PointState* state, float64_t dt, const Matrix* Q);

/*
Constant-acceleration predict step (requires order == 3).

@param state In-out PointState.
@param dt Positive timestep in seconds.
@param Q Process noise covariance (9-by-9).
@returns ErrorCode.
*/
ErrorCode point_predict_ca(PointState* state, float64_t dt, const Matrix* Q);

/*
Position observation update: H selects the position block of the state.

@param state In-out PointState.
@param z Measured position (length 3).
@param R Measurement noise covariance (3-by-3).
@returns ErrorCode.
*/
ErrorCode point_update_position(PointState* state, const Vector* z,
                                const Matrix* R);

/*
Velocity observation update: H selects the velocity block (requires order >= 2).

@param state In-out PointState.
@param z Measured velocity (length 3).
@param R Measurement noise covariance (3-by-3).
@returns ErrorCode.
*/
ErrorCode point_update_velocity(PointState* state, const Vector* z,
                                const Matrix* R);

}  // namespace filter
}  // namespace cvlib

#endif  // CVLIB_FILTER_POINT_FILTER_H_
