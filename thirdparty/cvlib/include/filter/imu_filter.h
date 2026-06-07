// 15-DOF IMU error-state Kalman filter helpers (strap-down).

#ifndef CVLIB_FILTER_IMU_FILTER_H_
#define CVLIB_FILTER_IMU_FILTER_H_

#include "filter/states.h"
#include "types.h"
#include "error_codes.h"

namespace cvlib {
namespace filter {

// Single IMU sample: body-frame accel + gyro + integration step.
struct ImuMeasurement {
    Vector    accel;
    Vector    gyro;
    float64_t dt;
};

/*
Strap-down predict step. Integrates the nominal state with the supplied
accel + gyro and propagates the 15-DOF error-state covariance.

@param state In-out ImuState.
@param meas Body-frame measurement bundle.
@param gravity World-frame gravity (length 3, e.g. [0, 0, -9.81]).
@param Q 15-by-15 process noise covariance for this step.
@returns ErrorCode.
*/
ErrorCode imu_predict(ImuState* state, ImuMeasurement meas,
                      const Vector* gravity, const Matrix* Q);

/*
Position update from a world-frame measurement (e.g. GPS).

@param state In-out ImuState.
@param p_meas Measured position (length 3).
@param R 3-by-3 measurement noise covariance.
@returns ErrorCode.
*/
ErrorCode imu_update_position(ImuState* state, const Vector* p_meas,
                              const Matrix* R);

/*
Pose update from a world-frame measurement.

@param state In-out ImuState.
@param R_meas Measured rotation (3-by-3).
@param p_meas Measured position (length 3).
@param R 6-by-6 measurement noise covariance ([dtheta, dp]).
@returns ErrorCode.
*/
ErrorCode imu_update_pose(ImuState* state,
                          const Matrix* R_meas, const Vector* p_meas,
                          const Matrix* R);

}  // namespace filter
}  // namespace cvlib

#endif  // CVLIB_FILTER_IMU_FILTER_H_
