// SE(3) pose ESKF / IEKF helpers built on the generic eskf / iekf primitives.

#ifndef CVLIB_FILTER_POSE_FILTER_H_
#define CVLIB_FILTER_POSE_FILTER_H_

#include "filter/iekf.h"
#include "filter/states.h"
#include "types.h"
#include "error_codes.h"

namespace cvlib {
namespace filter {

// Body-frame twist used by the ESKF predict step.
struct PoseTwist {
    Vector    omega;
    Vector    velocity;
    float64_t dt;
};

/*
ESKF predict step: T <- T * Exp(twist * dt). Covariance grows by Q.

@param state In-out PoseState.
@param twist Body-frame twist + timestep.
@param Q 6-by-6 process noise covariance on the tangent.
@returns ErrorCode.
*/
ErrorCode pose_eskf_predict(PoseState* state, PoseTwist twist,
                            const Matrix* Q);

/*
ESKF update from a measured full pose (R_meas, t_meas).

@param state In-out PoseState.
@param R_meas Measured rotation (3-by-3).
@param t_meas Measured translation (length 3).
@param R_cov 6-by-6 measurement noise covariance.
@returns ErrorCode.
*/
ErrorCode pose_eskf_update_pose(PoseState* state,
                                const Matrix* R_meas, const Vector* t_meas,
                                const Matrix* R_cov);

/*
ESKF update from a measured position only.

@param state In-out PoseState.
@param p_meas Measured position (length 3).
@param R_cov 3-by-3 measurement noise covariance.
@returns ErrorCode.
*/
ErrorCode pose_eskf_update_position(PoseState* state,
                                    const Vector* p_meas,
                                    const Matrix* R_cov);

/*
Iterated ESKF update from a measured full pose.

@param state In-out PoseState.
@param R_meas Measured rotation.
@param t_meas Measured translation.
@param R_cov Measurement noise covariance.
@param opts Iteration controls.
@returns ErrorCode.
*/
ErrorCode pose_iekf_update_pose(PoseState* state,
                                const Matrix* R_meas, const Vector* t_meas,
                                const Matrix* R_cov, IekfOptions opts);

}  // namespace filter
}  // namespace cvlib

#endif  // CVLIB_FILTER_POSE_FILTER_H_
