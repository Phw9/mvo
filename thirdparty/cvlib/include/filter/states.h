// Concrete state types used by the cvlib::filter algorithms.
//
// Three state families are exposed:
//   - PointState : Euclidean point with optional velocity / acceleration.
//   - PoseState  : SE(3) pose (rotation + translation) with 6-DOF tangent.
//   - ImuState   : 15-DOF IMU strap-down state for visual-inertial odometry.

#ifndef CVLIB_FILTER_STATES_H_
#define CVLIB_FILTER_STATES_H_

#include "types.h"
#include "error_codes.h"

#include <cstdint>

namespace cvlib {
namespace filter {

// PointState dimension constants.
static constexpr int32_t kPointStateMinOrder = 1;
static constexpr int32_t kPointStateMaxOrder = 3;
static constexpr int32_t kPointStateBlock     = 3;

// PoseState fixed sizes.
static constexpr int32_t kPoseRotationDim  = 3;
static constexpr int32_t kPoseTangentDim   = 6;
static constexpr int32_t kPoseTangentRotOff = 0;
static constexpr int32_t kPoseTangentPosOff = 3;

// ImuState fixed sizes.
static constexpr int32_t kImuTangentDim   = 15;
static constexpr int32_t kImuOffPos       = 0;
static constexpr int32_t kImuOffVel       = 3;
static constexpr int32_t kImuOffRot       = 6;
static constexpr int32_t kImuOffBa        = 9;
static constexpr int32_t kImuOffBg        = 12;
static constexpr int32_t kImuNoiseDim     = 12;
static constexpr int32_t kImuNoiseOffNa   = 0;
static constexpr int32_t kImuNoiseOffNg   = 3;
static constexpr int32_t kImuNoiseOffNba  = 6;
static constexpr int32_t kImuNoiseOffNbg  = 9;

// 3D point estimator with order in {1=P, 2=PV, 3=PVA}.
struct PointState {
    int32_t order;
    Vector  mean;
    Matrix  cov;
};

// SE(3) pose estimator. Tangent order: [delta_theta(3), delta_p(3)].
struct PoseState {
    Matrix R;
    Vector t;
    Matrix cov;
};

// 15-DOF IMU strap-down state. Tangent order: [dp dv dtheta dba dbg].
struct ImuState {
    Vector p;
    Vector v;
    Matrix R;
    Vector ba;
    Vector bg;
    Matrix cov;
};

/*
Allocates a zero-initialized PointState of the requested order.

@param order Polynomial order (1=P, 2=PV, 3=PVA); other values yield zeroed state.
@returns PointState with mean / cov sized as 3*order; zero on invalid order.
*/
PointState point_state_create(int32_t order);

/*
Releases buffers owned by a PointState; safe on already-freed states.

@param state Target state.
*/
void point_state_destroy(PointState* state);

/*
Allocates a PoseState with R = I, t = 0, cov = 0 (6x6).

@returns Freshly allocated identity PoseState.
*/
PoseState pose_state_create();

/*
Releases buffers owned by a PoseState; safe on already-freed states.

@param state Target state.
*/
void pose_state_destroy(PoseState* state);

/*
Allocates an ImuState with all nominal components zero and R = I.

@returns Freshly allocated identity ImuState.
*/
ImuState imu_state_create();

/*
Releases buffers owned by an ImuState; safe on already-freed states.

@param state Target state.
*/
void imu_state_destroy(ImuState* state);

/*
Returns the tangent (covariance) dimension of a PointState.

@param state Source state.
@returns 3 * order.
*/
int32_t point_state_dim(const PointState* state);

}  // namespace filter
}  // namespace cvlib

#endif  // CVLIB_FILTER_STATES_H_
