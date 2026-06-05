// Inverse kinematics (numerical and 2R planar analytic).

#ifndef CVLIB_KINEMATICS_IK_H_
#define CVLIB_KINEMATICS_IK_H_

#include "cvlib/types.h"
#include "cvlib/error_codes.h"

#include <cstdint>

namespace cvlib {
namespace kinematics {

/*
Computes the geometric Jacobian for the current configuration.
Row and column layout follows the implementation and DH convention.

@param dh_params DH parameter table.
@param joint_angles Current joint vector.
@param result Output Jacobian; must be pre-allocated to the expected shape.
@returns ErrorCode.

*/

ErrorCode jac(const Matrix* dh_params,
                           const Vector* joint_angles,
                           Matrix* result);

// Default IK iteration cap and convergence tolerance.
static constexpr int32_t   kIkDefaultMaxIter = 200;
static constexpr float64_t kIkDefaultTol     = 1e-6;

/*
Runs damped least-squares inverse kinematics toward a target pose.

@param dh_params DH parameter table.
@param target_pose Desired 4-by-4 end-effector pose.
@param result Output joint vector; must be pre-allocated.
@param success Output true if the tolerance was met.
@param initial_guess Optional starting joint vector; null uses zeros.
@param joint_limits Optional bounds; null leaves joints unconstrained.
@param max_iter Maximum IK iterations (default kIkDefaultMaxIter).
@param tol Convergence threshold (default kIkDefaultTol).
@returns ErrorCode.

*/

ErrorCode ik(const Matrix* dh_params, const Matrix* target_pose,
             Vector* result, bool* success,
             const Vector* initial_guess = nullptr,
             const Matrix* joint_limits = nullptr,
             int32_t max_iter = kIkDefaultMaxIter,
             float64_t tol = kIkDefaultTol);

/*
Closed-form IK for a planar 2R arm with given link lengths.

@param a1 First link length.
@param a2 Second link length.
@param target_pos Goal position in the arm plane; must have length 2.
@param result Output two joint angles; must be pre-allocated.
@param success Output false if the goal is unreachable.
@returns ErrorCode.

*/

ErrorCode ik_2r_planar(float64_t a1, float64_t a2,
                                  const Vector* target_pos,
                                  Vector* result,
                                  bool* success);

}  // namespace kinematics
}  // namespace cvlib

#endif  // CVLIB_KINEMATICS_IK_H_
