// Forward kinematics (DH convention).

#ifndef CVLIB_KINEMATICS_FK_H_
#define CVLIB_KINEMATICS_FK_H_

#include "../types.h"
#include "../error_codes.h"

#include <cstdint>

namespace cvlib {
namespace kinematics {

/*
Builds the single-link 4-by-4 homogeneous transform from standard DH parameters.

@param a DH link length.
@param alpha DH link twist (radians).
@param d DH link offset.
@param theta DH joint angle (radians).
@param result Output 4-by-4; must be pre-allocated.
@returns ErrorCode.

*/

ErrorCode dh_tf(float64_t a, float64_t alpha,
                              float64_t d, float64_t theta,
                              Matrix* result);

/*
Computes end-effector pose from a DH table and joint angles.

@param dh_params Rows-by-4 table [a, alpha, d, theta_offset] per joint.
@param joint_angles Length equals number of DH rows.
@param result Output 4-by-4 end-effector transform; must be pre-allocated.
@returns ErrorCode.

*/

ErrorCode fk(const Matrix* dh_params,
                             const Vector* joint_angles,
                             Matrix* result);

/*
Writes cumulative link transforms along the chain.

@param dh_params Same convention as fk.
@param joint_angles Same length as number of joints.
@param transforms Buffer for successive 4-by-4 transforms; pre-allocated.
@param count Output number of transforms written.
@returns ErrorCode.

*/

ErrorCode intermediate_tfs(const Matrix* dh_params,
                                      const Vector* joint_angles,
                                      Matrix* transforms,
                                      int32_t* count);

/*
Computes joint positions in the base frame; row 0 is the base origin.

@param dh_params Same convention as fk.
@param joint_angles Same length as number of joints.
@param positions Output matrix of joint origins; must be pre-allocated.
@returns ErrorCode.

*/

ErrorCode joint_positions(const Matrix* dh_params,
                              const Vector* joint_angles,
                              Matrix* positions);

}  // namespace kinematics
}  // namespace cvlib

#endif  // CVLIB_KINEMATICS_FK_H_
