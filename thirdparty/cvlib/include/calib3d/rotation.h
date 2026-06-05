// Rotation parameterizations (axis-angle, quaternion, RPY, Rodrigues).

#ifndef CVLIB_CALIB3D_ROTATION_H_
#define CVLIB_CALIB3D_ROTATION_H_

#include "../types.h"
#include "../error_codes.h"

#include <cstdint>

namespace cvlib {
namespace calib3d {

static constexpr int32_t kSequenceZYX = 0;
static constexpr int32_t kSequenceXYZ = 1;
static constexpr int32_t kSequenceXZY = 2;
static constexpr int32_t kSequenceYXZ = 3;
static constexpr int32_t kSequenceYZX = 4;
static constexpr int32_t kSequenceZXY = 5;
static constexpr int32_t kNumSequences = 6;

static constexpr int32_t kQuaternionSize = 4;

/*
Converts a rotation matrix to a unit axis, angle in radians, and axis vector.

@param rotation Input 3-by-3 rotation.
@param axis Output unit axis, length 3.
@param angle Output angle in radians.
@returns ErrorCode.
*/
ErrorCode mat2axis(const Matrix* rotation, Vector* axis, float64_t* angle);

/*
Builds a rotation matrix from an axis and angle; axis need not be unit.

@param axis Input axis, length 3 (normalized internally).
@param angle Angle in radians.
@param result Output 3-by-3 rotation; must be pre-allocated.
@returns ErrorCode.
*/
ErrorCode axis2mat(const Vector* axis, float64_t angle, Matrix* result);

/*
Converts a rotation matrix to a quaternion [w, x, y, z].

@param rotation Input 3-by-3 rotation.
@param result Output length 4.
@returns ErrorCode.
*/
ErrorCode mat2quat(const Matrix* rotation, Vector* result);

/*
Converts a quaternion to a rotation matrix.

@param quaternion Input length 4.
@param result Output 3-by-3 rotation; must be pre-allocated.
@returns ErrorCode.
*/
ErrorCode quat2mat(const Vector* quaternion, Matrix* result);

/*
Computes quaternion conjugate [w, -x, -y, -z].

@param quaternion Input quaternion, length 4.
@param result Output quaternion, length 4.
@returns ErrorCode.
*/
ErrorCode quaternion_conjugate(const Vector* quaternion, Vector* result);

/*
Computes quaternion inverse q^{-1} = conjugate(q) / ||q||^2.

@param quaternion Input quaternion, length 4.
@param result Output quaternion inverse, length 4.
@returns ErrorCode.
*/
ErrorCode quaternion_inverse(const Vector* quaternion, Vector* result);

/*
Computes quaternion product result = q1 * q2.

@param q1 Left quaternion, length 4.
@param q2 Right quaternion, length 4.
@param result Output quaternion, length 4.
@returns ErrorCode.
*/
ErrorCode quaternion_multiply(const Vector* q1, const Vector* q2,
                              Vector* result);

/*
Interpolates two quaternions with spherical linear interpolation.

@param q1 Start quaternion, length 4.
@param q2 End quaternion, length 4.
@param t Interpolation factor in [0, 1].
@param result Output quaternion, length 4.
@returns ErrorCode.
*/
ErrorCode quaternion_slerp(const Vector* q1, const Vector* q2,
                           float64_t t, Vector* result);

/*
Computes the Euclidean norm ||q|| of a quaternion.

@param quaternion Input quaternion, length 4.
@param norm_out Output norm value (>= 0).
@returns ErrorCode.
*/
ErrorCode quaternion_norm(const Vector* quaternion, float64_t* norm_out);

/*
Returns a unit quaternion q / ||q|| or kSingularMatrix when ||q|| < epsilon.

@param quaternion Input quaternion, length 4.
@param result Output unit quaternion, length 4.
@returns ErrorCode.
*/
ErrorCode quaternion_normalize(const Vector* quaternion, Vector* result);

/*
Computes the dot product of two quaternions, treated as length-4 vectors.

@param q1 First quaternion, length 4.
@param q2 Second quaternion, length 4.
@param dot_out Output scalar dot product.
@returns ErrorCode.
*/
ErrorCode quaternion_dot(const Vector* q1, const Vector* q2,
                         float64_t* dot_out);

/*
Returns the geodesic angle (radians) between two rotation matrices.

The result lies in [0, pi] and equals the rotation angle of R1^T * R2.

@param r1 First rotation matrix, 3-by-3.
@param r2 Second rotation matrix, 3-by-3.
@param angle_out Output angle in radians.
@returns ErrorCode.
*/
ErrorCode axis_angle_between_rotations(const Matrix* r1, const Matrix* r2,
                                       float64_t* angle_out);

/*
Converts a rotation matrix to extrinsic roll-pitch-yaw angles.

@param rotation Input 3-by-3 rotation.
@param result Output [roll, pitch, yaw], length 3.
@param sequence Euler order (kSequence* constants); defaults to ZYX.
@param degree True for degrees, false (default) for radians.
@returns ErrorCode.
*/
ErrorCode mat2rpy(const Matrix* rotation, Vector* result,
                  int32_t sequence = kSequenceZYX, bool degree = false);

/*
Converts roll-pitch-yaw angles to a rotation matrix (inverse of mat2rpy).

@param angles Input [roll, pitch, yaw], length 3.
@param result Output 3-by-3 rotation; must be pre-allocated.
@param sequence Euler order (kSequence* constants); defaults to ZYX.
@param degree True for degrees, false (default) for radians.
@returns ErrorCode.
*/
ErrorCode rpy2mat(const Vector* angles, Matrix* result,
                  int32_t sequence = kSequenceZYX, bool degree = false);

/*
Converts a rotation matrix to a Rodrigues vector (SO(3) logarithm).

@param rotation Input 3-by-3 rotation.
@param result Output Rodrigues vector, length 3.
@returns ErrorCode.
*/
ErrorCode mat2rod(const Matrix* rotation, Vector* result);

/*
Converts a Rodrigues vector to a rotation matrix (SO(3) exponential).

@param rodrigues Input Rodrigues vector, length 3.
@param result Output 3-by-3 rotation; must be pre-allocated.
@returns ErrorCode.
*/
ErrorCode rod2mat(const Vector* rodrigues, Matrix* result);

}  // namespace calib3d
}  // namespace cvlib

#endif  // CVLIB_CALIB3D_ROTATION_H_
