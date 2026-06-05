// Rigid transforms, projection, and reprojection error.

#ifndef CVLIB_CALIB3D_TRANSFORMS_H_
#define CVLIB_CALIB3D_TRANSFORMS_H_

#include "cvlib/types.h"
#include "cvlib/error_codes.h"
#include "cvlib/calib3d/rotation.h"

#include <cstdint>

namespace cvlib {
namespace calib3d {

static constexpr int32_t kDof6Size = 6;

/*
Applies R p + t to each row of points (3D points in rows).

@param points Input N-by-3.
@param rotation Input 3-by-3 rotation.
@param translation Input length 3.
@param result Output N-by-3; must be pre-allocated.
@returns ErrorCode.

*/

ErrorCode transform_points(const Matrix* points, const Matrix* rotation,
                           const Vector* translation, Matrix* result);

/*
Applies R to each row of points without translation.

@param points Input N-by-3.
@param rotation Input 3-by-3 rotation.
@param result Output N-by-3; must be pre-allocated.
@returns ErrorCode.

*/

ErrorCode rotate_points(const Matrix* points, const Matrix* rotation,
                        Matrix* result);

/*
Adds translation to each row of points.

@param points Input N-by-3.
@param translation Input length 3.
@param result Output N-by-3; must be pre-allocated.
@returns ErrorCode.

*/

ErrorCode translate_points(const Matrix* points, const Vector* translation,
                           Matrix* result);

/*
Decomposes a 4-by-4 rigid transform into rotation and translation.

@param transform Input 4-by-4 homogeneous transform.
@param rotation Output 3-by-3; must be pre-allocated.
@param translation Output length 3.
@returns ErrorCode.

*/

ErrorCode decompose_transform(const Matrix* transform, Matrix* rotation,
                              Vector* translation);

/*
Builds a 4-by-4 homogeneous transform from rotation and translation.

@param rotation Input 3-by-3 rotation.
@param translation Input length 3.
@param result Output 4-by-4; must be pre-allocated.
@returns ErrorCode.

*/

ErrorCode compose_transform(const Matrix* rotation, const Vector* translation,
                            Matrix* result);

/*
Inverts a rigid transform; the homogeneous flag selects the output layout.

@param transform Input 4-by-4 transform.
@param result Output inverse; pre-allocated to 4x4 if homogeneous else 3x4.
@param homogeneous True for 4x4 output (default), false for 3x4.
@returns ErrorCode.

*/

ErrorCode inv_transform(const Matrix* transform, Matrix* result,
                        bool homogeneous = true);

/*
Maps world points to the camera frame as R^T (p - t), where (R, t) is the
camera pose in the world frame.

@param points Input N-by-3 in world coordinates.
@param cam_rot Camera-to-world rotation 3-by-3.
@param cam_trans Camera origin in world frame, length 3.
@param result Output N-by-3 in camera frame; must be pre-allocated.
@returns ErrorCode.

*/

ErrorCode world2cam(const Matrix* points, const Matrix* cam_rot,
                    const Vector* cam_trans, Matrix* result);

/*
Maps camera-frame points to world coordinates as R p + t.

@param points Input N-by-3 in camera frame.
@param cam_rot Same convention as world2cam.
@param cam_trans Same convention as world2cam.
@param result Output N-by-3 in world frame; must be pre-allocated.
@returns ErrorCode.

*/

ErrorCode cam2world(const Matrix* points, const Matrix* cam_rot,
                    const Vector* cam_trans, Matrix* result);

/*
Projects camera-frame 3D points to 2D image coordinates.
Rejects non-finite or near-zero camera-frame depth.

@param points Input N-by-3 in camera frame.
@param k Camera intrinsics (3x3) with finite non-zero fx/fy.
@param result Output N-by-2 pixels; must be pre-allocated.
@param dist_coeff Distortion coefficients; null skips distortion.
@returns ErrorCode.

*/

ErrorCode cam2img(const Matrix* points, const Matrix* k,
                  Matrix* result, const Vector* dist_coeff = nullptr);

/*
Unprojects image points to camera-frame 3D using intrinsics.
Provides per-row depth via depth (length N) or unit depth when null.
Distortion is applied (undistortion of inputs) when dist_coeff is non-null.
Rejects non-finite or near-zero focal length/depth values.

@param points Input N-by-2 pixels.
@param k Camera intrinsics (3x3) with finite non-zero fx/fy.
@param result Output N-by-3; must be pre-allocated.
@param depth Optional per-row depth, length N.
@param dist_coeff Optional Brown-Conrady distortion coefficients.
@returns ErrorCode.

*/

ErrorCode img2cam(const Matrix* points, const Matrix* k,
                  Matrix* result,
                  const Vector* depth = nullptr,
                  const Vector* dist_coeff = nullptr);

/*
Full projection from world 3D to image 2D using camera-pose extrinsics and
intrinsics.

@param points Input N-by-3 world coordinates.
@param cam_rot Camera-to-world rotation 3-by-3.
@param cam_trans Camera origin in world frame, length 3.
@param k Camera intrinsics (3x3).
@param result Output N-by-2; must be pre-allocated.
@param dist_coeff Optional distortion coefficients.
@returns ErrorCode.

*/

ErrorCode world2img(const Matrix* points, const Matrix* cam_rot,
                    const Vector* cam_trans, const Matrix* k,
                    Matrix* result,
                    const Vector* dist_coeff = nullptr);

/*
Reprojection error between projected 3D points and 2D observations.

@param cam_rot Camera-to-world rotation 3-by-3.
@param cam_trans Camera origin in world frame, length 3.
@param pts3d Input N-by-3 world or model points.
@param pts2d Input N-by-2 image points.
@param k Camera intrinsics (3x3).
@param result Output scalar error.
@param dist_coeff Optional distortion coefficients.
@param return_mean True (default) for per-coordinate RMSE,
                   false for unaveraged L2 residual.
@returns ErrorCode.

*/

ErrorCode reproj_err(const Matrix* cam_rot, const Vector* cam_trans,
                     const Matrix* pts3d, const Matrix* pts2d,
                     const Matrix* k,
                     float64_t* result,
                     const Vector* dist_coeff = nullptr,
                     bool return_mean = true);

/*
Builds a 4-by-4 transform from translation + RPY (6-DOF).

@param dof6 Input [tx, ty, tz, roll, pitch, yaw], length 6.
@param result Output 4-by-4; must be pre-allocated.
@param sequence Euler order for the rotation part (default ZYX).
@param degree Angle units for RPY (default radians).
@returns ErrorCode.

*/

ErrorCode rpy62mat(const Vector* dof6, Matrix* result,
                   int32_t sequence = kSequenceZYX, bool degree = false);

/*
Builds a 4-by-4 transform from translation + Rodrigues rotation (6-DOF).

@param dof6 Input [tx, ty, tz, rx, ry, rz], length 6.
@param result Output 4-by-4; must be pre-allocated.
@returns ErrorCode.

*/

ErrorCode rod62mat(const Vector* dof6, Matrix* result);

/*
Decomposes a 4-by-4 transform into translation + RPY.

@param transform Input 4-by-4.
@param result Output length 6.
@param sequence Euler order for angles (default ZYX).
@param degree Angle units for output RPY (default radians).
@returns ErrorCode.

*/

ErrorCode mat2rpy6(const Matrix* transform, Vector* result,
                   int32_t sequence = kSequenceZYX, bool degree = false);

/*
Decomposes a 4-by-4 transform into translation + Rodrigues rotation.

@param transform Input 4-by-4.
@param result Output length 6.
@returns ErrorCode.

*/

ErrorCode mat2rod6(const Matrix* transform, Vector* result);

}  // namespace calib3d
}  // namespace cvlib

#endif  // CVLIB_CALIB3D_TRANSFORMS_H_
