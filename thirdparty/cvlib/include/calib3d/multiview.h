// Multi-view geometry: triangulation, epipolar, homography, rigid alignment.

#ifndef CVLIB_CALIB3D_MULTIVIEW_H_
#define CVLIB_CALIB3D_MULTIVIEW_H_

#include "../types.h"
#include "../error_codes.h"

#include <cstdint>

namespace cvlib {
namespace calib3d {

// Tunable parameters for RANSAC-based robust two-view estimators.
struct RansacParams {
    int32_t   max_iters;
    float64_t inlier_thresh;
    int32_t   min_inliers;
    uint32_t  seed;
};

/*
Triangulates 3D points from two projection matrices and matched image points.

@param p1 First 3-by-4 projection matrix.
@param p2 Second 3-by-4 projection matrix.
@param x1 First-view points, N-by-2.
@param x2 Second-view points, N-by-2.
@param points_3d Output triangulated points, N-by-3.
@returns ErrorCode.
*/
ErrorCode triangulate_points(const Matrix* p1, const Matrix* p2,
                             const Matrix* x1, const Matrix* x2,
                             Matrix* points_3d);

/*
Marks triangulated points as inliers when both views see positive depth.

@param points_3d Triangulated world points, N-by-3 (camera-1 frame).
@param p2 Second 3-by-4 projection matrix in the same world frame as p1=I.
@param mask Output length-N array; entry is 1 when both depths are positive.
@param num_valid Output count of inliers (0..N).
@returns ErrorCode.
*/
ErrorCode filter_chirality(const Matrix* points_3d, const Matrix* p2,
                           int32_t* mask, int32_t* num_valid);

/*
Estimates a homography H from point correspondences using normalized DLT.

@param src Source points, N-by-2 (N >= 4).
@param dst Destination points, N-by-2 (N >= 4).
@param h_out Output 3-by-3 homography (up to scale).
@returns ErrorCode.
*/
ErrorCode find_homography(const Matrix* src, const Matrix* dst, Matrix* h_out);

/*
Estimates a fundamental matrix F from correspondences using normalized 8-point.

@param x1 First-view points, N-by-2 (N >= 8).
@param x2 Second-view points, N-by-2 (N >= 8).
@param f_out Output 3-by-3 fundamental matrix (rank-2, up to scale).
@returns ErrorCode.
*/
ErrorCode find_fundamental_matrix(const Matrix* x1, const Matrix* x2,
                                  Matrix* f_out);

/*
Computes essential matrix E = K^T F K and enforces rank-2 with equal top SVs.

@param x1 First-view points, N-by-2.
@param x2 Second-view points, N-by-2.
@param k Camera intrinsics (3x3).
@param e_out Output 3-by-3 essential matrix.
@returns ErrorCode.
*/
ErrorCode find_essential_matrix(const Matrix* x1, const Matrix* x2,
                                const Matrix* k, Matrix* e_out);

/*
Decomposes essential matrix into two rotations and one translation direction.

@param e Essential matrix, 3-by-3.
@param r1_out First 3-by-3 rotation candidate.
@param r2_out Second 3-by-3 rotation candidate.
@param t_out Translation direction, length 3.
@returns ErrorCode.
*/
ErrorCode decompose_essential_matrix(const Matrix* e,
                                     Matrix* r1_out, Matrix* r2_out,
                                     Vector* t_out);

/*
Selects the (R, t) pair that yields the largest count of triangulated points
with positive depth in both cameras (chirality test).

Tries all four combinations from {R1, R2} x {+t, -t} using identity for the
first camera and (R, t) for the second.

@param e Essential matrix, 3-by-3.
@param x1 First-view points, N-by-2 (already in normalized camera coordinates).
@param x2 Second-view points, N-by-2 (already in normalized camera coordinates).
@param r_out Output selected 3-by-3 rotation.
@param t_out Output selected unit translation, length 3.
@returns ErrorCode.
*/
ErrorCode recover_pose_from_essential(const Matrix* e, const Matrix* x1,
                                      const Matrix* x2, Matrix* r_out,
                                      Vector* t_out);

/*
Faugeras-Malis style decomposition of a Euclidean homography returning up to
four physically valid (R, t, n) candidates. Use chirality and visibility tests
externally to pick the right one.

@param h Homography matrix, 3-by-3.
@param k Camera intrinsics, 3-by-3.
@param r_candidates Output rotations stacked vertically, 12-by-3 (4 x 3).
@param t_candidates Output translations stacked vertically, 4-by-3.
@param n_candidates Output plane normals stacked vertically, 4-by-3.
@param num_solutions Output number of valid solutions written (0..4).
@returns ErrorCode.
*/
ErrorCode decompose_homography(const Matrix* h, const Matrix* k,
                               Matrix* r_candidates, Matrix* t_candidates,
                               Matrix* n_candidates, int32_t* num_solutions);

/*
Computes rigid transform (Kabsch) that aligns src to dst: dst ~= R src + t.

@param src Source 3D points, N-by-3.
@param dst Destination 3D points, N-by-3.
@param r_out Output 3-by-3 rotation.
@param t_out Output translation, length 3.
@returns ErrorCode.
*/
ErrorCode rigid_transform_3d(const Matrix* src, const Matrix* dst,
                             Matrix* r_out, Vector* t_out);

/*
RANSAC homography estimator. Refits the consensus inlier set with the
normalised DLT used by find_homography.

@param src Source points, N-by-2.
@param dst Destination points, N-by-2.
@param params RANSAC parameters; inlier_thresh is symmetric pixel error.
@param h_out Output 3-by-3 homography (refit on inliers).
@param inlier_mask Optional output mask, length N (0/1 per row).
@param num_inliers Output number of inliers used in the refit.
@returns ErrorCode.
*/
ErrorCode find_homography_ransac(const Matrix* src, const Matrix* dst,
                                 RansacParams params, Matrix* h_out,
                                 int32_t* inlier_mask, int32_t* num_inliers);

/*
RANSAC fundamental-matrix estimator. Refits the consensus inlier set with the
normalised 8-point algorithm used by find_fundamental_matrix.

@param x1 First-view points, N-by-2.
@param x2 Second-view points, N-by-2.
@param params RANSAC parameters; inlier_thresh is the Sampson distance.
@param f_out Output 3-by-3 fundamental matrix (refit on inliers).
@param inlier_mask Optional output mask, length N (0/1 per row).
@param num_inliers Output number of inliers used in the refit.
@returns ErrorCode.
*/
ErrorCode find_fundamental_ransac(const Matrix* x1, const Matrix* x2,
                                  RansacParams params, Matrix* f_out,
                                  int32_t* inlier_mask, int32_t* num_inliers);

}  // namespace calib3d
}  // namespace cvlib

#endif  // CVLIB_CALIB3D_MULTIVIEW_H_
