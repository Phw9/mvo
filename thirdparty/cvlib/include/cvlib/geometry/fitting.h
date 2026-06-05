// Geometry fitting APIs.

#ifndef CVLIB_GEOMETRY_FITTING_H_
#define CVLIB_GEOMETRY_FITTING_H_

#include "cvlib/types.h"
#include "cvlib/error_codes.h"

#include <cstdint>

namespace cvlib {
namespace geometry {

// Tunable parameters for RANSAC-based robust fitters.
struct RansacParams {
    int32_t   max_iters;
    float64_t inlier_thresh;
    int32_t   min_inliers;
    uint32_t  seed;
};

/*
Fits a 3D plane ax + by + cz + d = 0 with ||[a,b,c]|| = 1.

@param points Input points, N-by-3 (N >= 3).
@param plane_out Output plane coefficients [a,b,c,d], length 4.
@returns ErrorCode.
*/
ErrorCode fit_plane(const Matrix* points, Vector* plane_out);

/*
Fits a 3D line by PCA and returns a point and unit direction.

@param points Input points, N-by-3 (N >= 2).
@param point_out Output point on line, length 3.
@param dir_out Output unit direction, length 3.
@returns ErrorCode.
*/
ErrorCode fit_line_3d(const Matrix* points, Vector* point_out, Vector* dir_out);

/*
Fits a 2D circle x^2+y^2+ax+by+c=0 from points.

@param points Input points, N-by-2 (N >= 3).
@param center_out Output center [cx, cy], length 2.
@param radius_out Output radius.
@returns ErrorCode.
*/
ErrorCode fit_circle(const Matrix* points, Vector* center_out,
                     float64_t* radius_out);

/*
Fits a 2D ellipse with direct least-squares constraint.

@param points Input points, N-by-2 (N >= 5).
@param coeff_out Output conic coeff [A,B,C,D,E,F], length 6.
@returns ErrorCode.
*/
ErrorCode fit_ellipse(const Matrix* points, Vector* coeff_out);

/*
RANSAC plane fitting: returns the consensus plane and inlier mask.

@param points Input points, N-by-3 (N >= 3).
@param params RANSAC parameters; inlier_thresh is the maximum signed distance.
@param plane_out Output plane [a, b, c, d], length 4 (refit on inliers).
@param inlier_mask Output 0/1 mask per input row, length N (may be nullptr).
@param num_inliers Output number of inliers used for the final fit.
@returns ErrorCode (kNotConverged when no consensus reaches min_inliers).
*/
ErrorCode fit_plane_ransac(const Matrix* points, RansacParams params,
                           Vector* plane_out, int32_t* inlier_mask,
                           int32_t* num_inliers);

/*
RANSAC circle fitting: returns the consensus circle and inlier mask.

@param points Input points, N-by-2 (N >= 3).
@param params RANSAC parameters; inlier_thresh is the absolute radial residual.
@param center_out Output centre [cx, cy], length 2 (refit on inliers).
@param radius_out Output radius (refit on inliers).
@param inlier_mask Output 0/1 mask per input row, length N (may be nullptr).
@param num_inliers Output number of inliers used for the final fit.
@returns ErrorCode (kNotConverged when no consensus reaches min_inliers).
*/
ErrorCode fit_circle_ransac(const Matrix* points, RansacParams params,
                            Vector* center_out, float64_t* radius_out,
                            int32_t* inlier_mask, int32_t* num_inliers);

}  // namespace geometry
}  // namespace cvlib

#endif  // CVLIB_GEOMETRY_FITTING_H_
