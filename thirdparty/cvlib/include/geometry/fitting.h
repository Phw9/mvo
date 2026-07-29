// Geometry fitting APIs.

#ifndef CVLIB_GEOMETRY_FITTING_H_
#define CVLIB_GEOMETRY_FITTING_H_

#include "../types.h"
#include "../error_codes.h"
#include "../ransac_params.h"

#include <cstdint>

namespace cvlib {
namespace geometry {

// Shared RANSAC parameters; see cvlib/ransac_params.h.
using RansacParams = ::cvlib::RansacParams;

/*
Fits a 3D plane ax + by + cz + d = 0 with ||[a,b,c]|| = 1.

@param points Input points, N-by-3 (N >= 3, not all collinear or coincident).
@param plane_out Output plane coefficients [a,b,c,d], length 4.
@returns ErrorCode (kSingularMatrix when the points are collinear or
         coincident, since the plane normal is then only one of infinitely
         many equally valid choices).
*/
ErrorCode fit_plane(const Matrix* points, Vector* plane_out);

/*
Fits a 3D line by PCA and returns a point and unit direction.

@param points Input points, N-by-3 (N >= 2, not all coincident).
@param point_out Output point on line, length 3.
@param dir_out Output unit direction, length 3.
@returns ErrorCode (kSingularMatrix when the points coincide).
*/
ErrorCode fit_line_3d(const Matrix* points, Vector* point_out, Vector* dir_out);

/*
Fits a 2D line ax + by + c = 0 with ||[a,b]|| = 1 by total least squares.

@param points Input points, N-by-2 (N >= 2, not all coincident).
@param line_out Output line coefficients [a,b,c], length 3.
@returns ErrorCode (kSingularMatrix when the points coincide).
*/
ErrorCode fit_line_2d(const Matrix* points, Vector* line_out);

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
@returns ErrorCode (kSingularMatrix when the unconstrained fit is not an
         ellipse, i.e. B^2-4AC >= 0, which degenerate or near-degenerate
         point sets can produce).
*/
ErrorCode fit_ellipse(const Matrix* points, Vector* coeff_out);

/*
Converts ellipse conic coefficients [A,B,C,D,E,F]
(A x^2 + B x y + C y^2 + D x + E y + F = 0) to geometric parameters via the
eigen-decomposition of [[A, B/2],[B/2, C]].

@param coeff Conic coefficients, length 6; must describe an ellipse
       (B^2 - 4AC < 0).
@param center_out Output centre [x, y], length 2.
@param semi_major_out Output semi-major axis length.
@param semi_minor_out Output semi-minor axis length.
@param angle_out Output major-axis angle in radians (defined mod pi).
@returns ErrorCode (kInvalidArgument when the conic is not an ellipse).
*/
ErrorCode ellipse_params(const Vector* coeff, Vector* center_out,
                         float64_t* semi_major_out, float64_t* semi_minor_out,
                         float64_t* angle_out);

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

/*
RANSAC 2D line fitting: returns the consensus line and inlier mask.

@param points Input points, N-by-2 (N >= 2).
@param params RANSAC parameters; inlier_thresh is the perpendicular distance.
@param line_out Output line [a, b, c], length 3 (refit on inliers).
@param inlier_mask Output 0/1 mask per input row, length N (may be nullptr).
@param num_inliers Output number of inliers used for the final fit.
@returns ErrorCode (kNotConverged when no consensus reaches min_inliers).
*/
ErrorCode fit_line_ransac(const Matrix* points, RansacParams params,
                          Vector* line_out, int32_t* inlier_mask,
                          int32_t* num_inliers);

/*
Fits a sphere with the algebraic least-squares linearization of
|x - c|^2 = r^2.

@param points Input points, N-by-3 (N >= 4, not coplanar).
@param center_out Output sphere center, length 3.
@param radius_out Output radius (> 0).
@returns ErrorCode (kSingularMatrix for degenerate point sets).
*/
ErrorCode fit_sphere(const Matrix* points, Vector* center_out,
                     float64_t* radius_out);

}  // namespace geometry
}  // namespace cvlib

#endif  // CVLIB_GEOMETRY_FITTING_H_
