// Polygon geometry APIs.

#ifndef CVLIB_GEOMETRY_POLYGON_H_
#define CVLIB_GEOMETRY_POLYGON_H_

#include "types.h"
#include "error_codes.h"

namespace cvlib {
namespace geometry {

/*
Computes 2D convex hull vertices in counter-clockwise order.

@param points Input points, N-by-2 (N >= 1).
@param hull_out Output hull points, capacity M-by-2; must satisfy M >= 1
  and M >= number of returned hull vertices. Unused rows are left untouched.
@param hull_size Output number of hull vertices written into hull_out.
@returns ErrorCode.
*/
ErrorCode convex_hull(const Matrix* points, Matrix* hull_out,
                      int32_t* hull_size);

/*
Tests if point is outside, on boundary, or inside polygon.

@param polygon Input polygon vertices, N-by-2 (N >= 3).
@param point Input query point, length 2.
@param result Output classification: -1 outside, 0 on edge, 1 inside.
@returns ErrorCode.
*/
ErrorCode point_in_polygon(const Matrix* polygon, const Vector* point,
                           int32_t* result);

/*
Computes the signed area of a simple polygon using the shoelace formula.

@param polygon Input polygon vertices, N-by-2 (N >= 3).
@param area Output signed area; positive for CCW, negative for CW order.
@returns ErrorCode.
*/
ErrorCode polygon_area(const Matrix* polygon, float64_t* area);

/*
Computes the centroid (geometric center) of a simple polygon.

@param polygon Input polygon vertices, N-by-2 (N >= 3).
@param centroid_out Output centroid coordinates, length 2.
@returns ErrorCode.
*/
ErrorCode polygon_centroid(const Matrix* polygon, Vector* centroid_out);

/*
Computes the perimeter (sum of edge lengths) of a closed polygon.

@param polygon Input polygon vertices, N-by-2 (N >= 2).
@param perimeter Output total perimeter length.
@returns ErrorCode.
*/
ErrorCode polygon_perimeter(const Matrix* polygon, float64_t* perimeter);

}  // namespace geometry
}  // namespace cvlib

#endif  // CVLIB_GEOMETRY_POLYGON_H_
