// Geometry primitives: distances, intersections.

#ifndef CVLIB_GEOMETRY_PRIMITIVES_H_
#define CVLIB_GEOMETRY_PRIMITIVES_H_

#include "cvlib/types.h"
#include "cvlib/error_codes.h"

namespace cvlib {
namespace geometry {

/*
Computes the perpendicular distance from a 2D/3D point to an infinite line.
The line is parameterised as point + t * direction.

@param point Query point, length 2 or 3.
@param line_point Point on the line, same length as point.
@param line_dir Line direction vector (need not be unit), same length as point.
@param distance_out Output non-negative distance.
@returns ErrorCode.
*/
ErrorCode point_to_line_distance(const Vector* point,
                                 const Vector* line_point,
                                 const Vector* line_dir,
                                 float64_t* distance_out);

/*
Computes the signed distance from a 3D point to a plane ax + by + cz + d = 0.

@param point Query point, length 3.
@param plane Plane coefficients [a, b, c, d], length 4 (need not be unit).
@param distance_out Output signed distance (positive on the normal side).
@returns ErrorCode.
*/
ErrorCode point_to_plane_distance(const Vector* point, const Vector* plane,
                                  float64_t* distance_out);

/*
Finds the intersection of two 3D lines via the closest-point pair midpoint.
Returns kSingularMatrix when the lines are parallel within tolerance.

@param p1 Point on first line, length 3.
@param d1 Direction of first line, length 3.
@param p2 Point on second line, length 3.
@param d2 Direction of second line, length 3.
@param point_out Output intersection (or midpoint of closest pair), length 3.
@returns ErrorCode.
*/
ErrorCode line_line_intersection(const Vector* p1, const Vector* d1,
                                 const Vector* p2, const Vector* d2,
                                 Vector* point_out);

/*
Intersects a ray (origin + t * direction, t >= 0) with a plane.
Returns kUnreachableTarget if the ray is parallel to the plane or the hit
lies behind the origin.

@param origin Ray origin, length 3.
@param direction Ray direction, length 3 (need not be unit).
@param plane Plane coefficients [a, b, c, d], length 4.
@param point_out Output intersection point, length 3.
@returns ErrorCode.
*/
ErrorCode ray_plane_intersection(const Vector* origin, const Vector* direction,
                                 const Vector* plane, Vector* point_out);

}  // namespace geometry
}  // namespace cvlib

#endif  // CVLIB_GEOMETRY_PRIMITIVES_H_
