// SO(3) / SE(3) geodesic distances and Fréchet means.

#ifndef CVLIB_CALIB3D_METRICS_H_
#define CVLIB_CALIB3D_METRICS_H_

#include "types.h"
#include "defs.h"
#include "error_codes.h"

#include <cstdint>

namespace cvlib {
namespace calib3d {

/*
Computes the geodesic angle between two rotations on SO(3).

@param R1 Input 3-by-3 rotation.
@param R2 Input 3-by-3 rotation.
@param result Output angle in radians.
@returns ErrorCode.

*/

ErrorCode so3_dist(const Matrix* R1, const Matrix* R2,
                                float64_t* result);

/*
Computes a left-invariant distance between two rigid transforms on SE(3).

@param T1 Input 4-by-4 transform.
@param T2 Input 4-by-4 transform.
@param result Output nonnegative distance.
@returns ErrorCode.

*/

ErrorCode se3_dist(const Matrix* T1, const Matrix* T2,
                                float64_t* result);

// Default Karcher-mean convergence tolerance (iter cap uses defs.h).

static constexpr float64_t kMeanDefaultTolerance = 1e-8;

/*
Computes a Karcher (Fréchet) mean of rotations on SO(3).

@param rotations Stacked or blocked rotation storage; count rotations total.
@param count Number of 3-by-3 rotations.
@param result Output 3-by-3 mean rotation; must be pre-allocated.
@param weights Optional nonnegative weights, length count; null for uniform.
@param max_iter Maximum iterations (default kMetricMaxIterations).
@param tol Convergence tolerance (default kMeanDefaultTolerance).
@returns ErrorCode.

*/

ErrorCode so3_mean(const Matrix* rotations, int32_t count, Matrix* result,
                   const float64_t* weights = nullptr,
                   int32_t max_iter = kMetricMaxIterations,
                   float64_t tol = kMeanDefaultTolerance);

/*
Computes a Karcher mean of rigid transforms on SE(3).

@param transformations Stacked 4-by-4 transforms; count samples total.
@param count Number of transforms.
@param result Output 4-by-4 mean; must be pre-allocated.
@param weights Optional nonnegative weights, length count; null for uniform.
@param max_iter Maximum iterations (default kMetricMaxIterations).
@param tol Convergence tolerance (default kMeanDefaultTolerance).
@returns ErrorCode.

*/

ErrorCode se3_mean(const Matrix* transformations, int32_t count,
                   Matrix* result,
                   const float64_t* weights = nullptr,
                   int32_t max_iter = kMetricMaxIterations,
                   float64_t tol = kMeanDefaultTolerance);

/*
Fills pairwise SO(3) geodesic distances for count rotations.

@param rotations Stacked rotations as required by the implementation.
@param count Number of rotations.
@param result Output count-by-count matrix; must be pre-allocated.
@returns ErrorCode.

*/

ErrorCode so3_pdist(const Matrix* rotations, int32_t count,
                                 Matrix* result);

/*
Fills pairwise SE(3) distances for count transforms.

@param transformations Stacked 4-by-4 transforms.
@param count Number of transforms.
@param result Output count-by-count matrix; must be pre-allocated.
@returns ErrorCode.

*/

ErrorCode se3_pdist(const Matrix* transformations, int32_t count,
                                 Matrix* result);

/*
Computes weighted variance of rotations about a reference mean on SO(3).

@param rotations Sample rotations; count samples.
@param count Number of samples.
@param result Output scalar variance.
@param mean_rot Optional reference 3-by-3 mean; null computes one internally.
@param weights Optional weights, length count; null for uniform.
@returns ErrorCode.

*/

ErrorCode so3_var(const Matrix* rotations, int32_t count, float64_t* result,
                  const Matrix* mean_rot = nullptr,
                  const float64_t* weights = nullptr);

/*
Computes weighted variance of transforms about a reference mean on SE(3).

@param transformations Sample 4-by-4 transforms; count samples.
@param count Number of samples.
@param result Output scalar variance.
@param mean_tf Optional reference 4-by-4 mean; null computes one internally.
@param weights Optional weights, length count; null for uniform.
@returns ErrorCode.

*/

ErrorCode se3_var(const Matrix* transformations, int32_t count,
                  float64_t* result,
                  const Matrix* mean_tf = nullptr,
                  const float64_t* weights = nullptr);

}  // namespace calib3d
}  // namespace cvlib

#endif  // CVLIB_CALIB3D_METRICS_H_
