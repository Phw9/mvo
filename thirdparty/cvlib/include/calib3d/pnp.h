// Perspective-n-Point: DLT initialization and LM refinement.

#ifndef CVLIB_CALIB3D_PNP_H_
#define CVLIB_CALIB3D_PNP_H_

#include "types.h"
#include "error_codes.h"
#include "optimize/lm.h"

#include <cstdint>

namespace cvlib {
namespace calib3d {

static constexpr int32_t kPnpMinPoints = 6;

// Options for solve_pnp.
struct SolvePnpOptions {
    int32_t              perturb_mode;
    optimize::LMOptions  lm;
};

SolvePnpOptions default_solve_pnp_options();

/*
Direct Linear Transform PnP estimation.
Computes (R, t) from N >= 6 known correspondences.
World points are 3D (N x 3); image points are pixel coords (N x 2).
Outputs use world-to-camera extrinsics: p_cam = R * p_world + t.

@param world_pts N-by-3 world points.
@param image_pts N-by-2 pixel observations.
@param k Camera intrinsics (3x3).
@param r_3x3 Output rotation (pre-allocated 3x3).
@param t_3 Output translation (pre-allocated length 3).
@param dist_coeff Optional distortion coefficients; null for no distortion.
@returns ErrorCode.
*/
ErrorCode dlt_pnp(const Matrix* world_pts, const Matrix* image_pts,
                  const Matrix* k,
                  Matrix* r_3x3, Vector* t_3,
                  const Vector* dist_coeff = nullptr);

/*
Solves PnP using LM refinement on SE(3); auto-initializes with DLT when
no initial pose is supplied.
Outputs use world-to-camera extrinsics: p_cam = R * p_world + t.

@param world_pts N-by-3 world points.
@param image_pts N-by-2 pixel observations.
@param k Camera intrinsics (3x3).
@param r_out Output rotation (3x3, must be allocated).
@param t_out Output translation (length 3, must be allocated).
@param dist_coeff Optional distortion coefficients; null for no distortion.
@param r_init Optional initial rotation; null triggers DLT initialization.
@param t_init Optional initial translation; null triggers DLT initialization.
@param options Optional solver options; null uses defaults.
@param report Optional output optimization report; may be null.
@returns ErrorCode.
*/
ErrorCode solve_pnp(const Matrix* world_pts, const Matrix* image_pts,
                    const Matrix* k,
                    Matrix* r_out, Vector* t_out,
                    const Vector* dist_coeff = nullptr,
                    const Matrix* r_init = nullptr,
                    const Vector* t_init = nullptr,
                    const SolvePnpOptions* options = nullptr,
                    optimize::OptimizeReport* report = nullptr);

}  // namespace calib3d
}  // namespace cvlib

#endif  // CVLIB_CALIB3D_PNP_H_
