// Shared numeric constants.

#ifndef CVLIB_DEFS_H_
#define CVLIB_DEFS_H_

#include <cstdint>

namespace cvlib {

using float64_t = double;
using float32_t = float;

// Typical geometry dimensions.
static constexpr int32_t kRotationMatrixSize   = 3;
static constexpr int32_t kTranslationVectorSize = 3;
static constexpr int32_t kTransformMatrixSize   = 4;
static constexpr int32_t kPoint3dSize           = 3;
static constexpr int32_t kPoint2dSize           = 2;

static constexpr float64_t kEpsilon = 1e-8;
static constexpr float64_t kPi      = 3.14159265358979323846;
static constexpr float64_t kTwoPi   = 6.28318530717958647692;
static constexpr float64_t kHalfPi  = 1.57079632679489661923;

// Numerical thresholds

static constexpr float64_t kSingularityThreshold = 1e-5;
static constexpr float64_t kDefaultTolerance     = 1e-10;
static constexpr int32_t   kDefaultMaxIterations  = 100;
static constexpr int32_t   kSvdMaxIterations      = 1000;

// Image processing constants

static constexpr int32_t kRgbChannels       = 3;
static constexpr int32_t kGrayscaleChannels = 1;
static constexpr int32_t kMaxChannels       = 4;

// Calibration constants

static constexpr int32_t   kMinCalibrationImages = 3;
static constexpr int32_t   kMinCalibrationPoints = 4;
static constexpr float64_t kDefaultSquareSize    = 1.0;

// Camera constants

static constexpr int32_t   kCameraMatrixSize     = 3;
static constexpr int32_t   kMaxDistortionCoeffs  = 8;
static constexpr int32_t   kQuaternionSize       = 4;
static constexpr float64_t kPositiveDepthEpsilon = 1e-9;

// Lie algebra constants

static constexpr int32_t   kSe3VectorSize               = 6;
static constexpr int32_t   kSo3VectorSize               = 3;
static constexpr float64_t kIdentityRotationThreshold    = 1e-8;
static constexpr float64_t kSmallAngleThreshold          = 1e-8;
static constexpr float64_t kPiRotationThreshold          = 1e-8;
static constexpr float64_t kTraceNormalizationFactor     = 0.5;
static constexpr float64_t kRodriguesMatrixScale         = 2.0;
static constexpr float64_t kHalfAngleDivisor             = 2.0;
static constexpr float64_t kSe3SmallAngleThreshold       = 1e-6;
static constexpr float64_t kSe3VerySmallAngleThreshold   = 1e-12;
static constexpr float64_t kSe3VerySmallCubeThreshold    = 1e-18;

// Metric constants

static constexpr int32_t   kMetricMaxIterations    = 100;
static constexpr float64_t kConvergenceThreshold   = 1e-8;
static constexpr float64_t kDefaultWeightTolerance = 1e-10;

// Kinematics constants

static constexpr int32_t   kDhParamCols      = 4;
static constexpr float64_t kPlanarThreshold   = 1e-6;
static constexpr float64_t kDampingFactor     = 0.5;
static constexpr float64_t kMinDampingFactor  = 0.1;
static constexpr float64_t kDampingDecay      = 0.9;
static constexpr float64_t kConditionLimit    = 1e12;

// Filter constants

static constexpr float64_t kWeightSumTolerance = 1e-6;
static constexpr float64_t kUkfDefaultAlpha    = 1e-3;
static constexpr float64_t kUkfDefaultBeta     = 2.0;
static constexpr float64_t kUkfDefaultKappa    = 0.0;
static constexpr float64_t kPendulumDt         = 0.01;
static constexpr float64_t kDefaultGravity     = 9.81;
static constexpr float64_t kDefaultLength      = 1.0;
static constexpr float64_t kDefaultMass        = 1.0;
static constexpr float64_t kDivisionGuard      = 1e-10;

}  // namespace cvlib

#endif  // CVLIB_DEFS_H_
