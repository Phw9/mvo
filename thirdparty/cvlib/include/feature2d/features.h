// Feature2D implementation status records.

#ifndef CVLIB_FEATURE2D_FEATURES_H_
#define CVLIB_FEATURE2D_FEATURES_H_

#include "error_codes.h"
#include "defs.h"

#include <cstdint>
#include <vector>

namespace cvlib {
namespace feature2d {

static constexpr int32_t kOrbDescriptorBytes = 32;
static constexpr int32_t kSiftDescriptorSize = 128;

/*
Read-only grayscale image view.

@param data Row-major grayscale samples.
@param rows Image row count.
@param cols Image column count.
@param stride Elements between consecutive rows.
*/
struct FeatureImageView {
    const float64_t* data = nullptr;
    int32_t rows = 0;
    int32_t cols = 0;
    int32_t stride = 0;
};

/*
Optional read-only feature mask view.

@param data Row-major mask samples, nonzero entries are enabled.
@param rows Mask row count.
@param cols Mask column count.
@param stride Elements between consecutive rows.
*/
struct FeatureMaskView {
    const uint8_t* data = nullptr;
    int32_t rows = 0;
    int32_t cols = 0;
    int32_t stride = 0;
};

/*
Two-dimensional feature keypoint.

@param x Horizontal pixel coordinate.
@param y Vertical pixel coordinate.
@param size Diameter of the meaningful keypoint region.
@param angle Orientation in degrees.
@param response Detector response strength.
@param octave Pyramid octave.
@param class_id User-defined keypoint class id.
*/
struct Keypoint {
    float64_t x = 0.0;
    float64_t y = 0.0;
    float64_t size = 0.0;
    float64_t angle = 0.0;
    float64_t response = 0.0;
    int32_t octave = 0;
    int32_t class_id = -1;
};

/*
ORB detector and descriptor parameters.

@param nfeatures Maximum number of output keypoints.
@param scale_factor Pyramid scale factor.
@param nlevels Pyramid level count.
@param edge_threshold Border excluded from detection.
@param first_level Reserved for OpenCV API compatibility.
@param wta_k Reserved; only OpenCV-compatible WTA_K=2 is supported.
@param use_harris_score Use Harris response ranking when true.
@param patch_size Descriptor and orientation patch size.
@param fast_threshold FAST-9 intensity threshold.
*/
struct OrbParameters {
    int32_t nfeatures = 500;
    float64_t scale_factor = 1.2;
    int32_t nlevels = 8;
    int32_t edge_threshold = 31;
    int32_t first_level = 0;
    int32_t wta_k = 2;
    bool use_harris_score = true;
    int32_t patch_size = 31;
    int32_t fast_threshold = 20;
};

/*
SIFT detector and descriptor parameters.

@param nfeatures Maximum output keypoints, or 0 for no explicit cap.
@param n_octave_layers Gaussian layers per octave.
@param contrast_threshold DoG contrast threshold.
@param edge_threshold Principal curvature edge threshold.
@param sigma Base Gaussian sigma.
*/
struct SiftParameters {
    int32_t nfeatures = 0;
    int32_t n_octave_layers = 3;
    float64_t contrast_threshold = 0.04;
    float64_t edge_threshold = 10.0;
    float64_t sigma = 1.6;
};

/*
Describes one feature2d implementation and its OpenCV reference.

@param name Feature family name.
@param opencv_reference Reference OpenCV API used for equivalence tests.
@param status Current implementation status.
@param notes Scope and comparison notes.
*/
struct FeatureTodo {
    const char* name;
    const char* opencv_reference;
    const char* status;
    const char* notes;
};

/*
Returns the number of planned feature2d items.

@returns Number of FeatureTodo records.
*/
int32_t feature_todo_count();

/*
Copies a feature2d status item by index.

@param index Zero-based item index.
@param todo Output record pointer.
@returns ErrorCode.
*/
ErrorCode feature_todo_at(int32_t index, FeatureTodo* todo);

/*
Returns OpenCV-compatible ORB defaults.

@returns OrbParameters.
*/
OrbParameters orb_default_parameters();

/*
Detects ORB keypoints and computes binary descriptors.

@param image Input grayscale image.
@param mask Optional mask; pass nullptr to enable all pixels.
@param parameters ORB parameters.
@param keypoints Output keypoints.
@param descriptors Output row-major descriptors, N-by-32.
@returns ErrorCode.
*/
ErrorCode orb_detect_and_compute(const FeatureImageView* image,
                                 const FeatureMaskView* mask,
                                 const OrbParameters* parameters,
                                 std::vector<Keypoint>* keypoints,
                                 std::vector<uint8_t>* descriptors);

/*
Returns OpenCV-compatible SIFT defaults.

@returns SiftParameters.
*/
SiftParameters sift_default_parameters();

/*
Detects SIFT keypoints and computes float descriptors.

@param image Input grayscale image.
@param mask Optional mask; pass nullptr to enable all pixels.
@param parameters SIFT parameters.
@param keypoints Output keypoints.
@param descriptors Output row-major descriptors, N-by-128.
@returns ErrorCode.
*/
ErrorCode sift_detect_and_compute(const FeatureImageView* image,
                                  const FeatureMaskView* mask,
                                  const SiftParameters* parameters,
                                  std::vector<Keypoint>* keypoints,
                                  std::vector<float32_t>* descriptors);

}  // namespace feature2d
}  // namespace cvlib

#endif  // CVLIB_FEATURE2D_FEATURES_H_
