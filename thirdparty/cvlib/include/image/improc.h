// Image processing on the image core types. Single-channel operations
// accept any supported pixel format (converted internally to float64)
// and write pre-created float64 outputs unless documented otherwise.

#ifndef CVLIB_IMAGE_IMPROC_H_
#define CVLIB_IMAGE_IMPROC_H_

#include "../types.h"
#include "../error_codes.h"
#include "../image/image.h"

#include <cstdint>

namespace cvlib {
namespace image {

/*
Converts a 3- or 4-channel colour image to single-channel luminance using
the weights 0.299/0.587/0.114 over channels 0, 1, 2 (R, G, B); any extra
channel is ignored. Bridges a colour image into the single-channel
improc / feature2d paths.

@param src Source view, 3 or 4 channels, any pixel format.
@param dst Output image, same rows/cols, single channel, any format;
       pre-created. Values are clamped/rounded on write for integer depths.
@returns ErrorCode.
*/
ErrorCode rgb2gray(const ImageView* src, Image* dst);

/*
Halves image resolution: blurs with the separable binomial kernel
[1 4 6 4 1]/16 (reflected borders) then keeps every second sample.

@param src Source view, single channel, any pixel format.
@param dst Output image, ((rows+1)/2)-by-((cols+1)/2), single channel, any
       format; pre-created.
@returns ErrorCode.
*/
ErrorCode pyr_down(const ImageView* src, Image* dst);

/*
Separable convolution: filters rows with kernel_x and columns with
kernel_y (both odd-length), with reflected borders (the edge sample is
not duplicated).

@param src Source view, single channel, any format.
@param kernel_x Horizontal kernel, odd length >= 1.
@param kernel_y Vertical kernel, odd length >= 1.
@param dst Output image, same rows/cols, single channel, k64FC1.
@returns ErrorCode.
*/
ErrorCode separable_filter(const ImageView* src, const Vector* kernel_x,
                           const Vector* kernel_y, Image* dst);

/*
Box blur: separable uniform average over an odd window.

@param src Source view, single channel.
@param window Odd window size >= 1.
@param dst Output image, same rows/cols, single channel, k64FC1.
@returns ErrorCode.
*/
ErrorCode box_blur(const ImageView* src, int32_t window, Image* dst);

/*
Gaussian blur with the sigma-derived kernel shared with the feature2d
detectors, so smoothing here and inside ORB/SIFT stay identical.

@param src Source view, single channel.
@param sigma Gaussian sigma (> 0).
@param dst Output image, same rows/cols, single channel, k64FC1.
@returns ErrorCode.
*/
ErrorCode gaussian_blur(const ImageView* src, float64_t sigma, Image* dst);

/*
First-order image gradients with the unnormalized 3x3 Sobel kernels
(smoothing [1 2 1] times difference [-1 0 1]), built on
separable_filter with reflected borders.

@param src Source view, single channel.
@param grad_x Output horizontal gradient, k64FC1; pre-created.
@param grad_y Output vertical gradient, k64FC1; pre-created.
@returns ErrorCode.
*/
ErrorCode sobel(const ImageView* src, Image* grad_x, Image* grad_y);

/*
First-order image gradients with the 3x3 Scharr kernels shared with the
KLT tracker (including its 1/32 normalization).

@param src Source view, single channel.
@param grad_x Output horizontal gradient, k64FC1; pre-created.
@param grad_y Output vertical gradient, k64FC1; pre-created.
@returns ErrorCode.
*/
ErrorCode scharr(const ImageView* src, Image* grad_x, Image* grad_y);

/*
Integral image: out(r, c) is the inclusive sum of all samples above and
left of (r, c), with a zero top row and left column, so any rectangle
sum is four lookups.

@param src Source view, single channel.
@param dst Output image, (rows+1) x (cols+1), single channel,
       k64FC1.
@returns ErrorCode.
*/
ErrorCode integral_image(const ImageView* src, Image* dst);

/*
Sample histogram over [range_min, range_max): equal-width bins, samples
outside the range are ignored, the last bin includes range_max.

@param src Source view, single channel.
@param range_min Lower range bound.
@param range_max Upper range bound (> range_min).
@param counts Output per-bin counts, length = bin count (>= 1);
       pre-allocated.
@returns ErrorCode.
*/
ErrorCode histogram(const ImageView* src, float64_t range_min,
                    float64_t range_max, Vector* counts);

// Threshold modes.
static constexpr int32_t kThresholdBinary    = 0;  // v > t ? max : 0
static constexpr int32_t kThresholdBinaryInv = 1;  // v > t ? 0 : max
static constexpr int32_t kThresholdTrunc     = 2;  // v > t ? t : v
static constexpr int32_t kThresholdToZero    = 3;  // v > t ? v : 0
static constexpr int32_t kThresholdToZeroInv = 4;  // v > t ? 0 : v

/*
Fixed-level thresholding.

@param src Source view, single channel.
@param thresh Threshold level.
@param max_value Output level for the binary modes.
@param mode One of the kThreshold* constants.
@param dst Output image, same rows/cols, single channel, k64FC1.
@returns ErrorCode.
*/
ErrorCode threshold(const ImageView* src, float64_t thresh,
                    float64_t max_value, int32_t mode, Image* dst);

/*
Bilinear resize: the output size is taken from dst; sample positions
map pixel centers proportionally, clamping at the borders.

@param src Source view, single channel.
@param dst Output image, any positive size, single channel, k64FC1.
@returns ErrorCode.
*/
ErrorCode resize(const ImageView* src, Image* dst);

/*
Generic remap: dst(r, c) samples src bilinearly at
(map_x(r, c), map_y(r, c)); samples outside the source are 0.

@param src Source view, single channel.
@param map_x Source x-coordinates per output pixel, dst-rows x dst-cols.
@param map_y Source y-coordinates per output pixel, same shape.
@param dst Output image, single channel, k64FC1.
@returns ErrorCode.
*/
ErrorCode remap(const ImageView* src, const Matrix* map_x,
                const Matrix* map_y, Image* dst);

/*
Affine warp: m is the 2x3 source-to-destination transform
[x'; y'] = M [x; y; 1]. The implementation inverts it and samples the
source bilinearly; destination pixels mapping outside are 0.

@param src Source view, single channel.
@param m Affine transform, 2-by-3 (invertible linear part).
@param dst Output image, single channel, k64FC1.
@returns ErrorCode.
*/
ErrorCode warp_affine(const ImageView* src, const Matrix* m, Image* dst);

/*
Perspective warp: h is the 3x3 source-to-destination homography. The
implementation inverts it and samples the source bilinearly;
destination pixels mapping outside (or behind the horizon, w <= 0)
are 0.

@param src Source view, single channel.
@param h Homography, 3-by-3 (invertible).
@param dst Output image, single channel, k64FC1.
@returns ErrorCode.
*/
ErrorCode warp_perspective(const ImageView* src, const Matrix* h,
                           Image* dst);

/*
Grayscale erosion: minimum over an odd square window with reflected
borders.

@param src Source view, single channel.
@param window Odd window size >= 1.
@param dst Output image, same rows/cols, single channel, k64FC1.
@returns ErrorCode.
*/
ErrorCode erode(const ImageView* src, int32_t window, Image* dst);

/*
Grayscale dilation: maximum over an odd square window with reflected
borders.

@param src Source view, single channel.
@param window Odd window size >= 1.
@param dst Output image, same rows/cols, single channel, k64FC1.
@returns ErrorCode.
*/
ErrorCode dilate(const ImageView* src, int32_t window, Image* dst);

/*
Morphological opening: erosion followed by dilation (removes bright
specks smaller than the window).
*/
ErrorCode morphology_open(const ImageView* src, int32_t window,
                          Image* dst);

/*
Morphological closing: dilation followed by erosion (fills dark holes
smaller than the window).
*/
ErrorCode morphology_close(const ImageView* src, int32_t window,
                           Image* dst);

/*
Edge detection: 3x3 sobel gradients, L2 magnitude, four-direction
non-maximum suppression, and double-threshold hysteresis with a
deterministic scan order. Edges are 255, background 0. The input is
not smoothed; blur beforehand for noisy images.

@param src Source view, single channel.
@param low_threshold Weak-edge magnitude threshold (>= 0).
@param high_threshold Strong-edge magnitude threshold
       (>= low_threshold).
@param dst Output edge map, same rows/cols, single channel, k64FC1.
@returns ErrorCode.
*/
ErrorCode canny(const ImageView* src, float64_t low_threshold,
                float64_t high_threshold, Image* dst);

}  // namespace image
}  // namespace cvlib

#endif  // CVLIB_IMAGE_IMPROC_H_
