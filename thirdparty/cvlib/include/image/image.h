// Image core: non-owning views and owning images with rows, cols,
// stride, interleaved channels, pixel format, and ROI sub-views.

#ifndef CVLIB_IMAGE_IMAGE_H_
#define CVLIB_IMAGE_IMAGE_H_

#include "../types.h"
#include "../error_codes.h"

#include <cstdint>

namespace cvlib {
namespace image {

// Images carry the shared depth+channels buffer type (see mat_type in
// cvlib/types.h); the default is 8-bit unsigned single channel. Samples
// are stored row-major with interleaved channels, and stride counts
// SAMPLES per row (stride >= cols * channels). Supported depths are
// k8U, k32F, and k64F.
static constexpr int32_t kMaxImageChannels = 4;

/*
Non-owning, read-only image view.

@param data First sample of the first pixel (depth-typed storage).
@param rows Row count (> 0).
@param cols Column count (> 0).
@param stride Samples per row (>= cols * channels).
@param type Buffer type (mat_type(depth, channels)); default k8UC1.
*/
struct ImageView {
    const void* data = nullptr;
    int32_t rows = 0;
    int32_t cols = 0;
    int32_t stride = 0;
    int32_t type = k8UC1;
};

/*
Owning image; create with image_create and release with image_destroy.
Storage is dense (stride == cols * channels).
*/
struct Image {
    void* data = nullptr;
    int32_t rows = 0;
    int32_t cols = 0;
    int32_t stride = 0;
    int32_t type = k8UC1;
};

/*
Returns the size of one sample (one channel element) of the buffer type
in bytes (0 for unknown depths).
*/
int32_t image_sample_bytes(int32_t type);

/*
Allocates a zero-filled dense image.

@param rows Row count (> 0).
@param cols Column count (> 0).
@param type Buffer type (mat_type(depth, channels)); depth in
       {k8U, k32F, k64F}, channels in 1..4.
@param out Output image; overwritten on success.
@returns ErrorCode.
*/
ErrorCode image_create(int32_t rows, int32_t cols, int32_t type,
                       Image* out);

/*
Releases image storage and resets the struct; safe on empty images.
*/
void image_destroy(Image* img);

/*
Borrows a read-only view of an owning image.
*/
ImageView image_view(const Image* img);

/*
Borrows a stride-preserving rectangular sub-view (ROI). The view
aliases the source storage; it stays valid only while the source does.

@param src Source view.
@param row0 Top row of the ROI (>= 0).
@param col0 Left column of the ROI (>= 0).
@param rows ROI row count (> 0, row0 + rows <= src->rows).
@param cols ROI column count (> 0, col0 + cols <= src->cols).
@param out Output view.
@returns ErrorCode.
*/
ErrorCode image_roi(const ImageView* src, int32_t row0, int32_t col0,
                    int32_t rows, int32_t cols, ImageView* out);

/*
Copies src into dst, converting the sample format when they differ.
dst must be pre-created with the same rows, cols, and channels.
u8 <-> float conversions keep numeric values (no 1/255 scaling);
float-to-u8 clamps to [0, 255] and rounds to nearest.

@param src Source view.
@param dst Destination image (same shape, any supported format).
@returns ErrorCode.
*/
ErrorCode image_convert(const ImageView* src, Image* dst);

/*
Reads one sample as float64 regardless of the stored format.

@param view Source view.
@param row Row index in [0, rows).
@param col Column index in [0, cols).
@param channel Channel index in [0, channels).
@param value_out Output sample value.
@returns ErrorCode.
*/
ErrorCode image_at(const ImageView* view, int32_t row, int32_t col,
                   int32_t channel, float64_t* value_out);

/*
Writes one sample from float64 into an owning image (clamped and
rounded for u8 storage).

@param img Destination image.
@param row Row index in [0, rows).
@param col Column index in [0, cols).
@param channel Channel index in [0, channels).
@param value Sample value.
@returns ErrorCode.
*/
ErrorCode image_set(Image* img, int32_t row, int32_t col, int32_t channel,
                    float64_t value);

/*
Copies a single-channel image view into a pre-created k64FC1 matrix,
reading samples of any supported depth as float64. This bridges an
image into the numeric domain (e.g. a disparity map to reproject).

@param src Single-channel source view.
@param dst Pre-created k64FC1 matrix, same rows and cols as src.
@returns ErrorCode.
*/
ErrorCode image_to_matrix(const ImageView* src, Matrix* dst);

/*
Copies a k64FC1 matrix into a pre-created single-channel image,
converting float64 values to the destination depth (u8 clamps and
rounds). This bridges numeric results back into the image domain.

@param src k64FC1 matrix.
@param dst Pre-created single-channel image, same rows and cols as src.
@returns ErrorCode.
*/
ErrorCode image_from_matrix(const Matrix* src, Image* dst);

}  // namespace image
}  // namespace cvlib

#endif  // CVLIB_IMAGE_IMAGE_H_
