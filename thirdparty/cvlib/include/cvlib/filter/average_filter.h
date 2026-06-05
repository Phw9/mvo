// Moving average filters.

#ifndef CVLIB_FILTER_AVERAGE_FILTER_H_
#define CVLIB_FILTER_AVERAGE_FILTER_H_

#include "cvlib/types.h"
#include "cvlib/error_codes.h"

#include <cstdint>

namespace cvlib {
namespace filter {

/*
Applies a centered moving average (box filter) to a matrix-valued signal.

@param signal Input samples in matrix form (1D or 2D layout per implementation).
@param window_size Window length in samples; odd length is preferred.
@param result Output filtered signal; must be pre-allocated.
@returns ErrorCode.

*/

ErrorCode average_filter(const Matrix* signal, int32_t window_size,
                         Matrix* result);

/*
Applies a weighted moving average; uniform weights if weights is null.

@param signal Input matrix.
@param window_size Window length; must match weights->size when weights is non-null.
@param result Output matrix; must be pre-allocated.
@param weights Optional nonnegative weights; null selects uniform weights.
@returns ErrorCode.

*/

ErrorCode moving_average_filter(const Matrix* signal, int32_t window_size,
                                Matrix* result,
                                const Vector* weights = nullptr);

}  // namespace filter
}  // namespace cvlib

#endif  // CVLIB_FILTER_AVERAGE_FILTER_H_
