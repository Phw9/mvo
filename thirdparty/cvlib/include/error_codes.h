// API result codes.

#ifndef CVLIB_ERROR_CODES_H_
#define CVLIB_ERROR_CODES_H_

#include <cstdint>

namespace cvlib {

enum class ErrorCode : int32_t {
    kSuccess              = 0,
    kNullPointer          = 1,
    kInvalidDimension     = 2,
    kSingularMatrix       = 3,
    kInvalidArgument      = 4,
    kOutOfBounds          = 5,
    kNotConverged         = 6,
    kNotPositiveDefinite  = 7,
    kInvalidShape         = 8,
    kEmptyInput           = 9,
    kUnknownMethod        = 10,
    kUnreachableTarget    = 11
};

}  // namespace cvlib

#endif  // CVLIB_ERROR_CODES_H_
