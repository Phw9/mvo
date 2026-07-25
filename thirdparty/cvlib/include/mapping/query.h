// Shared query conventions for spatial map-representation structures.

#ifndef CVLIB_MAPPING_QUERY_H_
#define CVLIB_MAPPING_QUERY_H_

#include <cstdint>

namespace cvlib {
namespace mapping {

// Distance metrics for nearest-neighbour and radius queries.
static constexpr int32_t kMetricL2 = 0;   // Euclidean
static constexpr int32_t kMetricL1 = 1;   // Manhattan

}  // namespace mapping
}  // namespace cvlib

#endif  // CVLIB_MAPPING_QUERY_H_
