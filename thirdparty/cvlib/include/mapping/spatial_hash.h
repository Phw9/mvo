// Uniform-grid spatial hash for fixed-cell neighbour queries over a
// point set. Fast and simple for roughly uniform density; complements
// the k-d tree / octree for non-uniform clouds.

#ifndef CVLIB_MAPPING_SPATIAL_HASH_H_
#define CVLIB_MAPPING_SPATIAL_HASH_H_

#include "../types.h"
#include "../error_codes.h"

#include <cstdint>
#include <vector>

namespace cvlib {
namespace mapping {

struct SpatialHashImpl;   // defined in the implementation

// Owning uniform-grid spatial hash (handle to a private hash table).
// Build with spatial_hash_build and release with spatial_hash_destroy.
struct SpatialHash {
    SpatialHashImpl* impl;
};

/*
Builds a spatial hash bucketing points into cells of edge cell_size:
cell(p) = floor(p / cell_size). Queries use Euclidean (L2) distance.

@param points Point set, N-by-D with D == 2 or 3 (N >= 1), k64FC1 finite.
@param cell_size Cubic cell edge length (> 0).
@param out Output hash; overwritten on success (release with
       spatial_hash_destroy).
@returns ErrorCode.
*/
ErrorCode spatial_hash_build(const Matrix* points, float64_t cell_size,
                             SpatialHash* out);

// Releases the hash storage and resets the handle; safe on an empty hash.
void spatial_hash_destroy(SpatialHash* hash);

/*
Collects every point within radius of a query (inclusive, Euclidean), in
ascending point-index order. Only the cells overlapping the query's
search box are examined.

@param hash Built hash.
@param query Query point, length D (finite).
@param radius Search radius (>= 0).
@param indices Output point indices; cleared then filled.
@returns ErrorCode.
*/
ErrorCode spatial_hash_radius(const SpatialHash* hash, const Vector* query,
                              float64_t radius,
                              std::vector<int32_t>* indices);

/*
Finds the k nearest points to a query (Euclidean) by expanding cell
rings outward, ordered by ascending distance with ties broken toward the
lower point index.

@param hash Built hash.
@param query Query point, length D (finite).
@param k Neighbour count (1 <= k <= point count).
@param indices Output point indices, length k; pre-allocated.
@param distances Output Euclidean distances, length k; pre-allocated.
@returns ErrorCode.
*/
ErrorCode spatial_hash_knn(const SpatialHash* hash, const Vector* query,
                           int32_t k, int32_t* indices,
                           float64_t* distances);

}  // namespace mapping
}  // namespace cvlib

#endif  // CVLIB_MAPPING_SPATIAL_HASH_H_
