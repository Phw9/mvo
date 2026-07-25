// Static balanced k-d tree for nearest-neighbour and radius queries over
// a point set (map points or feature coordinates).

#ifndef CVLIB_MAPPING_KDTREE_H_
#define CVLIB_MAPPING_KDTREE_H_

#include "../types.h"
#include "../error_codes.h"
#include "../mapping/query.h"

#include <cstdint>
#include <vector>

namespace cvlib {
namespace mapping {

// One tree node: the point it holds, the split axis, and child links.
struct KdTreeNode {
    int32_t point;   // row index of the point in the original matrix
    int32_t axis;    // split dimension
    int32_t left;    // child node index, or -1
    int32_t right;   // child node index, or -1
};

// Owning k-d tree. Build with kdtree_build and release with
// kdtree_destroy; the tree copies the input points, so it stays valid
// after the source matrix is destroyed.
struct KdTree {
    float64_t*  points;   // n-by-dim copy, row-major (owned)
    KdTreeNode* nodes;    // n nodes (owned)
    int32_t     n;
    int32_t     dim;
    int32_t     root;     // root node index, or -1 when empty
    int32_t     metric;
};

/*
Builds a balanced k-d tree from a point set.

@param points Point set, N-by-D (N >= 1, D >= 1), k64FC1 finite values.
@param metric kMetricL2 or kMetricL1.
@param out Output tree; overwritten on success (release with
       kdtree_destroy).
@returns ErrorCode.
*/
ErrorCode kdtree_build(const Matrix* points, int32_t metric, KdTree* out);

// Releases tree storage and resets the struct; safe on an empty tree.
void kdtree_destroy(KdTree* tree);

/*
Finds the k nearest points to a query, ordered by ascending distance
with ties broken toward the lower point index.

@param tree Built tree.
@param query Query point, length dim (finite).
@param k Neighbour count (1 <= k <= tree size).
@param indices Output point indices, length k; pre-allocated.
@param distances Output distances (metric units), length k; pre-allocated.
@returns ErrorCode.
*/
ErrorCode kdtree_knn(const KdTree* tree, const Vector* query, int32_t k,
                     int32_t* indices, float64_t* distances);

/*
Collects every point within radius of a query (inclusive), in ascending
point-index order.

@param tree Built tree.
@param query Query point, length dim (finite).
@param radius Search radius (>= 0, metric units).
@param indices Output point indices; cleared then filled.
@returns ErrorCode.
*/
ErrorCode kdtree_radius(const KdTree* tree, const Vector* query,
                        float64_t radius, std::vector<int32_t>* indices);

}  // namespace mapping
}  // namespace cvlib

#endif  // CVLIB_MAPPING_KDTREE_H_
