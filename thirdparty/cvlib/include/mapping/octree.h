// Point-region octree for 3D map representations: nearest-neighbour and
// radius queries over a point cloud with Euclidean (L2) distance.

#ifndef CVLIB_MAPPING_OCTREE_H_
#define CVLIB_MAPPING_OCTREE_H_

#include "../types.h"
#include "../error_codes.h"

#include <cstdint>
#include <vector>

namespace cvlib {
namespace mapping {

// One octree node: a cube (center, half-edge), eight child links, and,
// for leaves, a contiguous range into the point-order array.
struct OctreeNode {
    float64_t cx;
    float64_t cy;
    float64_t cz;
    float64_t half;
    int32_t children[8];   // child node index, or -1
    int32_t first;         // leaf: start index into order; else 0
    int32_t count;         // leaf: point count; internal node: 0
    int32_t leaf;          // 1 for a leaf, 0 for an internal node
};

// Owning point-region octree. Build with octree_build and release with
// octree_destroy; the tree copies the input points.
struct Octree {
    float64_t*  points;      // n-by-3 copy, row-major (owned)
    OctreeNode* nodes;       // node array (owned)
    int32_t*    order;       // permutation of point indices (owned)
    int32_t     n;
    int32_t     node_count;
    int32_t     root;        // root node index, or -1 when empty
};

/*
Builds a point-region octree over the axis-aligned bounding cube of the
points, subdividing a node into eight octants until it holds at most
max_points_per_leaf points or reaches max_depth.

@param points Point cloud, N-by-3 (N >= 1), k64FC1 finite values.
@param max_points_per_leaf Leaf capacity before splitting (>= 1).
@param max_depth Maximum subdivision depth (>= 0).
@param out Output tree; overwritten on success (release with
       octree_destroy).
@returns ErrorCode.
*/
ErrorCode octree_build(const Matrix* points, int32_t max_points_per_leaf,
                       int32_t max_depth, Octree* out);

// Releases tree storage and resets the struct; safe on an empty tree.
void octree_destroy(Octree* tree);

/*
Finds the k nearest points to a query (Euclidean), ordered by ascending
distance with ties broken toward the lower point index.

@param tree Built tree.
@param query Query point, length 3 (finite).
@param k Neighbour count (1 <= k <= tree size).
@param indices Output point indices, length k; pre-allocated.
@param distances Output Euclidean distances, length k; pre-allocated.
@returns ErrorCode.
*/
ErrorCode octree_knn(const Octree* tree, const Vector* query, int32_t k,
                     int32_t* indices, float64_t* distances);

/*
Collects every point within radius of a query (inclusive, Euclidean), in
ascending point-index order.

@param tree Built tree.
@param query Query point, length 3 (finite).
@param radius Search radius (>= 0).
@param indices Output point indices; cleared then filled.
@returns ErrorCode.
*/
ErrorCode octree_radius(const Octree* tree, const Vector* query,
                        float64_t radius, std::vector<int32_t>* indices);

}  // namespace mapping
}  // namespace cvlib

#endif  // CVLIB_MAPPING_OCTREE_H_
