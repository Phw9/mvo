// Voxel-grid downsampling for point-cloud map representations.

#ifndef CVLIB_MAPPING_VOXEL_GRID_H_
#define CVLIB_MAPPING_VOXEL_GRID_H_

#include "../types.h"
#include "../error_codes.h"

#include <cstdint>

namespace cvlib {
namespace mapping {

/*
Downsamples a 3D point cloud to one centroid per occupied cubic voxel of
the given leaf size. Voxel of a point p is floor(p / leaf_size); the
output holds the mean of all points falling in each occupied voxel, in a
deterministic voxel order (sorted by voxel coordinate). At most N voxels
are produced, so the output is pre-allocated to the input size.

@param points Input cloud, N-by-3 (N >= 1), k64FC1 finite values.
@param leaf_size Cubic voxel edge length (> 0).
@param out Output centroids, pre-allocated rows >= N, cols == 3; the
       first out_count rows are written.
@param out_count Output number of occupied voxels (<= N).
@returns ErrorCode.
*/
ErrorCode voxel_grid_downsample(const Matrix* points, float64_t leaf_size,
                                Matrix* out, int32_t* out_count);

}  // namespace mapping
}  // namespace cvlib

#endif  // CVLIB_MAPPING_VOXEL_GRID_H_
