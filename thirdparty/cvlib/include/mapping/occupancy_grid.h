// Two-dimensional log-odds occupancy grid map. Cells accumulate free and
// occupied evidence from range observations via ray casting, the standard
// probabilistic map representation for planar navigation.

#ifndef CVLIB_MAPPING_OCCUPANCY_GRID_H_
#define CVLIB_MAPPING_OCCUPANCY_GRID_H_

#include "../types.h"
#include "../error_codes.h"

#include <cstdint>

namespace cvlib {
namespace mapping {

/*
Inverse-sensor-model parameters, expressed as occupancy probabilities.

@param prob_hit Posterior occupancy for a cell where a beam ends (> 0.5).
@param prob_miss Posterior occupancy for a cell a beam passes through
       (< 0.5).
@param prob_min Lower clamp on a cell's occupancy probability (in (0, 1)).
@param prob_max Upper clamp on a cell's occupancy probability (in (0, 1)).
*/
struct OccupancyUpdateModel {
    float64_t prob_hit = 0.7;
    float64_t prob_miss = 0.4;
    float64_t prob_min = 0.12;
    float64_t prob_max = 0.97;
};

/*
Owning 2D occupancy grid. Cells store log-odds occupancy (0 == unknown,
probability 0.5); column index is world x, row index is world y. Build
with occupancy_grid_create and release with occupancy_grid_destroy.

@param log_odds Row-major cell log-odds, height rows by width columns.
@param width Cell count along x.
@param height Cell count along y.
@param resolution World units per cell edge.
@param origin_x World x of the lower corner of cell column 0.
@param origin_y World y of the lower corner of cell row 0.
@param l_hit Log-odds added to a beam-end cell.
@param l_miss Log-odds added to a pass-through cell.
@param l_min Lower log-odds clamp.
@param l_max Upper log-odds clamp.
*/
struct OccupancyGrid {
    float64_t* log_odds;
    int32_t width;
    int32_t height;
    float64_t resolution;
    float64_t origin_x;
    float64_t origin_y;
    float64_t l_hit;
    float64_t l_miss;
    float64_t l_min;
    float64_t l_max;
};

/*
Allocates a grid with every cell unknown (log-odds 0).

@param width Cell count along x (>= 1).
@param height Cell count along y (>= 1).
@param resolution World units per cell (> 0).
@param origin_x World x of cell column 0.
@param origin_y World y of cell row 0.
@param model Sensor model; null uses defaults. Probabilities must satisfy
       0 < prob_min <= prob_miss < 0.5 < prob_hit <= prob_max < 1.
@param out Output grid; overwritten on success (release with
       occupancy_grid_destroy).
@returns ErrorCode.
*/
ErrorCode occupancy_grid_create(int32_t width, int32_t height,
                                float64_t resolution, float64_t origin_x,
                                float64_t origin_y,
                                const OccupancyUpdateModel* model,
                                OccupancyGrid* out);

// Releases the grid storage and resets the handle; safe on an empty grid.
void occupancy_grid_destroy(OccupancyGrid* grid);

/*
Converts a world point to its integer cell, whether or not it lies inside
the grid.

@param grid Built grid.
@param world_x World x.
@param world_y World y.
@param cell_x Output column index.
@param cell_y Output row index.
@returns ErrorCode.
*/
ErrorCode occupancy_grid_world_to_cell(const OccupancyGrid* grid,
                                       float64_t world_x, float64_t world_y,
                                       int32_t* cell_x, int32_t* cell_y);

/*
Integrates one range scan: for each endpoint, the cells the beam crosses
gain miss evidence and the endpoint cell gains hit evidence, all clamped.
Beam segments are clipped to the grid; an endpoint outside the grid only
deposits miss evidence on the crossed in-bounds cells.

@param grid Built grid.
@param sensor_x World x of the beam origin.
@param sensor_y World y of the beam origin.
@param endpoints Beam endpoints, N-by-2 world coordinates (N >= 1),
       k64FC1 finite.
@returns ErrorCode.
*/
ErrorCode occupancy_grid_integrate_scan(OccupancyGrid* grid,
                                        float64_t sensor_x,
                                        float64_t sensor_y,
                                        const Matrix* endpoints);

/*
Reads the occupancy probability at a world point.

@param grid Built grid.
@param world_x World x.
@param world_y World y.
@param probability Output occupancy in (0, 1); 0.5 for an unknown or
       out-of-bounds cell.
@returns ErrorCode.
*/
ErrorCode occupancy_grid_probability(const OccupancyGrid* grid,
                                     float64_t world_x, float64_t world_y,
                                     float64_t* probability);

/*
Writes the whole grid as occupancy probabilities.

@param grid Built grid.
@param out Output matrix, height rows by width columns (k64FC1); cell
       (row, col) receives the probability of grid cell (col, row).
@returns ErrorCode.
*/
ErrorCode occupancy_grid_to_matrix(const OccupancyGrid* grid, Matrix* out);

}  // namespace mapping
}  // namespace cvlib

#endif  // CVLIB_MAPPING_OCCUPANCY_GRID_H_
