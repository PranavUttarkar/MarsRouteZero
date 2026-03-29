#pragma once

#include "types.h"

namespace mars {

class AStarPlanner {
public:
    static Path plan(const TerrainGrid& grid, GridPoint start, GridPoint goal);
    static Path straightLine(const TerrainGrid& grid, GridPoint start, GridPoint goal);

private:
    static float heuristic(GridPoint a, GridPoint b, double meters_per_pixel, float min_step_cost);
};

} // namespace mars
