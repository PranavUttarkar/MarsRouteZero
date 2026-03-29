#pragma once

#include "types.h"

namespace mars {

struct CostWeights {
    float slope_weight = 0.6f;
    float roughness_weight = 0.3f;
    float terrain_weight = 0.1f;
};

class CostmapBuilder {
public:
    static void buildCostmap(TerrainGrid& grid, const CostWeights& w = {});
    static void inflateObstacles(TerrainGrid& grid, int inflate_radius = 1);

    static constexpr float OBSTACLE_THRESHOLD = 0.85f;
    static constexpr float IMPASSABLE_COST = 1.0f;
};

} // namespace mars
