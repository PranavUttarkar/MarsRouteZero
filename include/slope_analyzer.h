#pragma once

#include "types.h"

namespace mars {

class SlopeAnalyzer {
public:
    static void analyze(TerrainGrid& grid);

    static constexpr float SLOPE_LIMIT_DEG = 20.0f;
    static constexpr float ROUGHNESS_LIMIT = 0.6f;
    static constexpr float PERSEVERANCE_MAX_SLOPE = 25.0f;
};

} // namespace mars
