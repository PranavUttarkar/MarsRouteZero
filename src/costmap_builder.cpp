#include "costmap_builder.h"
#include "slope_analyzer.h"

#include <algorithm>
#include <vector>

namespace mars {

void CostmapBuilder::buildCostmap(TerrainGrid& g, const CostWeights& w) {
    static constexpr float TERRAIN_COST[4] = {0.0f, 0.1f, 0.2f, 1.0f};

    for (auto& cell : g.cells) {
        float slope_norm = std::min(1.0f, cell.slope_deg / SlopeAnalyzer::SLOPE_LIMIT_DEG);
        float slope_cost = slope_norm * slope_norm;
        float rough_cost = cell.roughness;
        int tc = std::clamp(cell.terrain_class, 0, 3);
        float terrain_cost = TERRAIN_COST[tc];

        if (cell.slope_deg > SlopeAnalyzer::PERSEVERANCE_MAX_SLOPE) {
            cell.cost = IMPASSABLE_COST;
        } else {
            cell.cost = std::min(1.0f, w.slope_weight * slope_cost + w.roughness_weight * rough_cost +
                                            w.terrain_weight * terrain_cost);
        }
    }
}

void CostmapBuilder::inflateObstacles(TerrainGrid& g, int radius) {
    std::vector<float> inflated(g.cells.size());
    for (size_t i = 0; i < g.cells.size(); ++i)
        inflated[i] = g.cells[i].cost;

    for (int r = 0; r < g.height; ++r) {
        for (int c = 0; c < g.width; ++c) {
            if (g.at(r, c).cost >= OBSTACLE_THRESHOLD) {
                for (int dr = -radius; dr <= radius; ++dr)
                    for (int dc = -radius; dc <= radius; ++dc) {
                        int nr = r + dr, nc = c + dc;
                        if (nr >= 0 && nr < g.height && nc >= 0 && nc < g.width)
                            inflated[static_cast<size_t>(nr * g.width + nc)] =
                                std::max(inflated[static_cast<size_t>(nr * g.width + nc)], 0.7f);
                    }
            }
        }
    }
    for (size_t i = 0; i < g.cells.size(); ++i)
        g.cells[i].cost = inflated[i];
}

} // namespace mars
