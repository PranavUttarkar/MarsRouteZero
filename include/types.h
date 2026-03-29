#pragma once

#include <vector>

namespace mars {

struct GridPoint {
    int row, col;
    bool operator==(const GridPoint& o) const {
        return row == o.row && col == o.col;
    }
};

struct TerrainCell {
    float elevation_m;
    float slope_deg;
    float roughness;
    int terrain_class;
    float cost;
};

struct TerrainGrid {
    std::vector<TerrainCell> cells;
    int width, height;
    double origin_lat, origin_lon;
    double meters_per_pixel;

    TerrainCell& at(int row, int col) { return cells[row * width + col]; }
    const TerrainCell& at(int row, int col) const { return cells[row * width + col]; }
};

struct Path {
    std::vector<GridPoint> waypoints;
    float total_cost;
    float total_distance_m;
    float energy_score;
};

} // namespace mars
