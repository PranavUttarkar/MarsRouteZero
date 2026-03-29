#include "astar_planner.h"
#include "costmap_builder.h"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace mars;

static TerrainGrid make_uniform(int w, int h, float cost) {
    TerrainGrid g;
    g.width = w;
    g.height = h;
    g.meters_per_pixel = 1.0;
    g.cells.resize(static_cast<size_t>(w * h));
    for (auto& c : g.cells) {
        c.elevation_m = 0;
        c.slope_deg = 0;
        c.roughness = 0;
        c.terrain_class = 0;
        c.cost = cost;
    }
    return g;
}

TEST_CASE("A* reaches goal on open grid") {
    TerrainGrid g = make_uniform(16, 16, 0.1f);
    GridPoint start{0, 0};
    GridPoint goal{15, 15};
    Path p = AStarPlanner::plan(g, start, goal);
    REQUIRE_FALSE(p.waypoints.empty());
    REQUIRE(p.waypoints.front().row == 0);
    REQUIRE(p.waypoints.front().col == 0);
    REQUIRE(p.waypoints.back().row == 15);
    REQUIRE(p.waypoints.back().col == 15);
    REQUIRE(p.total_distance_m > 0.0f);
}

TEST_CASE("A* cannot cross impassable cell") {
    TerrainGrid g = make_uniform(5, 5, 0.1f);
    for (int c = 0; c < 5; ++c)
        g.at(2, c).cost = CostmapBuilder::IMPASSABLE_COST;
    Path p = AStarPlanner::plan(g, GridPoint{0, 2}, GridPoint{4, 2});
    REQUIRE(p.waypoints.size() == 1u);
}

TEST_CASE("Straight line samples grid cells") {
    TerrainGrid g = make_uniform(10, 10, 0.2f);
    Path p = AStarPlanner::straightLine(g, GridPoint{0, 0}, GridPoint{0, 9});
    REQUIRE(p.waypoints.size() >= 2u);
    REQUIRE(p.total_distance_m == Catch::Approx(9.0f));
}
