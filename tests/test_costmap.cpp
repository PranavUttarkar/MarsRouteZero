#include "costmap_builder.h"
#include "slope_analyzer.h"
#include "terrain_loader.h"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <filesystem>

using namespace mars;

static std::string write_flat_grid(int w, int h, float z) {
    TerrainGrid g;
    g.width = w;
    g.height = h;
    g.meters_per_pixel = 1.0;
    g.cells.resize(static_cast<size_t>(w * h));
    for (auto& c : g.cells) {
        c.elevation_m = z;
        c.slope_deg = 0;
        c.roughness = 0;
        c.terrain_class = 0;
        c.cost = 0;
    }
    auto path = (std::filesystem::temp_directory_path() / "mars_flat.bin").string();
    TerrainLoader::saveBinary(g, path);
    return path;
}

TEST_CASE("Flat terrain yields low slope and moderate cost") {
    auto path = write_flat_grid(32, 32, 100.0f);
    TerrainGrid g = TerrainLoader::loadBinary(path, 32, 32, 1.0);
    SlopeAnalyzer::analyze(g);
    CostmapBuilder::buildCostmap(g);
    REQUIRE(g.at(16, 16).slope_deg == Catch::Approx(0.0f).margin(0.05f));
    REQUIRE(g.at(16, 16).cost < 0.2f);
    std::remove(path.c_str());
}
