#include "terrain_loader.h"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <filesystem>

using namespace mars;

TEST_CASE("TerrainLoader save and load binary roundtrip") {
    TerrainGrid g;
    g.width = 3;
    g.height = 2;
    g.meters_per_pixel = 1.0;
    g.origin_lat = 0;
    g.origin_lon = 0;
    g.cells.resize(6);
    float z = 0.0f;
    for (auto& c : g.cells) {
        c.elevation_m = z;
        c.slope_deg = 0;
        c.roughness = 0;
        c.terrain_class = 0;
        c.cost = 0;
        z += 1.0f;
    }

    const auto tmp = (std::filesystem::temp_directory_path() / "mars_test_elev.bin").string();
    TerrainLoader::saveBinary(g, tmp);
    TerrainGrid g2 = TerrainLoader::loadBinary(tmp, 3, 2, 1.0);
    REQUIRE(g2.width == 3);
    REQUIRE(g2.height == 2);
    REQUIRE(g2.at(0, 0).elevation_m == Catch::Approx(0.0f));
    REQUIRE(g2.at(1, 1).elevation_m == Catch::Approx(4.0f));
    std::remove(tmp.c_str());
}

#ifndef MARS_USE_GDAL
TEST_CASE("loadGeoTIFF throws when GDAL disabled") {
    REQUIRE_THROWS_AS(TerrainLoader::loadGeoTIFF("dummy.tif"), std::runtime_error);
}
#endif
