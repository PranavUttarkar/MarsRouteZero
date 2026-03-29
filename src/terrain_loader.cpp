#include "terrain_loader.h"

#include <fstream>
#include <stdexcept>

#ifdef MARS_USE_GDAL
#include <cpl_conv.h>
#include <gdal_priv.h>
#endif

namespace mars {

void TerrainLoader::validateGDALDataset(void* dataset) {
    if (!dataset)
        throw std::runtime_error("GDAL dataset is null");
}

TerrainGrid TerrainLoader::loadGeoTIFF(const std::string& filepath, int cx, int cy, int cw,
                                       int ch) {
#ifndef MARS_USE_GDAL
    (void)filepath;
    (void)cx;
    (void)cy;
    (void)cw;
    (void)ch;
    throw std::runtime_error("MarsRouteZero built without GDAL; use loadBinary or rebuild with "
                             "MARS_USE_GDAL=ON");
#else
    GDALAllRegister();
    GDALDataset* ds = static_cast<GDALDataset*>(GDALOpen(filepath.c_str(), GA_ReadOnly));
    if (!ds)
        throw std::runtime_error("Cannot open GeoTIFF: " + filepath);
    validateGDALDataset(ds);

    GDALRasterBand* band = ds->GetRasterBand(1);
    double gt[6];
    ds->GetGeoTransform(gt);

    TerrainGrid grid;
    grid.width = cw;
    grid.height = ch;
    grid.meters_per_pixel = std::abs(gt[1]);
    grid.origin_lon = gt[0] + cx * gt[1];
    grid.origin_lat = gt[3] + cy * gt[5];
    grid.cells.resize(static_cast<size_t>(cw * ch));

    std::vector<float> buf(static_cast<size_t>(cw * ch));
    CPLErr err = band->RasterIO(GF_Read, cx, cy, cw, ch, buf.data(), cw, ch, GDT_Float32, 0, 0);
    if (err != CE_None) {
        GDALClose(ds);
        throw std::runtime_error("RasterIO failed");
    }

    for (int i = 0; i < cw * ch; ++i) {
        grid.cells[static_cast<size_t>(i)].elevation_m = buf[static_cast<size_t>(i)];
        grid.cells[static_cast<size_t>(i)].slope_deg = 0.0f;
        grid.cells[static_cast<size_t>(i)].roughness = 0.0f;
        grid.cells[static_cast<size_t>(i)].terrain_class = 0;
        grid.cells[static_cast<size_t>(i)].cost = 0.0f;
    }

    GDALClose(ds);
    return grid;
#endif
}

TerrainGrid TerrainLoader::loadBinary(const std::string& fp, int w, int h, double mpp) {
    std::ifstream file(fp, std::ios::binary);
    if (!file)
        throw std::runtime_error("Cannot open binary: " + fp);

    const auto expected = static_cast<std::streamsize>(sizeof(float) * static_cast<size_t>(w * h));
    TerrainGrid grid;
    grid.width = w;
    grid.height = h;
    grid.meters_per_pixel = mpp;
    grid.origin_lat = 0.0;
    grid.origin_lon = 0.0;
    grid.cells.resize(static_cast<size_t>(w * h));

    std::vector<float> buf(static_cast<size_t>(w * h));
    file.read(reinterpret_cast<char*>(buf.data()), expected);
    if (!file || file.gcount() != expected)
        throw std::runtime_error("Binary size mismatch for " + fp + " (expected " +
                                 std::to_string(w) + "x" + std::to_string(h) + " float32)");

    for (int i = 0; i < w * h; ++i) {
        auto& c = grid.cells[static_cast<size_t>(i)];
        c.elevation_m = buf[static_cast<size_t>(i)];
        c.slope_deg = 0.0f;
        c.roughness = 0.0f;
        c.terrain_class = 0;
        c.cost = 0.0f;
    }
    return grid;
}

void TerrainLoader::saveBinary(const TerrainGrid& grid, const std::string& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file)
        throw std::runtime_error("Cannot write binary: " + filepath);
    for (const auto& cell : grid.cells) {
        float z = cell.elevation_m;
        file.write(reinterpret_cast<const char*>(&z), sizeof(float));
    }
}

} // namespace mars
