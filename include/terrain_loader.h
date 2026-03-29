#pragma once

#include "types.h"
#include <string>

namespace mars {

class TerrainLoader {
public:
    static TerrainGrid loadGeoTIFF(const std::string& filepath, int crop_x_offset = 0,
                                   int crop_y_offset = 0, int crop_width = 512,
                                   int crop_height = 512);

    static TerrainGrid loadBinary(const std::string& filepath, int width, int height,
                                  double meters_per_pixel);

    static void saveBinary(const TerrainGrid& grid, const std::string& filepath);

private:
    static void validateGDALDataset(void* dataset);
};

} // namespace mars
