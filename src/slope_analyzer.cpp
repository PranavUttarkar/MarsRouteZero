#include "slope_analyzer.h"

#include <algorithm>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mars {

void SlopeAnalyzer::analyze(TerrainGrid& g) {
    const int W = g.width, H = g.height;
    const float mpp = static_cast<float>(g.meters_per_pixel);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 16)
#endif
    for (int r = 1; r < H - 1; ++r) {
        for (int c = 1; c < W - 1; ++c) {
            auto e = [&](int dr, int dc) -> float { return g.at(r + dr, c + dc).elevation_m; };

            float gx = (-e(-1, -1) + e(-1, 1) - 2 * e(0, -1) + 2 * e(0, 1) - e(1, -1) + e(1, 1)) /
                       (8.0f * mpp);
            float gy = (-e(-1, -1) - 2 * e(-1, 0) - e(-1, 1) + e(1, -1) + 2 * e(1, 0) + e(1, 1)) /
                       (8.0f * mpp);

            constexpr float kPi = 3.14159265f;
            float slope_rad = std::atan(std::sqrt(gx * gx + gy * gy));
            g.at(r, c).slope_deg = slope_rad * 180.0f / kPi;

            float sum = 0, sum2 = 0;
            for (int dr = -1; dr <= 1; ++dr)
                for (int dc = -1; dc <= 1; ++dc) {
                    float v = e(dr, dc);
                    sum += v;
                    sum2 += v * v;
                }
            float mean = sum / 9.0f;
            float var = sum2 / 9.0f - mean * mean;
            g.at(r, c).roughness =
                std::min(1.0f, std::sqrt(std::max(0.0f, var)) / 2.0f);
        }
    }

    for (int r = 0; r < H; ++r) {
        g.at(r, 0) = g.at(r, 1);
        g.at(r, W - 1) = g.at(r, W - 2);
    }
    for (int c = 0; c < W; ++c) {
        g.at(0, c) = g.at(1, c);
        g.at(H - 1, c) = g.at(H - 2, c);
    }
}

} // namespace mars
