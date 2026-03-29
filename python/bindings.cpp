#include "astar_planner.h"
#include "costmap_builder.h"
#include "slope_analyzer.h"
#include "terrain_loader.h"
#include "types.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace mars;

namespace {

py::array_t<float> grid_to_2d(const TerrainGrid& g, float (TerrainCell::*member)) {
    py::array_t<float> result({g.height, g.width});
    auto buf = result.mutable_unchecked<2>();
    for (int r = 0; r < g.height; ++r)
        for (int c = 0; c < g.width; ++c)
            buf(r, c) = g.at(r, c).*member;
    return result;
}

} // namespace

PYBIND11_MODULE(libmars, m) {
    m.doc() = "Mars terrain engine for rover path planning";

    py::class_<GridPoint>(m, "GridPoint")
        .def(py::init<int, int>(), py::arg("row"), py::arg("col"))
        .def_readwrite("row", &GridPoint::row)
        .def_readwrite("col", &GridPoint::col);

    py::class_<Path>(m, "Path")
        .def_readwrite("waypoints", &Path::waypoints)
        .def_readwrite("total_cost", &Path::total_cost)
        .def_readwrite("total_distance_m", &Path::total_distance_m)
        .def_readwrite("energy_score", &Path::energy_score);

    py::class_<TerrainGrid>(m, "TerrainGrid")
        .def_readwrite("width", &TerrainGrid::width)
        .def_readwrite("height", &TerrainGrid::height)
        .def_readwrite("meters_per_pixel", &TerrainGrid::meters_per_pixel)
        .def("get_costmap_array",
             [](const TerrainGrid& g) { return grid_to_2d(g, &TerrainCell::cost); })
        .def("get_elevation_array",
             [](const TerrainGrid& g) { return grid_to_2d(g, &TerrainCell::elevation_m); })
        .def("get_slope_array",
             [](const TerrainGrid& g) { return grid_to_2d(g, &TerrainCell::slope_deg); })
        .def("get_cell_cost",
             [](const TerrainGrid& g, int r, int c) { return g.at(r, c).cost; })
        .def("get_cell_elevation",
             [](const TerrainGrid& g, int r, int c) { return g.at(r, c).elevation_m; });

    m.def("load_binary", &TerrainLoader::loadBinary, py::arg("filepath"), py::arg("width"),
          py::arg("height"), py::arg("mpp"));

    m.def(
        "build_full_terrain",
        [](const std::string& fp, int w, int h, double mpp) {
            auto grid = TerrainLoader::loadBinary(fp, w, h, mpp);
            SlopeAnalyzer::analyze(grid);
            CostmapBuilder::buildCostmap(grid);
            CostmapBuilder::inflateObstacles(grid);
            return grid;
        },
        py::arg("filepath"), py::arg("width"), py::arg("height"), py::arg("mpp"));

    m.def("astar_plan", &AStarPlanner::plan);
    m.def("straight_line", &AStarPlanner::straightLine);
}
