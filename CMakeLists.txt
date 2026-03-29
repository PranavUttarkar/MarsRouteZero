cmake_minimum_required(VERSION 3.18)
project(MarsRouteZero VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(MARS_USE_GDAL "Enable GDAL GeoTIFF loading in TerrainLoader" OFF)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1)
FetchContent_MakeAvailable(pybind11)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.4.0)
FetchContent_MakeAvailable(Catch2)

add_library(mars_terrain STATIC
    src/terrain_loader.cpp
    src/slope_analyzer.cpp
    src/costmap_builder.cpp
    src/astar_planner.cpp)
target_include_directories(mars_terrain PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(MARS_USE_GDAL)
    find_package(GDAL REQUIRED)
    target_compile_definitions(mars_terrain PUBLIC MARS_USE_GDAL)
    target_link_libraries(mars_terrain PUBLIC GDAL::GDAL)
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(mars_terrain PUBLIC OpenMP::OpenMP_CXX)
endif()

pybind11_add_module(mars_py MODULE python/bindings.cpp)
target_link_libraries(mars_py PRIVATE mars_terrain)
set_target_properties(mars_py PROPERTIES OUTPUT_NAME libmars)

enable_testing()
add_executable(test_terrain tests/test_terrain.cpp)
target_link_libraries(test_terrain PRIVATE mars_terrain Catch2::Catch2WithMain)
add_test(NAME test_terrain COMMAND test_terrain)

add_executable(test_costmap tests/test_costmap.cpp)
target_link_libraries(test_costmap PRIVATE mars_terrain Catch2::Catch2WithMain)
add_test(NAME test_costmap COMMAND test_costmap)

add_executable(test_astar tests/test_astar.cpp)
target_link_libraries(test_astar PRIVATE mars_terrain Catch2::Catch2WithMain)
add_test(NAME test_astar COMMAND test_astar)
