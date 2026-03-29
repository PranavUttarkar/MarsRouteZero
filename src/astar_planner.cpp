#include "astar_planner.h"
#include "costmap_builder.h"

#include <algorithm>
#include <cmath>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace mars {

namespace {

inline int idx(int r, int c, int w) { return r * w + c; }

inline float edge_weight(const TerrainGrid& g, GridPoint a, GridPoint b) {
    double mpp = g.meters_per_pixel;
    int dr = b.row - a.row, dc = b.col - a.col;
    float step_m = static_cast<float>(mpp * std::sqrt(static_cast<double>(dr * dr + dc * dc)));
    float ca = g.at(a.row, a.col).cost;
    float cb = g.at(b.row, b.col).cost;
    // Positive weight even on flat terrain so A* prefers shorter geometric paths when costs tie.
    return step_m * (0.01f + 0.5f * (ca + cb));
}

float min_traversable_edge_factor(const TerrainGrid& g) {
    float m = 1.0f;
    for (const auto& cell : g.cells) {
        if (cell.cost < CostmapBuilder::IMPASSABLE_COST - 1e-3f)
            m = std::min(m, cell.cost);
    }
    return std::max(0.01f, 0.01f + m);
}

void path_metrics(const TerrainGrid& g, const std::vector<GridPoint>& wp, Path& out) {
    if (wp.size() < 2) {
        out.total_distance_m = 0.0f;
        out.total_cost = 0.0f;
        out.energy_score = 0.0f;
        return;
    }
    float dist_m = 0.0f;
    float gsum = 0.0f;
    float energy = 0.0f;
    for (size_t i = 0; i + 1 < wp.size(); ++i) {
        const auto& a = wp[i];
        const auto& b = wp[i + 1];
        int dr = b.row - a.row, dc = b.col - a.col;
        float step_m =
            static_cast<float>(g.meters_per_pixel * std::sqrt(static_cast<double>(dr * dr + dc * dc)));
        dist_m += step_m;
        gsum += edge_weight(g, a, b);
        float mid_cost = 0.5f * (g.at(a.row, a.col).cost + g.at(b.row, b.col).cost);
        energy += mid_cost * step_m;
    }
    out.total_distance_m = dist_m;
    out.total_cost = gsum;
    out.energy_score = energy;
}

} // namespace

float AStarPlanner::heuristic(GridPoint a, GridPoint b, double meters_per_pixel,
                              float min_step_cost) {
    float dr = static_cast<float>(a.row - b.row);
    float dc = static_cast<float>(a.col - b.col);
    float dist_m = static_cast<float>(meters_per_pixel * std::sqrt(static_cast<double>(dr * dr + dc * dc)));
    return dist_m * min_step_cost;
}

Path AStarPlanner::plan(const TerrainGrid& grid, GridPoint start, GridPoint goal) {
    Path result;
    const int W = grid.width, H = grid.height;
    if (start.row < 0 || start.row >= H || start.col < 0 || start.col >= W || goal.row < 0 ||
        goal.row >= H || goal.col < 0 || goal.col >= W) {
        result.waypoints.push_back(start);
        return result;
    }
    if (grid.at(start.row, start.col).cost >= CostmapBuilder::IMPASSABLE_COST - 1e-3f ||
        grid.at(goal.row, goal.col).cost >= CostmapBuilder::IMPASSABLE_COST - 1e-3f) {
        result.waypoints.push_back(start);
        return result;
    }

    float h0 = min_traversable_edge_factor(grid);

    struct OpenNode {
        float f;
        int r, c;
        bool operator>(const OpenNode& o) const { return f > o.f; }
    };

    std::priority_queue<OpenNode, std::vector<OpenNode>, std::greater<OpenNode>> open;
    std::unordered_map<int, float> g_score;
    std::unordered_map<int, GridPoint> came_from;

    const int start_i = idx(start.row, start.col, W);
    g_score[start_i] = 0.0f;
    open.push({heuristic(start, goal, grid.meters_per_pixel, h0), start.row, start.col});

    static const int DR[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    static const int DC[8] = {-1, 0, 1, -1, 1, -1, 0, 1};

    int goal_i = -1;
    while (!open.empty()) {
        OpenNode cur = open.top();
        open.pop();
        int cur_i = idx(cur.r, cur.c, W);
        float cur_g = g_score[cur_i];

        if (cur.r == goal.row && cur.col == goal.col) {
            goal_i = cur_i;
            break;
        }

        GridPoint cur_pt{cur.r, cur.c};
        for (int k = 0; k < 8; ++k) {
            int nr = cur.r + DR[k], nc = cur.c + DC[k];
            if (nr < 0 || nr >= H || nc < 0 || nc >= W)
                continue;
            if (grid.at(nr, nc).cost >= CostmapBuilder::IMPASSABLE_COST - 1e-3f)
                continue;
            GridPoint nxt{nr, nc};
            float tentative = cur_g + edge_weight(grid, cur_pt, nxt);
            int ni = idx(nr, nc, W);
            auto it = g_score.find(ni);
            if (it == g_score.end() || tentative < it->second) {
                came_from[ni] = cur_pt;
                g_score[ni] = tentative;
                float hf = heuristic(nxt, goal, grid.meters_per_pixel, h0);
                open.push({tentative + hf, nr, nc});
            }
        }
    }

    if (goal_i < 0) {
        result.waypoints.push_back(start);
        return result;
    }

    std::vector<GridPoint> rev;
    for (GridPoint at = goal;;) {
        rev.push_back(at);
        int ai = idx(at.row, at.col, W);
        if (ai == start_i)
            break;
        auto it = came_from.find(ai);
        if (it == came_from.end()) {
            result.waypoints = {start};
            return result;
        }
        at = it->second;
    }
    std::reverse(rev.begin(), rev.end());
    result.waypoints = std::move(rev);
    path_metrics(grid, result.waypoints, result);
    return result;
}

Path AStarPlanner::straightLine(const TerrainGrid& grid, GridPoint start, GridPoint goal) {
    Path result;
    int r0 = start.row, c0 = start.col, r1 = goal.row, c1 = goal.col;
    int dr = r1 - r0, dc = c1 - c0;
    int steps = std::max(std::abs(dr), std::abs(dc));
    if (steps == 0) {
        result.waypoints.push_back(start);
        path_metrics(grid, result.waypoints, result);
        return result;
    }
    std::unordered_set<int> seen;
    for (int s = 0; s <= steps; ++s) {
        int r = r0 + (dr * s) / steps;
        int c = c0 + (dc * s) / steps;
        int k = idx(r, c, grid.width);
        if (seen.insert(k).second)
            result.waypoints.push_back(GridPoint{r, c});
    }
    path_metrics(grid, result.waypoints, result);
    return result;
}

} // namespace mars
