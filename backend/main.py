"""FastAPI service for terrain metadata and path planning (requires built libmars)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

for _extra in (
    os.environ.get("MARS_LIBMARS_BUILD", ""),
    str(_ROOT / "build"),
    str(_ROOT / "build" / "Release"),
    str(_ROOT / "build" / "Debug"),
    str(_ROOT / "build" / "RelWithDebInfo"),
):
    if _extra and Path(_extra).is_dir() and _extra not in sys.path:
        sys.path.insert(0, _extra)

import numpy as np

import libmars

_TERRAIN_PATH = Path(
    os.environ.get("MARS_ELEVATION_BIN", _ROOT / "data/costmap/jezero_elevation.bin")
)
_GRID = int(os.environ.get("MARS_GRID_SIZE", "512"))
_MPP = float(os.environ.get("MARS_METERS_PER_PIXEL", "1.0"))

if not _TERRAIN_PATH.is_file():
    raise FileNotFoundError(
        f"Elevation binary not found: {_TERRAIN_PATH}. "
        "Run: python scripts/gen_synthetic_terrain.py"
    )

TERRAIN = libmars.build_full_terrain(str(_TERRAIN_PATH), _GRID, _GRID, _MPP)
ELEV = np.asarray(TERRAIN.get_elevation_array())
COST = np.asarray(TERRAIN.get_costmap_array())
SLOPE = np.asarray(TERRAIN.get_slope_array())

app = FastAPI(title="Mars Route Zero API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("MARS_CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/terrain")
async def get_terrain_metadata():
    return {
        "width": TERRAIN.width,
        "height": TERRAIN.height,
        "meters_per_pixel": TERRAIN.meters_per_pixel,
        "elevation_min": float(ELEV.min()),
        "elevation_max": float(ELEV.max()),
        "elevation_range_m": float(ELEV.max() - ELEV.min()),
        "mean_slope_deg": float(SLOPE.mean()),
        "traversable_fraction": float((COST < 0.85).mean()),
    }


class PlanRequest(BaseModel):
    start_col: int
    start_row: int
    goal_col: int
    goal_row: int
    planner: str = "astar"


class PlanResponse(BaseModel):
    waypoints: list[list[int]]
    total_cost: float
    total_distance_m: float
    energy_score: float
    planner: str


@app.post("/api/plan", response_model=PlanResponse)
async def plan_route(req: PlanRequest):
    start = libmars.GridPoint(req.start_row, req.start_col)
    goal = libmars.GridPoint(req.goal_row, req.goal_col)
    if req.planner == "astar":
        path = libmars.astar_plan(TERRAIN, start, goal)
    else:
        path = libmars.straight_line(TERRAIN, start, goal)
    return PlanResponse(
        waypoints=[[p.col, p.row] for p in path.waypoints],
        total_cost=float(path.total_cost),
        total_distance_m=float(path.total_distance_m),
        energy_score=float(path.energy_score),
        planner=req.planner,
    )


@app.get("/api/costmap-tiles/{row}/{col}")
async def get_costmap_tile(row: int, col: int, size: int = 64):
    r0 = max(0, row * size)
    c0 = max(0, col * size)
    r1 = min(TERRAIN.height, r0 + size)
    c1 = min(TERRAIN.width, c0 + size)
    tile = COST[r0:r1, c0:c1].flatten().tolist()
    return {"data": tile, "r0": r0, "c0": c0, "rows": r1 - r0, "cols": c1 - c0}


@app.websocket("/ws/rl-episode")
async def rl_episode_ws(ws: WebSocket):
    """Placeholder: full ONNX episode streaming comes after training (PRD §7)."""
    await ws.accept()
    try:
        await ws.receive_json()
        await ws.send_json(
            {
                "step": 0,
                "error": "RL WebSocket stub — train policy and wire onnxruntime (see PRD).",
                "done": True,
            }
        )
    except WebSocketDisconnect:
        pass
