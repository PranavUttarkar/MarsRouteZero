"""Smoke tests for MarsRoverEnv (requires built libmars)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "build", ROOT / "build" / "Release"):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

pytest.importorskip("gymnasium")

try:
    import libmars  # noqa: F401
except ImportError:
    pytest.skip("libmars extension not built", allow_module_level=True)

from rl.mars_env import MarsRoverEnv


def test_reset_fixed_start_goal():
    bin_path = ROOT / "data" / "costmap" / "jezero_elevation.bin"
    if not bin_path.is_file():
        pytest.skip("synthetic terrain missing; run scripts/gen_synthetic_terrain.py")
    env = MarsRoverEnv(str(bin_path), grid_size=512, mpp=1.0, random_start_goal=False)
    obs, _ = env.reset(
        options={
            "start_col": 50,
            "start_row": 50,
            "goal_col": 200,
            "goal_row": 200,
        }
    )
    assert obs.shape[0] == 15 * 15 + 5
    assert env.pos[0] == 50 and env.pos[1] == 50


def test_step_runs():
    bin_path = ROOT / "data" / "costmap" / "jezero_elevation.bin"
    if not bin_path.is_file():
        pytest.skip("no terrain binary")
    env = MarsRoverEnv(str(bin_path), 512, 1.0, random_start_goal=False)
    env.reset(
        options={
            "start_col": 80,
            "start_row": 80,
            "goal_col": 250,
            "goal_row": 250,
        }
    )
    import numpy as np

    a = np.array([0.1, 0.8], dtype=np.float32)
    _obs, _r, _t, _tr, info = env.step(a)
    assert "pos" in info
