"""Gymnasium environment wrapping libmars (build C++ extension first)."""
from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

_RL_ROOT = Path(__file__).resolve().parents[1]
if str(_RL_ROOT) not in sys.path:
    sys.path.insert(0, str(_RL_ROOT))

from dll_windows import ensure_mingw_dll_dirs

ensure_mingw_dll_dirs(_RL_ROOT)

import libmars


class MarsRoverEnv(gym.Env):
    """Mars rover navigation on HiRISE-derived elevation grid (via libmars costmap)."""

    metadata = {"render_modes": ["rgb_array"]}

    PATCH_SIZE = 15
    MAX_STEPS = 2000
    GOAL_RADIUS = 3
    STEP_SIZE_M = 2.0
    HEADING_LIMIT = 30.0

    def __init__(
        self,
        terrain_path: str,
        grid_size: int = 512,
        mpp: float = 1.0,
        random_start_goal: bool = True,
    ):
        super().__init__()
        self.grid = libmars.build_full_terrain(terrain_path, grid_size, grid_size, mpp)
        self.W = self.grid.width
        self.H = self.grid.height
        self.random_start_goal = random_start_goal

        patch_cells = self.PATCH_SIZE * self.PATCH_SIZE
        obs_dim = patch_cells + 5
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )
        self._reset_state()

    def _reset_state(self):
        self.pos = np.array([self.W // 2, self.H // 2], dtype=float)
        self.goal = np.array([self.W * 3 // 4, self.H * 3 // 4], dtype=float)
        self.heading_deg = 0.0
        self.step_count = 0
        self.prev_dist = np.linalg.norm(self.goal - self.pos)
        self.trajectory = [self.pos.copy()]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        # Fixed start/goal (e.g. WebSocket demo): col, row order matches planner / libmars
        if "start_col" in options and "goal_col" in options:
            self.pos = np.array(
                [float(options["start_col"]), float(options["start_row"])], dtype=float
            )
            self.goal = np.array(
                [float(options["goal_col"]), float(options["goal_row"])], dtype=float
            )
        elif self.random_start_goal:
            margin = self.PATCH_SIZE
            rng = self.np_random
            while True:
                r = int(rng.integers(margin, self.H - margin))
                c = int(rng.integers(margin, self.W - margin))
                if self.grid.get_cell_cost(r, c) < 0.5:
                    self.pos = np.array([c, r], dtype=float)
                    break
            while True:
                r = int(rng.integers(margin, self.H - margin))
                c = int(rng.integers(margin, self.W - margin))
                dist = np.linalg.norm(np.array([c, r]) - self.pos)
                if self.grid.get_cell_cost(r, c) < 0.5 and dist > 50:
                    self.goal = np.array([c, r], dtype=float)
                    break
        else:
            self._reset_state()

        self.heading_deg = float(options.get("heading_deg", 0.0))
        self.step_count = 0
        self.prev_dist = np.linalg.norm(self.goal - self.pos)
        self.trajectory = [self.pos.copy()]
        return self._get_obs(), {}

    def step(self, action):
        delta_h = float(action[0]) * self.HEADING_LIMIT
        speed_f = float(action[1])

        self.heading_deg += delta_h
        self.heading_deg = self.heading_deg % 360.0

        rad = np.deg2rad(self.heading_deg)
        dx = np.cos(rad) * self.STEP_SIZE_M * (0.3 + 0.7 * speed_f)
        dy = np.sin(rad) * self.STEP_SIZE_M * (0.3 + 0.7 * speed_f)

        new_pos = self.pos + np.array([dx, dy])
        pad = self.PATCH_SIZE // 2
        new_pos[0] = np.clip(new_pos[0], pad, self.W - pad - 1)
        new_pos[1] = np.clip(new_pos[1], pad, self.H - pad - 1)

        self.pos = new_pos
        self.trajectory.append(self.pos.copy())
        self.step_count += 1

        col, row = int(self.pos[0]), int(self.pos[1])
        cell_cost = self.grid.get_cell_cost(row, col)

        dist_to_goal = np.linalg.norm(self.goal - self.pos)
        progress = (self.prev_dist - dist_to_goal) / max(self.prev_dist, 1.0)
        self.prev_dist = dist_to_goal

        reward = progress * 10.0
        reward -= cell_cost * 2.0
        reward -= 0.01

        terminated = False
        truncated = False

        if dist_to_goal < self.GOAL_RADIUS:
            reward += 100.0
            terminated = True
        elif cell_cost >= 0.85:
            reward -= 20.0
            terminated = True
        elif self.step_count >= self.MAX_STEPS:
            reward -= 10.0
            truncated = True

        return (
            self._get_obs(),
            reward,
            terminated,
            truncated,
            {
                "cell_cost": cell_cost,
                "dist_to_goal": dist_to_goal,
                "pos": self.pos.copy(),
                "trajectory": self.trajectory.copy(),
            },
        )

    def _get_obs(self) -> np.ndarray:
        half = self.PATCH_SIZE // 2
        col, row = int(self.pos[0]), int(self.pos[1])

        patch = np.zeros((self.PATCH_SIZE, self.PATCH_SIZE), dtype=np.float32)
        for dr in range(-half, half + 1):
            for dc in range(-half, half + 1):
                r2, c2 = row + dr, col + dc
                if 0 <= r2 < self.H and 0 <= c2 < self.W:
                    patch[dr + half, dc + half] = self.grid.get_cell_cost(r2, c2)
                else:
                    patch[dr + half, dc + half] = 1.0

        goal_vec = self.goal - self.pos
        goal_dist = np.linalg.norm(goal_vec)
        goal_dir = goal_vec / (goal_dist + 1e-8)
        dist_norm = goal_dist / (np.sqrt(self.W**2 + self.H**2))
        heading_rad = np.deg2rad(self.heading_deg)

        context = np.array(
            [
                goal_dir[0],
                goal_dir[1],
                dist_norm,
                np.cos(heading_rad),
                np.sin(heading_rad),
            ],
            dtype=np.float32,
        )

        return np.concatenate([patch.flatten(), context])
