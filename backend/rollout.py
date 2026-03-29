"""Policy rollout helpers for WebSocket (PPO or goal-seeking heuristic)."""
from __future__ import annotations

from typing import Any

import numpy as np


def heuristic_action(env: Any) -> np.ndarray:
    """Steer toward goal with moderate speed; works without a trained network."""
    goal_vec = env.goal - env.pos
    target_deg = float(np.degrees(np.arctan2(goal_vec[1], goal_vec[0])) % 360.0)
    err = (target_deg - env.heading_deg + 180.0) % 360.0 - 180.0
    delta = float(np.clip(err / env.HEADING_LIMIT, -1.0, 1.0))
    return np.array([delta, 0.88], dtype=np.float32)


def predict_action(model: Any, obs: np.ndarray) -> np.ndarray:
    """SB3 PPO.predict wrapper."""
    action, _ = model.predict(obs, deterministic=True)
    return np.asarray(action, dtype=np.float32).reshape(-1)
