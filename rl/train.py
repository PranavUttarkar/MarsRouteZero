"""Train PPO on MarsRoverEnv (requires built libmars on PYTHONPATH / build/)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

for _p in (
    _ROOT,
    _ROOT / "build",
    _ROOT / "build" / "Release",
    _ROOT / "build" / "Debug",
):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.mars_env import MarsRoverEnv


def main() -> None:
    ap = argparse.ArgumentParser(description="PPO training for Mars Route Zero")
    ap.add_argument(
        "--terrain",
        type=Path,
        default=_ROOT / "data/costmap/jezero_elevation.bin",
        help="Float32 elevation binary",
    )
    ap.add_argument("--grid", type=int, default=512)
    ap.add_argument("--mpp", type=float, default=1.0)
    ap.add_argument("--timesteps", type=int, default=50_000, help="Use --smoke for a short run")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Quick dev run (~8k steps, small batch)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "models" / "mars_ppo_latest.zip",
        help="Where to save the SB3 zip",
    )
    ap.add_argument(
        "--tensorboard",
        type=Path,
        nargs="?",
        const=_ROOT / "logs",
        default=None,
        metavar="DIR",
        help="Log to TensorBoard (requires: pip install tensorboard). Default DIR: ./logs",
    )
    args = ap.parse_args()

    if not args.terrain.is_file():
        raise SystemExit(f"Terrain file not found: {args.terrain}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    ts = 8_192 if args.smoke else args.timesteps
    n_steps = 512 if args.smoke else 2048
    batch = 128 if args.smoke else 256

    def make_env():
        return MarsRoverEnv(str(args.terrain), args.grid, args.mpp, random_start_goal=True)

    venv = DummyVecEnv([make_env])

    tb_log = str(args.tensorboard) if args.tensorboard is not None else None

    model = PPO(
        "MlpPolicy",
        venv,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=batch,
        n_epochs=6,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[256, 256, 128]),
        verbose=1,
        tensorboard_log=tb_log,
    )
    model.learn(total_timesteps=ts)
    model.save(str(args.out))
    print(f"Saved policy to {args.out}")


if __name__ == "__main__":
    main()
