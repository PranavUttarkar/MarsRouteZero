"""PPO training entrypoint for MarsRoverEnv."""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

_ROOT = Path(__file__).resolve().parents[1]


def _add_libmars_paths() -> None:
    for extra in (
        os.environ.get("MARS_LIBMARS_BUILD", ""),
        str(_ROOT / "build"),
        str(_ROOT / "build" / "Release"),
        str(_ROOT / "build" / "Debug"),
        str(_ROOT / "build" / "RelWithDebInfo"),
    ):
        if extra and Path(extra).is_dir() and extra not in sys.path:
            sys.path.insert(0, extra)


_add_libmars_paths()

from rl.mars_env import MarsRoverEnv


@dataclass(frozen=True)
class PPOConfig:
    total_timesteps: int = 2_000_000
    n_envs: int = 8
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.005
    eval_freq: int = 10_000
    checkpoint_freq: int = 25_000
    progress_bar: bool = True


def _terrain_default() -> Path:
    return Path(
        os.environ.get("MARS_ELEVATION_BIN", _ROOT / "data/costmap/jezero_elevation.bin")
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PPO policy for MarsRoverEnv")
    parser.add_argument("--terrain-path", type=Path, default=_terrain_default())
    parser.add_argument("--grid-size", type=int, default=int(os.environ.get("MARS_GRID_SIZE", "512")))
    parser.add_argument(
        "--meters-per-pixel",
        type=float,
        default=float(os.environ.get("MARS_METERS_PER_PIXEL", "1.0")),
    )
    parser.add_argument("--output-dir", type=Path, default=_ROOT / "models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=PPOConfig.total_timesteps)
    parser.add_argument("--n-envs", type=int, default=PPOConfig.n_envs)
    parser.add_argument("--learning-rate", type=float, default=PPOConfig.learning_rate)
    parser.add_argument("--n-steps", type=int, default=PPOConfig.n_steps)
    parser.add_argument("--batch-size", type=int, default=PPOConfig.batch_size)
    parser.add_argument("--n-epochs", type=int, default=PPOConfig.n_epochs)
    parser.add_argument("--gamma", type=float, default=PPOConfig.gamma)
    parser.add_argument("--gae-lambda", type=float, default=PPOConfig.gae_lambda)
    parser.add_argument("--clip-range", type=float, default=PPOConfig.clip_range)
    parser.add_argument("--ent-coef", type=float, default=PPOConfig.ent_coef)
    parser.add_argument("--eval-freq", type=int, default=PPOConfig.eval_freq)
    parser.add_argument("--checkpoint-freq", type=int, default=PPOConfig.checkpoint_freq)
    parser.add_argument("--no-progress-bar", action="store_true")
    return parser


def _make_env_factory(
    terrain_path: Path,
    grid_size: int,
    meters_per_pixel: float,
    random_start_goal: bool = True,
):
    def _factory() -> MarsRoverEnv:
        return MarsRoverEnv(
            terrain_path=str(terrain_path),
            grid_size=grid_size,
            mpp=meters_per_pixel,
            random_start_goal=random_start_goal,
        )

    return _factory


def train_ppo(
    config: PPOConfig,
    terrain_path: Path,
    grid_size: int,
    meters_per_pixel: float,
    output_dir: Path,
    seed: int,
) -> Path:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.env_util import make_vec_env

    if not terrain_path.is_file():
        raise FileNotFoundError(f"Terrain binary not found: {terrain_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    best_dir = output_dir / "best"
    eval_logs_dir = output_dir / "eval"
    tb_log_dir = output_dir / "tensorboard"
    for directory in (checkpoints_dir, best_dir, eval_logs_dir, tb_log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    env_factory = _make_env_factory(terrain_path, grid_size, meters_per_pixel, True)
    train_env = make_vec_env(env_factory, n_envs=config.n_envs, seed=seed)
    eval_env = make_vec_env(env_factory, n_envs=1, seed=seed + 1)

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        policy_kwargs={"net_arch": [256, 256, 128]},
        tensorboard_log=str(tb_log_dir),
        seed=seed,
        verbose=1,
    )

    callbacks: Sequence[object] = (
        EvalCallback(
            eval_env=eval_env,
            best_model_save_path=str(best_dir),
            log_path=str(eval_logs_dir),
            eval_freq=max(config.eval_freq // config.n_envs, 1),
            n_eval_episodes=5,
            deterministic=True,
            render=False,
        ),
        CheckpointCallback(
            save_freq=max(config.checkpoint_freq // config.n_envs, 1),
            save_path=str(checkpoints_dir),
            name_prefix="mars_ppo",
        ),
    )

    model.learn(
        total_timesteps=config.total_timesteps,
        callback=list(callbacks),
        progress_bar=config.progress_bar,
    )

    model_path = output_dir / "mars_ppo_final"
    model.save(str(model_path))

    train_env.close()
    eval_env.close()
    return model_path.with_suffix(".zip")


def main() -> None:
    args = _build_arg_parser().parse_args()
    config = PPOConfig(
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        eval_freq=args.eval_freq,
        checkpoint_freq=args.checkpoint_freq,
        progress_bar=not args.no_progress_bar,
    )

    model_zip = train_ppo(
        config=config,
        terrain_path=args.terrain_path.resolve(),
        grid_size=args.grid_size,
        meters_per_pixel=args.meters_per_pixel,
        output_dir=args.output_dir.resolve(),
        seed=args.seed,
    )
    print(f"Saved final PPO model: {model_zip}")


if __name__ == "__main__":
    main()
