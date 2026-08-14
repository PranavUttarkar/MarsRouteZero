"""Optuna hyperparameter optimization for PPO on MarsRoverEnv."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

from rl.mars_env import MarsRoverEnv


def _make_env_factory(terrain_path: Path, grid_size: int, meters_per_pixel: float):
    def _factory() -> MarsRoverEnv:
        return MarsRoverEnv(
            terrain_path=str(terrain_path),
            grid_size=grid_size,
            mpp=meters_per_pixel,
            random_start_goal=True,
        )

    return _factory


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune PPO hyperparameters for MarsRoverEnv")
    parser.add_argument("--terrain-path", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=512)
    parser.add_argument("--meters-per-pixel", type=float, default=1.0)
    parser.add_argument("--timesteps-per-trial", type=int, default=500_000)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--study-name", type=str, default="mars_ppo_hpo")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_mars.db")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    terrain_path = args.terrain_path.resolve()
    if not terrain_path.is_file():
        raise FileNotFoundError(f"Terrain binary not found: {terrain_path}")

    env_factory = _make_env_factory(terrain_path, args.grid_size, args.meters_per_pixel)

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
        n_epochs = trial.suggest_int("n_epochs", 5, 20)
        gamma = trial.suggest_float("gamma", 0.95, 0.999)
        gae_lambda = trial.suggest_float("gae_lambda", 0.85, 0.99)
        clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
        ent_coef = trial.suggest_float("ent_coef", 0.0, 0.02)

        train_env = make_vec_env(env_factory, n_envs=args.n_envs, seed=args.seed + trial.number)
        eval_env = make_vec_env(env_factory, n_envs=1, seed=args.seed + 10_000 + trial.number)

        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=lr,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            policy_kwargs={"net_arch": [256, 256, 128]},
            seed=args.seed + trial.number,
            verbose=0,
        )
        model.learn(total_timesteps=args.timesteps_per_trial, progress_bar=False)

        mean_reward, _ = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=10,
            deterministic=True,
            return_episode_rewards=False,
        )
        train_env.close()
        eval_env.close()
        return float(np.asarray(mean_reward).item())

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.n_trials, n_jobs=1)

    print(f"Best value: {study.best_value}")
    print(f"Best params: {study.best_params}")


if __name__ == "__main__":
    main()
