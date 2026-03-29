"""Optuna hyperparameter search for PPO on MarsRoverEnv (PRD §6.3)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import optuna
from optuna.pruners import MedianPruner
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "build", _ROOT / "build" / "Release"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dll_windows import ensure_mingw_dll_dirs

ensure_mingw_dll_dirs(_ROOT)

from rl.mars_env import MarsRoverEnv


def main() -> None:
    ap = argparse.ArgumentParser(description="Optuna HPO for Mars PPO")
    ap.add_argument("--terrain", type=Path, default=_ROOT / "data/costmap/jezero_elevation.bin")
    ap.add_argument("--grid", type=int, default=512)
    ap.add_argument("--mpp", type=float, default=1.0)
    ap.add_argument("--timesteps-per-trial", type=int, default=12_000)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--storage", type=str, default="", help="e.g. sqlite:///optuna_mars.db")
    ap.add_argument("--study-name", type=str, default="mars_ppo")
    args = ap.parse_args()

    if not args.terrain.is_file():
        raise SystemExit(f"Terrain not found: {args.terrain}")

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True)
        ent = trial.suggest_float("ent_coef", 0.0, 0.012)
        clip = trial.suggest_float("clip_range", 0.1, 0.22)
        n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])
        gamma = trial.suggest_float("gamma", 0.985, 0.997)

        def make_env():
            return MarsRoverEnv(str(args.terrain), args.grid, args.mpp, random_start_goal=True)

        venv = DummyVecEnv([make_env])
        model = PPO(
            "MlpPolicy",
            venv,
            learning_rate=lr,
            n_steps=n_steps,
            batch_size=min(256, n_steps),
            n_epochs=10,
            gamma=gamma,
            gae_lambda=0.92,
            clip_range=clip,
            ent_coef=ent,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[256, 256, 128]),
            verbose=0,
            tensorboard_log=None,
        )
        model.learn(total_timesteps=args.timesteps_per_trial)
        mean_r, _ = evaluate_policy(model, venv, n_eval_episodes=3, deterministic=True)
        return float(mean_r)

    study_kwargs: dict = {
        "direction": "maximize",
        "pruner": MedianPruner(n_startup_trials=2, n_warmup_steps=0),
    }
    if args.storage.strip():
        study_kwargs["storage"] = args.storage
        study_kwargs["study_name"] = args.study_name
        study_kwargs["load_if_exists"] = True

    study = optuna.create_study(**study_kwargs)
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)
    print("Best value:", study.best_value)
    print("Best params:", study.best_params)
    out_csv = _ROOT / "results" / "optuna_results.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
