"""PPO training entrypoint (requires libmars + MarsRoverEnv)."""
from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "Training pipeline not wired in this scaffold. "
        "Build libmars, generate data/costmap/jezero_elevation.bin, then "
        "add Stable-Baselines3 VecEnv setup per mars-route-zero-prd.md §6.2."
    )


if __name__ == "__main__":
    main()
