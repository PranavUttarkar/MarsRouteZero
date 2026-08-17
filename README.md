# RL Proximal Policy Optimization for Energy-Efficient Rover Navigation

**Mars Route Zero** trains a reinforcement-learning agent to navigate real Martian terrain while minimizing energy expenditure. A PPO (Proximal Policy Optimization) policy learns to steer a rover across HiRISE-derived elevation models of Jezero Crater — avoiding steep slopes, ridges, and craters — and is compared side-by-side against a classical energy-weighted A\* planner in an interactive 3D visualization.

## How it works

1. **Terrain pipeline** — NASA HiRISE Digital Terrain Models of Jezero Crater (AWS Open Data) are resampled into a 512×512 float32 elevation grid. A C++ core (`libmars`) computes Sobel slope maps and a traversal costmap from the elevation data.
2. **Simulation environment** — A Gymnasium environment wraps `libmars` (via pybind11). The rover observes a local 15×15 costmap patch plus goal direction, distance, and heading; it acts with continuous steering and speed commands. Rewards combine potential-based goal shaping with energy-aligned penalties: per-step energy cost (`cost × distance`), steep-slope transitions, ridge/valley crossings, and revisit penalties.
3. **Learning** — Stable-Baselines3 PPO trains on vectorized environments with evaluation and checkpoint callbacks; hyperparameters are tuned with Optuna. The trained policy exports to ONNX for in-browser inference.
4. **Visualization** — A FastAPI backend serves terrain data, A\* plans, and live RL rollouts over WebSocket to a React + Three.js frontend, where learned trajectories render on the 3D terrain alongside the classical planner's path.

## Why it matters

Energy is the binding constraint on planetary rovers: solar and RTG power budgets limit how far a rover can drive per sol, and every meter of unnecessary climb costs science time. Route planning today relies on human operators and classical planners over precomputed costmaps. A learned policy that internalizes terrain-energy trade-offs can propose efficient routes directly from local observations — without a global replan — and the same approach transfers to Earth applications like autonomous navigation for agricultural, mining, and search-and-rescue robots operating on rough terrain.

## Repository layout

```
include/, src/       C++ core "libmars": terrain I/O, Sobel slope, costmap, A* planner
python/bindings.cpp  pybind11 bindings exposing libmars to Python
rl/                  Gymnasium env (mars_env.py), PPO training (train.py),
                     Optuna HPO (optimize.py), ONNX export (export_onnx.py)
backend/             FastAPI server: terrain metadata, /api/plan, RL WebSocket streaming
frontend/            Vite + React + Three.js visualization (onnxruntime-web capable)
scripts/             Terrain data pipeline: synthetic grid, HiRISE DTM download/preprocess
tests/               C++ (ctest) and Python (pytest) tests
data/                Terrain binaries + metadata (large HiRISE rasters are local-only)
```

## Quick start

Requirements: CMake 3.18+, a C++17 compiler, Python 3 with development headers, Node.js.

### 1. Build the C++ core

```bash
python scripts/gen_synthetic_terrain.py        # dev terrain (or use real HiRISE data below)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

The `libmars` extension module is built into `build/`. Add that directory to `PYTHONPATH`, or set `MARS_LIBMARS_BUILD` to point at it. On Windows without MSVC, use the MinGW preset: `cmake --preset windows-mingw` (see `CMakePresets.json`).

### 2. Train the PPO policy

```bash
pip install -r requirements.txt
python -m rl.train --smoke                            # quick dev run (~8k steps)
python -m rl.train --timesteps 2000000 --n-envs 8     # full training run
```

Outputs (under `--out-dir`, default `models/`):

- `models/mars_ppo_latest.zip` — final policy (loaded by the backend)
- `models/best/` — best model saved by the eval callback
- `models/checkpoints/` — periodic checkpoints
- `models/eval/` — evaluation logs

All PPO hyperparameters are exposed as flags (`--learning-rate`, `--n-steps`, `--batch-size`, `--n-epochs`, `--gamma`, `--gae-lambda`, `--clip-range`, `--ent-coef`); defaults are Optuna-tuned. Add `--tensorboard` for TensorBoard logging.

### 3. Tune hyperparameters (optional)

```bash
python -m rl.optimize --timesteps-per-trial 12000 --trials 8
```

Add `--storage sqlite:///optuna_mars.db` to persist and resume studies. Results are written to `results/optuna_results.csv`.

### 4. Export to ONNX (optional, browser inference)

```bash
python -m rl.export_onnx
```

Writes `frontend/public/mars_policy.onnx` for `onnxruntime-web`.

### 5. Run the app

```bash
# Backend (repo root, with PYTHONPATH including build/)
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

# Frontend
cd frontend && npm install && npm run dev
```

Open **http://127.0.0.1:5173** — click the terrain to set **start** and **goal**, then run the A\* planner or stream a live RL episode (uses the trained PPO policy if `models/mars_ppo_latest.zip` exists, otherwise a goal-seeking heuristic).

## Real NASA terrain data

Development uses a synthetic grid. For real Jezero Crater terrain:

```bash
./scripts/download_jezero_dtm.sh      # or .ps1 on Windows — AWS Open Data, no account needed
python scripts/hirise_geotiff_to_elevation_bin.py --input data/dtm/aws_sync/DTEEC_*.tif
```

This writes `data/costmap/jezero_elevation.bin` and `data/dtm/jezero_meta.json` (including `meters_per_pixel`). Restart the backend to reload. See `scripts/README.md` for details. The raw multi-GB GeoTIFFs stay local and are not committed.

## Tests

```bash
pytest tests/test_env.py -v         # Python environment tests
ctest --test-dir build              # C++ tests
```

## License

MIT — see `LICENSE`.
