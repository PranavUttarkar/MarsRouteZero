# Mars Route Zero

Interactive Mars terrain rover navigation simulator (see `mars-route-zero-prd.md`).

## Build order (from the PRD)

1. **Data** — Jezero DTM from AWS (production) or synthetic grid for dev:  
   `python scripts/gen_synthetic_terrain.py`
2. **C++ core** — `libmars` via CMake: terrain I/O, Sobel slope, costmap, A\*, pybind11.
3. **Python** — RL env (`rl/mars_env.py`), training, ONNX export.
4. **Backend** — FastAPI serves terrain metadata and `/api/plan`.
5. **Frontend** — Vite + React + Three.js visualization.

## C++ and Python extension

Requirements: CMake 3.18+, C++17 compiler, Python 3 with development headers.

```bash
python scripts/gen_synthetic_terrain.py
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

The extension module is built as `libmars` (e.g. `libmars*.so` or `libmars*.pyd` inside `build/`). Add that directory to `PYTHONPATH`, or set `MARS_LIBMARS_BUILD` to point at it.

Optional GeoTIFF loading: configure with `-DMARS_USE_GDAL=ON` and install GDAL dev packages.

### Windows (no Visual Studio)

If CMake picks **NMake** and fails with `nmake` / `CMAKE_CXX_COMPILER not set`, you do not have the MSVC toolchain on `PATH`. Options:

1. **MSYS2 MinGW** (if you have `C:\msys64\ucrt64\bin\g++.exe`): configure with the MinGW generator so CMake does not use NMake:

   ```powershell
   $env:CC = "C:/msys64/ucrt64/bin/gcc.exe"
   $env:CXX = "C:/msys64/ucrt64/bin/g++.exe"
   cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -B build -S .
   cmake --build build
   ```

   Or use the preset: `cmake --preset windows-mingw` then `cmake --build build`.

2. **Visual Studio 2022** with “Desktop development with C++”: then  
   `cmake -G "Visual Studio 17 2022" -A x64 -B build -S .`  
   and build the `Release` configuration (output is often under `build/Release/`).

After a **MinGW** build, Python needs the folder containing `libmars*.pyd` (usually `build\`) on `PYTHONPATH`, **and** the MinGW runtime DLLs (`libgcc`, `libstdc++`, `libgomp`, …) from `ucrt64\bin`. The backend calls `os.add_dll_directory()` for common MSYS2 paths and prepends `PATH`; if you still see **`DLL load failed while importing libmars`**, set:

`set MSYS2_UCRT64_BIN=C:\msys64\ucrt64\bin` (adjust if your MSYS2 install path differs), then restart uvicorn.

## Backend and frontend

```bash
pip install -r requirements.txt
# from repo root, with PYTHONPATH including CMake build output:
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

```bash
cd frontend && npm install && npm run dev
```

The Vite dev server proxies `/api` and `/ws` to port 8000.

Open **http://127.0.0.1:5173** — click terrain for **start** and **goal**, then **Run A* + straight** or **Stream RL** (WebSocket uses a trained PPO zip if `models/mars_ppo_latest.zip` exists, otherwise a goal-seeking heuristic).

## RL training (smoke)

From the repo root, with `PYTHONPATH` including the CMake `build/` directory (where `libmars` lives):

```bash
pip install -r requirements.txt
python -m rl.train --smoke
```

This writes **`models/mars_ppo_latest.zip`**. Restart `uvicorn` so `/ws/rl-episode` can load the policy (requires `stable-baselines3`, `torch`, `gymnasium`). TensorBoard logging is off by default; use **`--tensorboard`** (optional directory, default `./logs`) after `pip install tensorboard`.

### Optuna HPO

```bash
PYTHONPATH=build python -m rl.optimize --timesteps-per-trial 12000 --trials 8
```

Optional: `--storage sqlite:///optuna_mars.db` to resume.

### ONNX (browser / deployment)

```bash
python -m rl.export_onnx
```

Writes **`frontend/public/mars_policy.onnx`**. The UI includes `onnxruntime-web`; server-side RL streaming remains the primary demo path.

## NASA DTM scripts (production data)

- **`scripts/download_jezero_dtm.sh`** — `aws s3 sync` from AWS Open Data (needs AWS CLI).
- **`scripts/preprocess_terrain.py`** — crop with `gdal_translate` (needs GDAL).
- **`scripts/export_geotransform.py`** — GeoTIFF → JSON metadata (needs Python GDAL bindings).

Dev builds continue to use **`scripts/gen_synthetic_terrain.py`**.

## Tests

```bash
pytest tests/test_env.py -v
```

(C++ tests: `ctest --test-dir build`.)

## Docker (optional, Linux/WSL)

Requires a Linux-built `libmars` under `./build`. See `docker-compose.yml`.

## License

MIT — see `LICENSE`.
