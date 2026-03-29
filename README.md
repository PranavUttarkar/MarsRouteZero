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

## License

MIT — see `LICENSE`.
