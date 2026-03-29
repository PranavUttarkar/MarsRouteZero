# Mars Route Zero — Implementation Status

Living checklist vs `mars-route-zero-prd.md`.

---

## Summary

| Area | Status |
|------|--------|
| C++ `libmars` | Done — binary, slope, costmap, A*, tests, pybind, `dll_windows.py` |
| Real NASA DTM | **In pipeline** — `download_jezero_dtm.*` (astrogeo-ard), `hirise_geotiff_to_elevation_bin.py` → `jezero_elevation.bin` + `jezero_meta.json`; API reads `meters_per_pixel` from meta unless `MARS_METERS_PER_PIXEL` is set |
| RL training | **Done** — `train.py` (PPO, `--tensorboard` optional) |
| RL HPO | **Done** — `optimize.py` (Optuna, optional SQLite storage) |
| ONNX export | **Done** — `python -m rl.export_onnx` → `frontend/public/mars_policy.onnx` |
| FastAPI | **Done** — terrain, plan, heightmap, **costmap PNG**, cell, **rl-status**, **perseverance-waypoints**, WebSocket |
| Frontend | **Done** — story mode, explore, cost overlay, Perseverance markers, stats + insight copy, `onnxruntime-web` dep |
| Browser ONNX | **Optional** — `loadOnnxPolicySession()` in `src/rl/onnxClient.js` (not wired into main loop; server RL is default) |

---

## Commands (quick reference)

| Task | Command |
|------|---------|
| Synthetic terrain | `python scripts/gen_synthetic_terrain.py` |
| HiRISE COG → app grid | `python scripts/hirise_geotiff_to_elevation_bin.py --input data/dtm/aws_sync/DTEEC_*.tif` |
| Train PPO | `PYTHONPATH=build python -m rl.train --smoke` |
| Optuna HPO | `PYTHONPATH=build python -m rl.optimize --trials 8` |
| Export ONNX | `python -m rl.export_onnx` |
| API | `uvicorn backend.main:app --reload --port 8000` |
| UI | `cd frontend && npm run dev` |

---

## Not done / stretch (PRD Phase 5)

- [ ] Multi-tile HiRISE Jezero mosaic (optional; single-tile real DTM path works).
- [ ] AI4MARS terrain class overlay.
- [ ] Elevation **cross-section** chart (D3/Chart.js).
- [ ] RL **value heatmap** overlay.
- [ ] Full **browser** ONNX rollout (same as server) — client stub exists.
- [ ] Mobile layout polish, LOD tiles.

---

## Milestones (PRD §11)

| Milestone | Met? |
|-----------|------|
| `build_full_terrain` → 512 | Yes |
| PPO + zip | Yes (`models/mars_ppo_latest.zip`) |
| ONNX in `/public` | After `export_onnx` |
| E2E terrain + paths + RL | Yes |

---

*Updated: 2026-03-29 — ONNX, Optuna, data scripts, story UI, cost/waypoint APIs.*
