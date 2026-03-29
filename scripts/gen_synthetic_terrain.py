#!/usr/bin/env python3
"""Write float32 elevation for dev builds — Jezero-crater-inspired (not real HiRISE).

Real competition data: use AWS HiRISE pipeline in README + PRD §4.1.
This replaces the old sin×cos heightfield that looked like a checkerboard in 3D.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def synthetic_jezero_like(n: int) -> np.ndarray:
    """Radial crater depression + rim + multi-frequency undulation (single channel, no grid artifacts)."""
    xv, yv = np.meshgrid(np.arange(n, dtype=np.float32), np.arange(n, dtype=np.float32), indexing="xy")
    cx = cy = (n - 1) * 0.5
    # Normalized radial coords (~1 at edge of ellipse)
    dx = (xv - cx) / (n * 0.38)
    dy = (yv - cy) / (n * 0.36)
    r = np.sqrt(dx * dx + dy * dy)

    # Main bowl + raised annulus (crater rim)
    bowl = -42.0 * np.exp(-(r * r) / 0.55)
    rim = 9.0 * np.exp(-((r - 0.95) ** 2) / 0.06)

    # Medium bumps (dunes / yardangs) — several incommensurate frequencies avoids regular stripes
    und = (
        4.2 * np.sin(0.031 * xv + 0.7) * np.cos(0.027 * yv - 0.4)
        + 2.8 * np.sin(0.071 * xv - 0.2 * yv)
        + 1.9 * np.cos(0.052 * (xv + yv))
        + 1.1 * np.sin(0.12 * xv) * np.sin(0.09 * yv)
    )

    # Fine roughness
    fine = 0.65 * np.sin(0.19 * xv + 0.11 * yv) + 0.35 * np.cos(0.23 * yv)

    z = 118.0 + bowl + rim + und + fine
    return z.astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--out-bin", type=Path, default=Path("data/costmap/jezero_elevation.bin"))
    p.add_argument("--out-meta", type=Path, default=Path("data/dtm/jezero_meta.json"))
    args = p.parse_args()
    n = args.size
    args.out_bin.parent.mkdir(parents=True, exist_ok=True)
    args.out_meta.parent.mkdir(parents=True, exist_ok=True)

    z = synthetic_jezero_like(n)
    args.out_bin.write_bytes(z.tobytes())

    meta = {
        "width": n,
        "height": n,
        "meters_per_pixel": 1.0,
        "description": "Synthetic crater-like heightfield for Mars Route Zero (replace with HiRISE Jezero DTM for production)",
    }
    args.out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {args.out_bin} ({n}x{n} float32), z=[{float(z.min()):.2f}, {float(z.max()):.2f}] m")
    print(f"Wrote {args.out_meta}")


if __name__ == "__main__":
    main()
