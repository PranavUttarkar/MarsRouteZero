#!/usr/bin/env python3
"""Write a float32 elevation grid for development when Jezero DTM is not downloaded."""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--out-bin", type=Path, default=Path("data/costmap/jezero_elevation.bin"))
    p.add_argument("--out-meta", type=Path, default=Path("data/dtm/jezero_meta.json"))
    args = p.parse_args()
    n = args.size
    args.out_bin.parent.mkdir(parents=True, exist_ok=True)
    args.out_meta.parent.mkdir(parents=True, exist_ok=True)

    x = np.linspace(0, 4 * np.pi, n, dtype=np.float32)
    y = np.linspace(0, 4 * np.pi, n, dtype=np.float32)
    xv, yv = np.meshgrid(x, y, indexing="xy")
    # Meters-ish undulation + gentle plane (synthetic, not real Mars)
    z = 100.0 + 8.0 * np.sin(xv) * np.cos(yv) + 0.02 * xv + 0.01 * yv
    z = z.astype(np.float32)
    args.out_bin.write_bytes(z.tobytes())

    meta = {
        "width": n,
        "height": n,
        "meters_per_pixel": 1.0,
        "description": "Synthetic height field for Mars Route Zero dev builds",
    }
    args.out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {args.out_bin} ({n}x{n} float32)")
    print(f"Wrote {args.out_meta}")


if __name__ == "__main__":
    main()
