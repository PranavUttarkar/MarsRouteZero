#!/usr/bin/env python3
"""Crop a merged Jezero DTM GeoTIFF (PRD §4.1). Extend with gdal_translate for binary/PNG exports.

Example after download_jezero_dtm.sh + gdal_merge:
  python scripts/preprocess_terrain.py --mosaic jezero_mosaic.tif --crop-out data/dtm/jezero_5km.tif

Then export float32 grid for libmars (from repo root, GDAL CLI):
  gdal_translate -ot Float32 -outsize 512 512 data/dtm/jezero_5km.tif data/costmap/jezero_float.tif
  python -c "import numpy as np; from osgeo import gdal; ..."  # or use gdal_translate -of ENVI
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mosaic", type=Path, required=True, help="Merged GeoTIFF path")
    p.add_argument(
        "--crop-out",
        type=Path,
        default=Path("data/dtm/jezero_5km.tif"),
        help="Output cropped GeoTIFF",
    )
    p.add_argument(
        "--projwin",
        nargs=4,
        type=float,
        default=[77.3, 18.6, 77.55, 18.35],
        metavar=("ULX", "ULY", "LRX", "LRY"),
        help="Upper-left / lower-right corner (lon/lat) for gdal_translate -projwin",
    )
    args = p.parse_args()
    if not args.mosaic.is_file():
        raise SystemExit(f"Not found: {args.mosaic}")

    args.crop_out.parent.mkdir(parents=True, exist_ok=True)
    ulx, uly, lrx, lry = args.projwin
    run(
        [
            "gdal_translate",
            "-projwin",
            str(ulx),
            str(uly),
            str(lrx),
            str(lry),
            str(args.mosaic),
            str(args.crop_out),
        ]
    )
    print("Wrote", args.crop_out)
    print("See script docstring for float32 binary + Three.js PNG steps.")


if __name__ == "__main__":
    main()
