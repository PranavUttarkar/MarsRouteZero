#!/usr/bin/env python3
"""Resample a HiRISE DTM GeoTIFF (COG) to float32 grid for libmars + update jezero_meta.json.

Uses rasterio (pip install rasterio) when GDAL command-line tools are not installed.

Example (repo root, after download_jezero_dtm):
  python scripts/hirise_geotiff_to_elevation_bin.py \\
    --input data/dtm/aws_sync/DTEEC_045994_1985_046060_1985_U01.tif
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    try:
        import rasterio
        from rasterio.enums import Resampling
    except ImportError as e:
        raise SystemExit("Install rasterio: pip install rasterio") from e

    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=Path,
        default=Path("data/dtm/aws_sync/DTEEC_045994_1985_046060_1985_U01.tif"),
        help="HiRISE DTM COG (.tif)",
    )
    p.add_argument("--size", type=int, default=512, help="Output height = width")
    p.add_argument(
        "--out-bin",
        type=Path,
        default=Path("data/costmap/jezero_elevation.bin"),
    )
    p.add_argument(
        "--out-meta",
        type=Path,
        default=Path("data/dtm/jezero_meta.json"),
    )
    args = p.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Not found: {args.input}")

    n = args.size
    args.out_bin.parent.mkdir(parents=True, exist_ok=True)
    args.out_meta.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(args.input) as ds:
        arr = ds.read(
            1,
            out_shape=(n, n),
            resampling=Resampling.bilinear,
        ).astype(np.float32)
        # HiRISE COGs use a huge negative float as nodata; bilinear can blend edges.
        arr = np.where(np.isfinite(arr) & (arr > -1e10), arr, np.nan)

        t = ds.transform
        width_m = ds.width * abs(t.a)
        height_m = ds.height * abs(t.e)
        mpp_x = width_m / n
        mpp_y = height_m / n
        meters_per_pixel = float((mpp_x + mpp_y) * 0.5)

        meta_extra = {
            "crs": ds.crs.to_string() if ds.crs else None,
            "source_geotiff": str(args.input.as_posix()),
            "source_shape": [int(ds.height), int(ds.width)],
            "geotransform_source": [t.a, t.b, t.c, t.d, t.e, t.f],
        }

    valid = arr[np.isfinite(arr)]
    fill = float(np.median(valid)) if valid.size else 0.0
    arr = np.where(np.isfinite(arr), arr, fill)

    args.out_bin.write_bytes(arr.tobytes(order="C"))

    meta = {
        "width": n,
        "height": n,
        "meters_per_pixel": meters_per_pixel,
        "description": "Resampled from HiRISE controlled DTM COG (Mars Route Zero)",
        "elevation_min_m": float(arr.min()),
        "elevation_max_m": float(arr.max()),
        **meta_extra,
    }
    args.out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"Wrote {args.out_bin} ({n}x{n} float32), z=[{float(arr.min()):.2f}, {float(arr.max()):.2f}] m, "
        f"mpp={meters_per_pixel:.3f}"
    )
    print(f"Wrote {args.out_meta}")


if __name__ == "__main__":
    main()
