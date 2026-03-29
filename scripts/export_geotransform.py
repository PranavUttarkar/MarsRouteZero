#!/usr/bin/env python3
"""Print GDAL geotransform JSON for a GeoTIFF (PRD §4.1)."""
from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("geotiff", type=str)
    args = p.parse_args()
    try:
        from osgeo import gdal
    except ImportError:
        print("Install GDAL Python bindings: pip install gdal", file=sys.stderr)
        raise SystemExit(1)
    gdal.AllRegister()
    ds = gdal.Open(args.geotiff)
    if not ds:
        raise SystemExit(f"Cannot open {args.geotiff}")
    gt = ds.GetGeoTransform()
    meta = {
        "geotransform": list(gt),
        "width": ds.RasterXSize,
        "height": ds.RasterYSize,
        "projection": ds.GetProjection(),
    }
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
