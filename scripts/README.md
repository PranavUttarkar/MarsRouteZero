# Scripts — where to run them

Run **all commands from the repository root** (`MarsRouteZero/`), i.e. the folder that contains `CMakeLists.txt` and `data/`.

## 1. Synthetic terrain (dev — no AWS)

Creates `data/costmap/jezero_elevation.bin` for local development:

```powershell
# Windows PowerShell or cmd (from repo root)
python -m pip install numpy
python scripts/gen_synthetic_terrain.py
```

Then **restart the FastAPI server** so it reloads the elevation file.

```bash
# Linux / macOS
python3 scripts/gen_synthetic_terrain.py
```

## 2. Download real HiRISE DTMs (AWS Open Data)

**Registry:** [NASA / USGS Released HiRISE Digital Terrain Models](https://registry.opendata.aws/nasa-usgs-mars-hirise-dtms/)

**S3:** `s3://astrogeo-ard/mars/mro/hirise/controlled/dtm/`  
**ARN:** `arn:aws:s3:::astrogeo-ard/mars/mro/hirise/controlled/dtm`  
**Region:** `us-west-2` — use **`--region us-west-2`** on every `aws` call (other regions often yield **NoSuchBucket**).

Requires **[AWS CLI v2](https://aws.amazon.com/cli/)** (`aws --version`). Public data: **`--no-sign-request`** (no AWS account).

Under that prefix, each **stereo pair** is a folder (for example `ESP_045994_1985_ESP_046060_1985/`). List them:

```powershell
aws s3 ls s3://astrogeo-ard/mars/mro/hirise/controlled/dtm/ --no-sign-request --region us-west-2
```

### Windows (PowerShell)

```powershell
cd C:\Users\...\MarsRouteZero
# Default: one Jezero central stereo folder (smaller download)
.\scripts\download_jezero_dtm.ps1
# Another product (e.g. eastern delta pair from PRD)
.\scripts\download_jezero_dtm.ps1 -Prefix "ESP_048842_1985_ESP_048908_1985"
# Entire controlled DTM tree — very large, long runtime
.\scripts\download_jezero_dtm.ps1 -SyncAll
```

### Linux / macOS / Git Bash / WSL

```bash
cd /path/to/MarsRouteZero
chmod +x scripts/download_jezero_dtm.sh
bash scripts/download_jezero_dtm.sh
bash scripts/download_jezero_dtm.sh ./data/dtm/aws_sync "ESP_048842_1985_ESP_048908_1985"
SYNC_ALL=1 bash scripts/download_jezero_dtm.sh   # entire catalog — huge
```

After sync, merge/crop with GDAL per `mars-route-zero-prd.md` section 4.1 and `preprocess_terrain.py` if you need a lon/lat window. For a **single downloaded stereo folder** you can skip merge and build the app grid directly:

```powershell
python -m pip install rasterio
python scripts/hirise_geotiff_to_elevation_bin.py --input data/dtm/aws_sync/DTEEC_045994_1985_046060_1985_U01.tif
```

That writes `data/costmap/jezero_elevation.bin` and `data/dtm/jezero_meta.json` (including `meters_per_pixel`). **Restart the API** so it reloads the binary; if `MARS_METERS_PER_PIXEL` is unset, the backend reads `meters_per_pixel` from `jezero_meta.json`.

Full GDAL workflow (optional): `gdal_merge.py` then `python scripts/preprocess_terrain.py --mosaic ...` when you have multiple tiles.

## 3. Other scripts

| Script | Purpose |
|--------|---------|
| `hirise_geotiff_to_elevation_bin.py` | Resample one HiRISE DTM COG to `jezero_elevation.bin` (rasterio) |
| `preprocess_terrain.py` | Crop merged GeoTIFF with GDAL (`gdal_translate`) |
| `export_geotransform.py` | Print GeoTIFF geotransform JSON (needs Python GDAL) |
