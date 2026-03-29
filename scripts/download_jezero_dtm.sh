#!/usr/bin/env bash
# Download Jezero HiRISE DTM COGs from AWS Open Data (PRD §4.1).
# Requires: aws CLI configured or anonymous access (--no-sign-request).
set -euo pipefail
DEST="${1:-./data/dtm/aws_sync}"
mkdir -p "$DEST"
echo "Syncing s3://nasa-usgs-mars-hirise-dtms/jezero/ -> $DEST"
aws s3 sync "s3://nasa-usgs-mars-hirise-dtms/jezero/" "$DEST" --no-sign-request
echo "Done. Next: gdal_merge / gdal_translate per mars-route-zero-prd.md §4.1"
