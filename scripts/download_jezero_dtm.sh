#!/usr/bin/env bash
# Download NASA/USGS HiRISE controlled DTMs (AWS Open Data).
# s3://astrogeo-ard/mars/mro/hirise/controlled/dtm/  region: us-west-2
# Usage (from repo root):
#   bash scripts/download_jezero_dtm.sh
#   bash scripts/download_jezero_dtm.sh ./data/dtm/aws_sync ESP_048842_1985_ESP_048908_1985
#   SYNC_ALL=1 bash scripts/download_jezero_dtm.sh   # entire catalog - huge
# Requires: aws CLI, --no-sign-request

set -euo pipefail

BASE="s3://astrogeo-ard/mars/mro/hirise/controlled/dtm"
REGION="${REGION:-us-west-2}"
DEST="${1:-./data/dtm/aws_sync}"
# Default: Jezero central stereo folder (PRD Jezero_C DTEEC pair naming)
DEFAULT_PREFIX="ESP_045994_1985_ESP_046060_1985"
PREFIX="${2:-${PREFIX:-$DEFAULT_PREFIX}}"

mkdir -p "${DEST}"

if [[ "${SYNC_ALL:-0}" == "1" ]]; then
  echo "WARNING: Syncing entire controlled DTM catalog - many GB."
  SOURCE="${BASE}/"
else
  PREFIX="${PREFIX#/}"
  PREFIX="${PREFIX%/}"
  if [[ -z "${PREFIX}" ]]; then
    echo "Set PREFIX to a stereo folder or use SYNC_ALL=1. List:"
    echo "  aws s3 ls ${BASE}/ --no-sign-request --region ${REGION}"
    exit 1
  fi
  SOURCE="${BASE}/${PREFIX}/"
fi

echo "Syncing ${SOURCE} -> ${DEST}"
echo "Region: ${REGION}"
aws s3 sync "${SOURCE}" "${DEST}" --no-sign-request --region "${REGION}"
echo "Done. Next: gdal_merge / gdal_translate per mars-route-zero-prd.md section 4.1"
