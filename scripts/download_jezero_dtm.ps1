# Download NASA/USGS HiRISE controlled DTMs from AWS Open Data Registry.
# ARN: arn:aws:s3:::astrogeo-ard/mars/mro/hirise/controlled/dtm
# Region: us-west-2 (required).
# Run from repo root:
#   .\scripts\download_jezero_dtm.ps1
#   .\scripts\download_jezero_dtm.ps1 -Prefix "ESP_048842_1985_ESP_048908_1985"
#   .\scripts\download_jezero_dtm.ps1 -SyncAll   # entire catalog - very large
# Requires: AWS CLI v2, --no-sign-request (no AWS account).

param(
    [string]$Dest = ".\data\dtm\aws_sync",
    [string]$Region = "us-west-2",
    # Stereo-pair folder under .../controlled/dtm/ (list via aws s3 ls below).
    # Default matches Jezero central crater floor stereo product naming (PRD).
    [string]$Prefix = "ESP_045994_1985_ESP_046060_1985",
    [switch]$SyncAll
)

$ErrorActionPreference = "Stop"

$Base = "s3://astrogeo-ard/mars/mro/hirise/controlled/dtm"

New-Item -ItemType Directory -Force -Path $Dest | Out-Null

if ($SyncAll) {
    $Source = "${Base}/"
    Write-Host "WARNING: Syncing entire controlled DTM catalog - many GB and long runtime."
} else {
    $key = if ($Prefix) { $Prefix.Trim().Trim("/") } else { "" }
    if (-not $key) {
        Write-Host "Specify -Prefix stereo folder or -SyncAll. List folders:"
        Write-Host "  aws s3 ls ${Base}/ --no-sign-request --region $Region"
        exit 1
    }
    $Source = "${Base}/${key}/"
}

Write-Host "Syncing $Source -> $Dest"
Write-Host "Region: $Region"
Write-Host "(This may take a while depending on prefix size.)"

aws s3 sync $Source $Dest --no-sign-request --region $Region

Write-Host "Done. Next: gdal_merge / gdal_translate per mars-route-zero-prd.md section 4.1 and scripts/preprocess_terrain.py"
