# Baoji Building Extraction

This project extracts building footprints from the Baoji high-resolution GeoTIFF,
regularizes the building polygons, and compares GeoAI and SAM extraction results.

## Setup

```powershell
conda create -n baoji-buildings python=3.11 -y
conda activate baoji-buildings
pip install -r requirements.txt
```

The base Python environment in this workspace does not currently contain the GIS
and deep learning dependencies, so use a dedicated environment before running
full inference.

## Quick Test

```powershell
python extract_buildings.py --check-env
python extract_buildings.py --copy-data --test-crop --models sam_vit_b --device cpu
python extract_buildings.py --test-crop --models geoai,sam_vit_b,sam_vit_l
```

## Full Run

```powershell
python extract_buildings.py --models geoai,sam_vit_b,sam_vit_l,sam_vit_h --device cuda
```

Outputs are written under `outputs/`:

- `outputs/<run>/raw.shp`
- `outputs/<run>/regularized.shp`
- `outputs/<run>/regularized.gpkg`
- `outputs/compare/summary.csv`
- `outputs/compare/best_regularized.shp`
- `outputs/preview/*.png`

