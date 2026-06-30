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
python extract_buildings.py --test-crop --models geoai,sam_vit_b,sam_vit_l
```

Missing GeoAI and SAM weights are downloaded into the project `models/`
directory by default, then loaded from there on later runs. Use
`--no-download-models` for offline runs that should fail fast if a model is
missing.

If `data/I48E006018.tif` is missing and `--test-crop` is enabled, the script
creates a small synthetic GeoTIFF under `data/` so the pipeline can be smoke
tested without an external source file. For real imagery, pass
`--source-tif <path> --copy-data` once to copy it into the project.

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

