"""Baoji building extraction and regularization pipeline.

The script intentionally imports heavy GIS/deep-learning packages inside
functions so `--check-env` can run in a bare Python environment.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.request import Request, urlopen


PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"
LOCAL_TIF = DATA_DIR / "I48E006018.tif"
DEFAULT_SOURCE_TIF = LOCAL_TIF
EXISTING_SAM_MODEL_DIR = REPO_ROOT / "sam" / "models"

os.environ.setdefault("HF_HOME", str(MODEL_DIR / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(MODEL_DIR / "huggingface" / "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(MODEL_DIR / "huggingface" / "hub"))
os.environ.setdefault("TORCH_HOME", str(MODEL_DIR / "torch"))

SAM_MODELS = {
    "sam_vit_b": {
        "model_type": "vit_b",
        "filename": "sam_vit_b_01ec64.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "expected_bytes": 393891747,
    },
    "sam_vit_l": {
        "model_type": "vit_l",
        "filename": "sam_vit_l_0b3195.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "expected_bytes": 1249599818,
    },
    "sam_vit_h": {
        "model_type": "vit_h",
        "filename": "sam_vit_h_4b8939.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "expected_bytes": 2566786440,
    },
}

GEOAI_BUILDING_MODEL = {
    "repo_id": "giswqs/geoai",
    "filename": "building_footprints_usa.pth",
}

SAM_PARAM_GRID = [
    {"points_per_side": 16, "pred_iou_thresh": 0.86, "stability_score_thresh": 0.92},
    {"points_per_side": 24, "pred_iou_thresh": 0.88, "stability_score_thresh": 0.95},
]

REQUIRED_PACKAGES = [
    "numpy",
    "rasterio",
    "geopandas",
    "shapely",
    "pyproj",
    "torch",
    "samgeo",
]
OPTIONAL_PACKAGES = ["geoai", "matplotlib", "skimage", "scipy"]


@dataclass
class RunResult:
    method: str
    model: str
    params: Dict[str, Any]
    raw_path: Optional[Path]
    regularized_path: Optional[Path]
    gpkg_path: Optional[Path]
    count: int
    total_area_m2: float
    mean_area_m2: float
    elapsed_seconds: float
    status: str
    message: str = ""


def log(message: str) -> None:
    print(f"[building-extraction] {message}", flush=True)


def parse_models(value: str) -> List[str]:
    models = [item.strip() for item in value.split(",") if item.strip()]
    valid = {"geoai", *SAM_MODELS.keys()}
    unknown = [m for m in models if m not in valid]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown model(s): {', '.join(unknown)}")
    return models


def check_env() -> int:
    missing_required = []
    for name in REQUIRED_PACKAGES:
        try:
            module = __import__(name)
            version = getattr(module, "__version__", "ok")
            log(f"{name}: {version}")
        except Exception as exc:
            missing_required.append(name)
            log(f"{name}: missing ({type(exc).__name__}: {exc})")

    for name in OPTIONAL_PACKAGES:
        try:
            module = __import__(name)
            version = getattr(module, "__version__", "ok")
            log(f"{name}: {version}")
        except Exception as exc:
            log(f"{name}: optional missing ({type(exc).__name__}: {exc})")

    if missing_required:
        log("Missing required packages. Install with: pip install -r requirements.txt")
        return 2
    return 0


def prepare_dirs() -> None:
    for path in (DATA_DIR, MODEL_DIR, OUTPUT_DIR, OUTPUT_DIR / "compare", OUTPUT_DIR / "preview"):
        path.mkdir(parents=True, exist_ok=True)


def create_synthetic_image(path: Path, size: int = 768) -> Path:
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    path.parent.mkdir(parents=True, exist_ok=True)
    y, x = np.mgrid[0:size, 0:size]
    image = np.zeros((3, size, size), dtype=np.uint8)
    image[0] = np.clip(95 + x * 70 / size + y * 20 / size, 0, 255).astype(np.uint8)
    image[1] = np.clip(105 + y * 60 / size, 0, 255).astype(np.uint8)
    image[2] = np.clip(115 + (x + y) * 35 / size, 0, 255).astype(np.uint8)

    buildings = [
        (110, 120, 230, 245),
        (310, 95, 445, 210),
        (500, 150, 635, 300),
        (160, 405, 290, 555),
        (430, 430, 620, 610),
    ]
    for left, top, right, bottom in buildings:
        image[:, top:bottom, left:right] = np.array([210, 205, 195], dtype=np.uint8)[:, None, None]
        image[:, top : top + 4, left:right] = 70
        image[:, bottom - 4 : bottom, left:right] = 70
        image[:, top:bottom, left : left + 4] = 70
        image[:, top:bottom, right - 4 : right] = 70

    profile = {
        "driver": "GTiff",
        "height": size,
        "width": size,
        "count": 3,
        "dtype": "uint8",
        "crs": "EPSG:3857",
        "transform": from_origin(0, float(size), 0.3, 0.3),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(image)
    log(f"Created synthetic test image: {path}")
    return path


def prepare_data(source_tif: Path, copy_data: bool, create_synthetic_if_missing: bool) -> Path:
    prepare_dirs()
    if LOCAL_TIF.exists() and not copy_data:
        return LOCAL_TIF

    if source_tif.exists():
        if source_tif.resolve() == LOCAL_TIF.resolve():
            return LOCAL_TIF
        if LOCAL_TIF.exists() and LOCAL_TIF.stat().st_size == source_tif.stat().st_size:
            log(f"Data already copied: {LOCAL_TIF}")
        else:
            log(f"Copying image to project data: {LOCAL_TIF}")
            shutil.copy2(source_tif, LOCAL_TIF)
    elif create_synthetic_if_missing:
        log(f"Source image not found: {source_tif}. Creating a project-local synthetic test image.")
        create_synthetic_image(LOCAL_TIF)
    else:
        raise FileNotFoundError(f"Source image not found: {source_tif}")

    if not LOCAL_TIF.exists():
        raise FileNotFoundError(f"Project image not found: {LOCAL_TIF}. Use --copy-data.")
    return LOCAL_TIF


def valid_model_file(path: Path, expected_bytes: int = 0) -> bool:
    if not path.exists() or not path.is_file():
        return False
    if expected_bytes <= 0:
        return path.stat().st_size > 0
    return path.stat().st_size >= expected_bytes * 0.95


def download_url(url: str, dst: Path, expected_bytes: int = 0, retries: int = 5) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_suffix(dst.suffix + ".part")
    if dst.exists() and not valid_model_file(dst, expected_bytes) and not part.exists():
        dst.replace(part)

    last_error: Optional[BaseException] = None
    for attempt in range(1, retries + 1):
        try:
            resume_at = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={resume_at}-"} if resume_at else {}
            req = Request(url, headers=headers)
            with urlopen(req, timeout=120) as response:
                if resume_at and getattr(response, "status", 200) == 200:
                    resume_at = 0
                mode = "ab" if resume_at else "wb"
                downloaded = resume_at
                with part.open(mode) as fh:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        fh.write(chunk)
                        downloaded += len(chunk)
                        if downloaded // (256 * 1024 * 1024) != (downloaded - len(chunk)) // (256 * 1024 * 1024):
                            log(f"Downloaded {dst.name}: {downloaded / 1024 / 1024:.1f} MB")
            if valid_model_file(part, expected_bytes):
                part.replace(dst)
                return dst
            log(f"Download incomplete for {dst.name}; retrying from {part.stat().st_size / 1024 / 1024:.1f} MB")
        except Exception as exc:
            last_error = exc
            log(f"Download attempt {attempt}/{retries} failed for {dst.name}: {exc}")
            time.sleep(min(5 * attempt, 20))

    raise RuntimeError(f"Failed to download {dst.name}: {last_error}")


def ensure_sam_model(model_key: str, allow_download: bool) -> Path:
    info = SAM_MODELS[model_key]
    dst = MODEL_DIR / info["filename"]
    if valid_model_file(dst, info["expected_bytes"]):
        return dst

    src = EXISTING_SAM_MODEL_DIR / info["filename"]
    if valid_model_file(src, info["expected_bytes"]):
        log(f"Reusing existing SAM model: {src}")
        shutil.copy2(src, dst)
        return dst

    if not allow_download:
        raise FileNotFoundError(
            f"Missing {info['filename']}. Put it in {MODEL_DIR} or rerun without --no-download-models."
        )

    log(f"Downloading SAM model {model_key}: {info['url']}")
    download_url(info["url"], dst, info["expected_bytes"])
    if not valid_model_file(dst, info["expected_bytes"]):
        raise RuntimeError(f"Downloaded model is incomplete: {dst}")
    return dst


def ensure_geoai_model(allow_download: bool) -> Path:
    dst = MODEL_DIR / "geoai" / GEOAI_BUILDING_MODEL["filename"]
    if valid_model_file(dst):
        return dst

    legacy = MODEL_DIR / "geoai" / "building_footprints.pth"
    if valid_model_file(legacy):
        log(f"Reusing existing GeoAI model: {legacy}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy, dst)
        return dst

    if not allow_download:
        raise FileNotFoundError(
            f"Missing {GEOAI_BUILDING_MODEL['filename']}. Put it in {dst.parent} or rerun without --no-download-models."
        )

    log(
        "Downloading GeoAI building model "
        f"{GEOAI_BUILDING_MODEL['repo_id']}/{GEOAI_BUILDING_MODEL['filename']} to {dst.parent}"
    )
    from huggingface_hub import hf_hub_download

    downloaded = Path(
        hf_hub_download(
            repo_id=GEOAI_BUILDING_MODEL["repo_id"],
            filename=GEOAI_BUILDING_MODEL["filename"],
            local_dir=str(dst.parent),
        )
    )
    if downloaded.resolve() != dst.resolve():
        shutil.copy2(downloaded, dst)
    if not valid_model_file(dst):
        raise RuntimeError(f"Downloaded model is incomplete: {dst}")
    return dst


def ensure_models(models: Sequence[str], allow_download: bool) -> Dict[str, Path]:
    prepare_dirs()
    paths = {}
    for model in models:
        if model == "geoai":
            paths[model] = ensure_geoai_model(allow_download)
        elif model.startswith("sam_"):
            paths[model] = ensure_sam_model(model, allow_download)
    return paths


def make_test_crop(image_path: Path, tile_size: int) -> Path:
    import rasterio
    from rasterio.windows import Window

    crop_path = OUTPUT_DIR / "test_crop.tif"
    with rasterio.open(image_path) as src:
        width = min(tile_size, src.width)
        height = min(tile_size, src.height)
        col_off = max((src.width - width) // 2, 0)
        row_off = max((src.height - height) // 2, 0)
        window = Window(col_off, row_off, width, height)
        data = src.read(window=window)
        profile = src.profile.copy()
        profile.update(width=width, height=height, transform=src.window_transform(window))
        with rasterio.open(crop_path, "w", **profile) as dst:
            dst.write(data)
    log(f"Created test crop: {crop_path}")
    return crop_path


def empty_gdf_like(crs: Any = None):
    import geopandas as gpd

    return gpd.GeoDataFrame(
        columns=["method", "model", "params", "area_m2", "regularized", "geometry"],
        geometry="geometry",
        crs=crs,
    )


def add_area_m2(gdf: Any) -> Any:
    if len(gdf) == 0:
        gdf["area_m2"] = []
        return gdf
    out = gdf.copy()
    try:
        if out.crs is not None and out.crs.is_geographic:
            metric = out.to_crs(out.estimate_utm_crs())
            out["area_m2"] = metric.geometry.area.values
        else:
            out["area_m2"] = out.geometry.area
    except Exception:
        out["area_m2"] = out.geometry.area
    return out


def normalize_gdf(gdf: Any, method: str, model: str, params: Dict[str, Any], min_area: float) -> Any:
    if gdf is None:
        return empty_gdf_like()
    out = gdf.copy()
    if "geometry" not in out:
        return empty_gdf_like(getattr(out, "crs", None))
    out = out[out.geometry.notna()].copy()
    if len(out) > 0:
        out["geometry"] = out.geometry.make_valid()
        out = out[~out.geometry.is_empty].copy()
    out = add_area_m2(out)
    if "area_m2" in out and len(out) > 0:
        out = out[out["area_m2"] >= min_area].copy()
    out["method"] = method
    out["model"] = model
    out["params"] = json.dumps(params, ensure_ascii=False, sort_keys=True)
    out["regularized"] = False
    keep = ["method", "model", "params", "area_m2", "regularized", "geometry"]
    return out[keep].reset_index(drop=True)


def load_regularizer():
    path = REPO_ROOT / "building-regularize" / "backend" / "regularize.py"
    spec = importlib.util.spec_from_file_location("building_regularize_pipeline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load regularizer from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.RegularizeConfig, module.RegularizePipeline


def regularize_gdf(gdf: Any, min_area: float, simplify: float) -> Any:
    import geopandas as gpd

    if len(gdf) == 0:
        out = gdf.copy()
        out["regularized"] = []
        return out

    RegularizeConfig, RegularizePipeline = load_regularizer()
    original_crs = gdf.crs
    work = gdf.to_crs("EPSG:4326") if original_crs and not original_crs.is_geographic else gdf.copy()
    polygons = list(work.geometry)
    cfg = RegularizeConfig(
        min_area=min_area,
        dp_tolerance=simplify,
        angle_threshold=10.0,
        snap_angles=[0, 45, 90, 135],
        fix_topology=True,
    )
    regularized = RegularizePipeline(cfg).run(polygons)
    out = work.copy()
    out["geometry"] = regularized
    out["regularized"] = True
    out = gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:4326")
    if original_crs and str(original_crs) != "EPSG:4326":
        out = out.to_crs(original_crs)
    out = add_area_m2(out)
    return out


def save_outputs(gdf: Any, run_dir: Path, regularize: bool, min_area: float, simplify: float) -> Tuple[Path, Optional[Path], Optional[Path], Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / "raw.shp"
    gdf.to_file(raw_path, driver="ESRI Shapefile", encoding="utf-8")

    if not regularize:
        return raw_path, None, None, gdf

    reg = regularize_gdf(gdf, min_area=min_area, simplify=simplify)
    reg_path = run_dir / "regularized.shp"
    gpkg_path = run_dir / "regularized.gpkg"
    reg.to_file(reg_path, driver="ESRI Shapefile", encoding="utf-8")
    reg.to_file(gpkg_path, driver="GPKG")
    return raw_path, reg_path, gpkg_path, reg


def run_geoai(image_path: Path, model_path: Path, args: argparse.Namespace) -> RunResult:
    start = time.time()
    run_dir = OUTPUT_DIR / "geoai"
    params = {"tile_size": args.tile_size, "overlap": args.overlap, "min_area": args.min_area}
    try:
        import geopandas as gpd
        import rasterio
        from geoai.extract import BuildingFootprintExtractor

        with rasterio.open(image_path) as src:
            crs = src.crs

        extractor_kwargs = {"device": args.device, "model_path": str(model_path)}

        log(f"Running GeoAI BuildingFootprintExtractor with model: {model_path}")
        extractor = BuildingFootprintExtractor(**extractor_kwargs)
        gdf = extractor.process_raster(
            str(image_path),
            batch_size=args.batch_size,
            filter_edges=True,
            edge_buffer=max(8, args.overlap // 4),
        )
        if gdf is None:
            gdf = empty_gdf_like(crs)
        if gdf.crs is None:
            gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=crs)
        try:
            gdf = extractor.regularize_buildings(
                gdf,
                min_area=int(args.min_area),
                angle_threshold=15,
                rectangularity_threshold=0.7,
            )
        except Exception as exc:
            log(f"GeoAI internal regularization skipped: {exc}")

        gdf = normalize_gdf(gdf, "geoai", "geoai_building", params, args.min_area)
        raw_path, reg_path, gpkg_path, reg = save_outputs(gdf, run_dir, args.regularize, args.min_area, args.simplify)
        stats_gdf = reg if reg_path else gdf
        return build_result("geoai", "geoai_building", params, raw_path, reg_path, gpkg_path, stats_gdf, start, "ok")
    except Exception as exc:
        return build_result("geoai", "geoai_building", params, None, None, None, None, start, "failed", str(exc))


def combine_sam_masks(masks: Any, shape: Tuple[int, int]) -> Any:
    import numpy as np

    combined = np.zeros(shape, dtype=bool)
    if masks is None:
        return combined
    if isinstance(masks, list):
        for item in masks:
            if isinstance(item, dict) and "segmentation" in item:
                combined |= np.asarray(item["segmentation"], dtype=bool)
        return combined
    arr = np.asarray(masks)
    if arr.ndim == 4:
        arr = arr[:, 0, :, :]
    if arr.ndim == 3:
        combined = arr.astype(bool).any(axis=0)
    elif arr.ndim == 2:
        combined = arr.astype(bool)
    return combined


def filter_building_like(gdf: Any, min_area: float) -> Any:
    if len(gdf) == 0:
        return gdf
    out = gdf.copy()
    bounds = out.geometry.bounds
    widths = bounds["maxx"] - bounds["minx"]
    heights = bounds["maxy"] - bounds["miny"]
    bbox_area = (widths * heights).replace(0, 1)
    rectangularity = out.geometry.area / bbox_area
    aspect = widths.combine(heights, max) / widths.combine(heights, min).replace(0, 1)
    out = out[(out["area_m2"] >= min_area) & (rectangularity >= 0.25) & (aspect <= 8)].copy()
    return out.reset_index(drop=True)


def read_rgb_for_sam(image_path: Path) -> Any:
    import numpy as np
    import rasterio

    with rasterio.open(image_path) as src:
        indexes = list(range(1, min(src.count, 3) + 1))
        data = src.read(indexes)

    if data.shape[0] == 1:
        data = np.repeat(data, 3, axis=0)

    rgb = np.moveaxis(data[:3], 0, -1)
    if rgb.dtype == np.uint8:
        return rgb

    rgb = rgb.astype("float32")
    out = np.zeros_like(rgb, dtype=np.uint8)
    for band in range(rgb.shape[2]):
        values = rgb[:, :, band]
        low, high = np.nanpercentile(values, [2, 98])
        if high <= low:
            high = float(np.nanmax(values) or 1)
            low = float(np.nanmin(values))
        scaled = (values - low) * 255.0 / max(high - low, 1e-6)
        out[:, :, band] = np.clip(scaled, 0, 255).astype(np.uint8)
    return out


def run_sam(image_path: Path, model_key: str, checkpoint: Path, args: argparse.Namespace) -> List[RunResult]:
    results = []
    for params in SAM_PARAM_GRID:
        start = time.time()
        run_name = f"{model_key}_pps{params['points_per_side']}_iou{params['pred_iou_thresh']}"
        run_dir = OUTPUT_DIR / run_name
        all_params = {**params, "min_area": args.min_area}
        try:
            import gc
            import numpy as np
            import rasterio
            import torch
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

            sys.path.insert(0, str(REPO_ROOT / "sam"))
            from geoai_sam import MaskPostProcessor, MaskVectorizer

            with rasterio.open(image_path) as src:
                shape = (src.height, src.width)

            log(f"Running SAM auto segmentation: {model_key} {params}")
            sam = sam_model_registry[SAM_MODELS[model_key]["model_type"]](checkpoint=str(checkpoint))
            sam.to(device=args.device)
            generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=params["points_per_side"],
                pred_iou_thresh=params["pred_iou_thresh"],
                stability_score_thresh=params["stability_score_thresh"],
                min_mask_region_area=max(16, int(args.min_area)),
            )
            masks = generator.generate(read_rgb_for_sam(image_path))
            combined = combine_sam_masks(masks, shape)
            clean = MaskPostProcessor.default_pipeline(
                combined.astype(np.uint8),
                min_size=max(16, int(args.min_area)),
                fill_holes_flag=True,
                smooth_sigma=0.8,
                opening_radius=1,
                closing_radius=2,
            )
            vectorizer = MaskVectorizer()
            gdf = vectorizer.vectorize(clean.astype(np.uint8), reference_image=str(image_path), min_area=max(1, int(args.min_area)))
            gdf = normalize_gdf(gdf, "sam", model_key, all_params, args.min_area)
            gdf = filter_building_like(gdf, args.min_area)
            raw_path, reg_path, gpkg_path, reg = save_outputs(gdf, run_dir, args.regularize, args.min_area, args.simplify)
            stats_gdf = reg if reg_path else gdf
            results.append(build_result("sam", model_key, all_params, raw_path, reg_path, gpkg_path, stats_gdf, start, "ok"))
        except Exception as exc:
            results.append(build_result("sam", model_key, all_params, None, None, None, None, start, "failed", str(exc)))
        finally:
            if "generator" in locals():
                del generator
            if "sam" in locals():
                del sam
            if "torch" in sys.modules:
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            if "gc" in locals():
                gc.collect()
    return results


def build_result(
    method: str,
    model: str,
    params: Dict[str, Any],
    raw_path: Optional[Path],
    regularized_path: Optional[Path],
    gpkg_path: Optional[Path],
    gdf: Any,
    start: float,
    status: str,
    message: str = "",
) -> RunResult:
    count = 0
    total = 0.0
    mean = 0.0
    if gdf is not None and len(gdf) > 0:
        count = int(len(gdf))
        total = float(gdf["area_m2"].sum()) if "area_m2" in gdf else float(gdf.geometry.area.sum())
        mean = total / count if count else 0.0
    return RunResult(
        method=method,
        model=model,
        params=params,
        raw_path=raw_path,
        regularized_path=regularized_path,
        gpkg_path=gpkg_path,
        count=count,
        total_area_m2=round(total, 2),
        mean_area_m2=round(mean, 2),
        elapsed_seconds=round(time.time() - start, 2),
        status=status,
        message=message,
    )


def write_summary(results: Sequence[RunResult]) -> Path:
    path = OUTPUT_DIR / "compare" / "summary.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "method",
                "model",
                "params",
                "count",
                "total_area_m2",
                "mean_area_m2",
                "elapsed_seconds",
                "status",
                "message",
                "raw_path",
                "regularized_path",
                "gpkg_path",
            ],
        )
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "method": item.method,
                    "model": item.model,
                    "params": json.dumps(item.params, ensure_ascii=False, sort_keys=True),
                    "count": item.count,
                    "total_area_m2": item.total_area_m2,
                    "mean_area_m2": item.mean_area_m2,
                    "elapsed_seconds": item.elapsed_seconds,
                    "status": item.status,
                    "message": item.message,
                    "raw_path": str(item.raw_path or ""),
                    "regularized_path": str(item.regularized_path or ""),
                    "gpkg_path": str(item.gpkg_path or ""),
                }
            )
    log(f"Wrote summary: {path}")
    return path


def choose_best(results: Sequence[RunResult]) -> Optional[RunResult]:
    ok = [r for r in results if r.status == "ok" and r.count > 0 and r.regularized_path]
    if not ok:
        return None
    geoai = [r for r in ok if r.method == "geoai"]
    if geoai:
        return max(geoai, key=lambda r: r.count)
    return max(ok, key=lambda r: (r.count, r.total_area_m2))


def copy_best(best: Optional[RunResult]) -> Optional[Path]:
    if best is None or best.regularized_path is None:
        log("No non-empty regularized result available for best_regularized.shp")
        return None
    target_base = OUTPUT_DIR / "compare" / "best_regularized"
    for src in best.regularized_path.parent.glob(best.regularized_path.stem + ".*"):
        shutil.copy2(src, target_base.with_suffix(src.suffix))
    log(f"Best result copied from {best.regularized_path}")
    return target_base.with_suffix(".shp")


def make_preview(image_path: Path, result: RunResult) -> Optional[Path]:
    if result.regularized_path is None or not result.regularized_path.exists():
        return None
    try:
        import geopandas as gpd
        import matplotlib.pyplot as plt
        import numpy as np
        import rasterio
        from rasterio.plot import show

        gdf = gpd.read_file(result.regularized_path)
        if len(gdf) == 0:
            return None
        preview_path = OUTPUT_DIR / "preview" / f"{result.model}_{result.method}.png"
        with rasterio.open(image_path) as src:
            fig, ax = plt.subplots(figsize=(10, 10))
            show(src, ax=ax)
            if gdf.crs and src.crs and gdf.crs != src.crs:
                gdf = gdf.to_crs(src.crs)
            gdf.boundary.plot(ax=ax, edgecolor="red", linewidth=0.7)
            ax.set_axis_off()
            ax.set_title(f"{result.method} {result.model}: {result.count} buildings")
            fig.tight_layout()
            fig.savefig(preview_path, dpi=180)
            plt.close(fig)
        return preview_path
    except Exception as exc:
        log(f"Preview skipped for {result.model}: {exc}")
        return None


def validate_outputs(results: Sequence[RunResult]) -> None:
    import geopandas as gpd

    for item in results:
        for path in (item.raw_path, item.regularized_path):
            if path is None:
                continue
            gdf = gpd.read_file(path)
            invalid = 0 if len(gdf) == 0 else int((~gdf.geometry.is_valid).sum())
            log(f"Validated {path}: {len(gdf)} features, invalid geometries={invalid}, crs={gdf.crs}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract and regularize Baoji building footprints.")
    parser.add_argument("--source-tif", type=Path, default=DEFAULT_SOURCE_TIF)
    parser.add_argument("--copy-data", action="store_true", help="Copy source GeoTIFF into building-extraction/data.")
    parser.add_argument("--models", type=parse_models, default=parse_models("geoai,sam_vit_b,sam_vit_l,sam_vit_h"))
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-area", type=float, default=20.0)
    parser.add_argument("--simplify", type=float, default=0.5)
    add_bool_flag(parser, "regularize", True, "Run polygon regularization.")
    parser.add_argument("--test-crop", action=argparse.BooleanOptionalAction, default=True)
    add_bool_flag(parser, "download-models", True, "Download missing models into the project models directory.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--check-env", action="store_true", help="Only check Python dependencies.")
    add_bool_flag(parser, "validate", True, "Validate generated vector files.")
    return parser


def add_bool_flag(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    attr = name.replace("-", "_")
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(f"--{name}", action=argparse.BooleanOptionalAction, default=default, help=help_text)
        return
    parser.set_defaults(**{attr: default})
    parser.add_argument(f"--{name}", dest=attr, action="store_true", help=help_text)
    parser.add_argument(f"--no-{name}", dest=attr, action="store_false", help=f"Disable: {help_text}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.check_env:
        return check_env()

    if check_env() != 0:
        return 2

    prepare_dirs()
    image = prepare_data(args.source_tif, args.copy_data, create_synthetic_if_missing=args.test_crop)
    model_paths = ensure_models(args.models, args.download_models)
    run_image = make_test_crop(image, args.tile_size) if args.test_crop else image

    results: List[RunResult] = []
    if "geoai" in args.models:
        result = run_geoai(run_image, model_paths["geoai"], args)
        log(f"GeoAI result: {result.status}, count={result.count}, message={result.message}")
        results.append(result)

    for model in args.models:
        if model.startswith("sam_"):
            for result in run_sam(run_image, model, model_paths[model], args):
                log(f"SAM result: {result.model}, {result.status}, count={result.count}, message={result.message}")
                results.append(result)

    write_summary(results)
    best = choose_best(results)
    copy_best(best)
    for item in results:
        make_preview(run_image, item)
    if args.validate:
        validate_outputs(results)

    failed = [r for r in results if r.status != "ok"]
    if failed and len(failed) == len(results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
