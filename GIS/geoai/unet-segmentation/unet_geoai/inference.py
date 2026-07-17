from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Callable

import cv2
import geopandas as gpd
import numpy as np
import rasterio
import torch
from PIL import Image
from rasterio.features import shapes
from shapely.geometry import shape

from .config import CLASS_COLORS, CLASS_NAMES, OUTPUT_DIR
from .data import colorize_mask
from .models import build_model


def load_checkpoint(path: Path, device: torch.device | None = None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(path, map_location=device, weights_only=False)
    model = build_model(
        state["model_name"],
        num_classes=int(state["num_classes"]),
        base_channels=int(state["base_channels"]),
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, state, device


def _normalize_image(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)
    valid = np.isfinite(array)
    if valid.any():
        low, high = np.percentile(array[valid], [2, 98])
        array = np.clip((array - low) / max(high - low, 1e-6), 0, 1)
    else:
        array[:] = 0
    return array.astype(np.float32, copy=False)


def sliding_window_predict(model, image: np.ndarray, device: torch.device, tile_size: int = 256,
                           overlap: int = 64, progress: Callable[[int, str], None] | None = None) -> tuple[np.ndarray, np.ndarray]:
    channels, height, width = image.shape
    stride = max(1, tile_size - overlap)
    rows = list(range(0, max(height - tile_size, 0) + 1, stride))
    cols = list(range(0, max(width - tile_size, 0) + 1, stride))
    if not rows or rows[-1] != max(height - tile_size, 0):
        rows.append(max(height - tile_size, 0))
    if not cols or cols[-1] != max(width - tile_size, 0):
        cols.append(max(width - tile_size, 0))
    sample = torch.from_numpy(image[:, :min(tile_size, height), :min(tile_size, width)]).unsqueeze(0).to(device)
    sample = torch.nn.functional.interpolate(sample, size=(tile_size, tile_size), mode="bilinear", align_corners=False)
    with torch.no_grad():
        num_classes = model(sample).shape[1]
    probabilities = np.zeros((num_classes, height, width), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    window_1d = np.hanning(tile_size) if tile_size > 2 else np.ones(tile_size)
    blend = np.maximum(np.outer(window_1d, window_1d), 0.05).astype(np.float32)
    total = len(rows) * len(cols)
    completed = 0
    for row in rows:
        for col in cols:
            tile = image[:, row:min(row + tile_size, height), col:min(col + tile_size, width)]
            original_shape = tile.shape[-2:]
            tensor = torch.from_numpy(tile).unsqueeze(0).to(device)
            if original_shape != (tile_size, tile_size):
                tensor = torch.nn.functional.interpolate(tensor, size=(tile_size, tile_size), mode="bilinear", align_corners=False)
            with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(tensor)
                probs = torch.softmax(logits, 1)[0]
            if original_shape != (tile_size, tile_size):
                probs = torch.nn.functional.interpolate(probs.unsqueeze(0), size=original_shape, mode="bilinear", align_corners=False)[0]
            probs_np = probs.float().cpu().numpy()
            local_weight = blend[:original_shape[0], :original_shape[1]]
            probabilities[:, row:row + original_shape[0], col:col + original_shape[1]] += probs_np * local_weight
            weights[row:row + original_shape[0], col:col + original_shape[1]] += local_weight
            completed += 1
            if progress:
                progress(int(completed / total * 100), f"推理瓦片 {completed}/{total}")
    probabilities /= np.maximum(weights[None, ...], 1e-6)
    # RasterIO 读取整数影像时，归一化中的 float64 百分位数可能让结果隐式升级；
    # 在进入 CUDA 前强制回到模型权重使用的 float32，避免 DoubleTensor/FloatTensor 冲突。
    probabilities = probabilities.astype(np.float32, copy=False)
    return probabilities.argmax(0).astype(np.uint8), probabilities


def predict_geotiff(input_path: Path, checkpoint_path: Path, output_dir: Path | None = None,
                    tile_size: int = 256, overlap: int = 64,
                    progress: Callable[[int, str], None] | None = None) -> dict:
    output_dir = Path(output_dir or OUTPUT_DIR / f"prediction_{input_path.stem}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model, state, device = load_checkpoint(checkpoint_path)
    with rasterio.open(input_path) as source:
        indexes = list(range(1, min(source.count, 3) + 1))
        image = source.read(indexes)
        if image.shape[0] == 1:
            image = np.repeat(image, 3, axis=0)
        profile = source.profile.copy()
        valid_mask = source.read_masks(indexes[0]) > 0
        crs = source.crs
        transform = source.transform
    image = _normalize_image(image[:3])
    mask, probabilities = sliding_window_predict(model, image, device, tile_size, overlap, progress)
    mask[~valid_mask] = 255
    probabilities[:, ~valid_mask] = np.nan
    mask_path = output_dir / "class_mask.tif"
    profile.update(count=1, dtype="uint8", nodata=255, compress="deflate")
    with rasterio.open(mask_path, "w", **profile) as sink:
        sink.write(mask, 1)
        sink.write_colormap(1, {class_id: tuple(color) + (255,) for class_id, color in CLASS_COLORS.items()})
    probability_path = output_dir / "probabilities.tif"
    probability_profile = profile.copy()
    probability_profile.update(count=probabilities.shape[0], dtype="float32", nodata=np.nan)
    with rasterio.open(probability_path, "w", **probability_profile) as sink:
        sink.write(probabilities.astype(np.float32))
    preview_path = output_dir / "preview.png"
    Image.fromarray(colorize_mask(mask)).save(preview_path)
    vector_paths = vectorize_mask(mask_path, output_dir)
    result = {
        "input": str(input_path), "checkpoint": str(checkpoint_path), "model": state["model_name"],
        "mask": str(mask_path), "probabilities": str(probability_path), "preview": str(preview_path),
        "vectors": vector_paths, "crs": str(crs), "transform": list(transform)[:6],
    }
    (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def vectorize_mask(mask_path: Path, output_dir: Path | None = None, min_pixels: int = 12) -> dict:
    output_dir = Path(output_dir or mask_path.parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(mask_path) as source:
        mask = source.read(1)
        transform = source.transform
        crs = source.crs
    records = []
    for geometry, value in shapes(mask, mask=mask != 0, transform=transform):
        class_id = int(value)
        if class_id <= 0 or class_id >= len(CLASS_NAMES):
            continue
        geom = shape(geometry)
        if geom.is_empty:
            continue
        records.append({"class_id": class_id, "class_name": CLASS_NAMES[class_id], "geometry": geom})
    gdf = gpd.GeoDataFrame(records, columns=["class_id", "class_name", "geometry"], geometry="geometry", crs=crs)
    if not gdf.empty:
        pixel_area = abs(transform.a * transform.e)
        gdf = gdf[gdf.geometry.map(lambda geometry: geometry.area) >= pixel_area * min_pixels].copy()
    geojson_path = output_dir / "classes.geojson"
    gpkg_path = output_dir / "classes.gpkg"
    shp_dir = output_dir / "shapefiles"
    shp_dir.mkdir(exist_ok=True)
    # 即使没有任何目标，也写出合法的空 GeoJSON/GPKG，方便前端统一提供成果下载。
    gdf.to_file(geojson_path, driver="GeoJSON")
    gdf.to_file(gpkg_path, driver="GPKG", layer="landcover")
    if not gdf.empty:
        for class_id, group in gdf.groupby("class_id"):
            group.to_file(shp_dir / f"{CLASS_NAMES[int(class_id)]}.shp", driver="ESRI Shapefile", encoding="utf-8")
    archive = shutil.make_archive(str(output_dir / "shapefiles"), "zip", root_dir=shp_dir)
    return {"geojson": str(geojson_path), "gpkg": str(gpkg_path) if gpkg_path.exists() else None, "shapefiles_zip": archive}
