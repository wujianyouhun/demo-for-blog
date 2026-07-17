from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import rasterio
import torch
from PIL import Image
from rasterio.features import shapes
from shapely.geometry import shape
from torch.utils.data import Dataset

SUPPORTED = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def read_image(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(path) as source:
            array = source.read(list(range(1, min(3, source.count) + 1)))
        if array.shape[0] == 1:
            array = np.repeat(array, 3, axis=0)
        image = np.moveaxis(array[:3], 0, -1).astype(np.float32)
        low, high = np.percentile(image[np.isfinite(image)], [2, 98])
        return np.clip((image - low) / max(high - low, 1e-6) * 255, 0, 255).astype(np.uint8)
    return np.asarray(Image.open(path).convert("RGB"))


def read_mask(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(path) as source:
            return source.read(1)
    return np.asarray(Image.open(path).convert("L"))


class SegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, size=256, augment=False, indices=None):
        images = {path.stem: path for path in Path(image_dir).iterdir() if path.suffix.lower() in SUPPORTED}
        masks = {path.stem: path for path in Path(mask_dir).iterdir() if path.suffix.lower() in SUPPORTED}
        if set(images) != set(masks):
            raise ValueError(f"影像标签未严格配对，缺标签={sorted(set(images)-set(masks))}，缺影像={sorted(set(masks)-set(images))}")
        pairs = [(images[key], masks[key]) for key in sorted(images)]
        if indices is not None:
            pairs = [pairs[index] for index in indices]
        if not pairs:
            raise ValueError("数据集为空")
        self.pairs = pairs
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image = cv2.resize(read_image(image_path), (self.size, self.size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize((read_mask(mask_path) > 0).astype(np.uint8), (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        if self.augment:
            if random.random() < .5:
                image, mask = np.fliplr(image).copy(), np.fliplr(mask).copy()
            if random.random() < .5:
                image, mask = np.flipud(image).copy(), np.flipud(mask).copy()
            turns = random.randint(0, 3)
            image, mask = np.rot90(image, turns).copy(), np.rot90(mask, turns).copy()
        return torch.from_numpy(np.moveaxis(image.astype(np.float32) / 255, -1, 0)), torch.from_numpy(mask.astype(np.int64))


def split_indices(length: int, val_ratio=.25, seed=42):
    if length < 2:
        raise ValueError("至少需要两组样本")
    generator = np.random.default_rng(seed)
    indices = generator.permutation(length).tolist()
    val_count = max(1, int(round(length * val_ratio)))
    return indices[val_count:], indices[:val_count]


def calculate_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    pred = prediction.astype(bool)
    truth = target.astype(bool)
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    iou = tp / max(tp + fp + fn, 1)
    dice = 2 * tp / max(2 * tp + fp + fn, 1)
    return {"iou": iou, "dice": dice, "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)}


def mask_to_vectors(mask: np.ndarray, transform, crs, output_path: Path, min_area_pixels=20) -> dict:
    records = []
    for geometry, value in shapes(mask.astype(np.uint8), mask=mask.astype(bool), transform=transform):
        if int(value) != 1:
            continue
        geom = shape(geometry)
        if geom.area >= abs(transform.a * transform.e) * min_area_pixels:
            records.append({"class_name": "building", "geometry": geom})
    gdf = gpd.GeoDataFrame(records, columns=["class_name", "geometry"], geometry="geometry", crs=crs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".gpkg":
        gdf.to_file(output_path, driver="GPKG", layer="prediction")
    elif not gdf.empty:
        gdf.to_file(output_path, driver="ESRI Shapefile", encoding="utf-8")
    return {"path": str(output_path), "features": len(gdf)}
