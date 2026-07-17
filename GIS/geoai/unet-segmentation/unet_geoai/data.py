from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import rasterio
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .config import CLASS_COLORS, CLASS_NAMES, DATA_DIR, DEFAULT_SEED, PROFILES, REAL_BUILDING_DIR

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _texture(rng: np.random.Generator, size: int, color: tuple[int, int, int], strength: int = 22) -> np.ndarray:
    base = np.empty((size, size, 3), dtype=np.float32)
    base[:] = color
    coarse = rng.normal(0, strength, (max(2, size // 16), max(2, size // 16), 3)).astype(np.float32)
    coarse = cv2.resize(coarse, (size, size), interpolation=cv2.INTER_CUBIC)
    fine = rng.normal(0, strength * 0.35, base.shape)
    return np.clip(base + coarse + fine, 0, 255).astype(np.uint8)


def generate_landcover_sample(index: int, size: int, seed: int = DEFAULT_SEED) -> tuple[np.ndarray, np.ndarray]:
    """生成包含几何、纹理、阴影和遮挡的六类遥感风格样本。"""
    rng = np.random.default_rng(seed * 100_003 + index)
    py_rng = random.Random(seed * 10_007 + index)
    background = py_rng.choice([(88, 96, 102), (118, 112, 98), (105, 110, 108)])
    image = _texture(rng, size, background, 28)
    mask = np.zeros((size, size), dtype=np.uint8)

    # 裸地和植被采用不规则多边形，避免整图类别可由平均颜色直接识别。
    for class_id, color, count in ((5, (151, 126, 85), 2), (4, (63, 126, 67), 3)):
        for _ in range(count):
            center = np.array([py_rng.randrange(size), py_rng.randrange(size)])
            points = []
            for angle in np.linspace(0, 2 * math.pi, py_rng.randint(6, 10), endpoint=False):
                radius = py_rng.uniform(size * 0.10, size * 0.32)
                point = center + [math.cos(angle) * radius, math.sin(angle) * radius]
                points.append(np.clip(point, 0, size - 1).astype(np.int32))
            polygon = np.asarray(points, dtype=np.int32)
            cv2.fillPoly(mask, [polygon], class_id)
            layer = _texture(rng, size, color, 25)
            region = np.zeros_like(mask)
            cv2.fillPoly(region, [polygon], 1)
            image[region == 1] = layer[region == 1]

    # 水体包含弯曲轮廓和轻微高光。
    water_points = []
    horizontal = py_rng.random() < 0.5
    for step in range(7):
        axis = int(step * (size - 1) / 6)
        offset = int(size * (0.42 + 0.15 * math.sin(step * 1.4 + py_rng.random())))
        water_points.append((axis, offset) if horizontal else (offset, axis))
    water_line = np.asarray(water_points, np.int32)
    water_width = py_rng.randint(max(5, size // 14), max(8, size // 7))
    cv2.polylines(mask, [water_line], False, 3, thickness=water_width)
    cv2.polylines(image, [water_line], False, (48, 96, 145), thickness=water_width)
    cv2.polylines(image, [water_line], False, (70, 121, 165), thickness=max(1, water_width // 4))

    # 道路采用多方向折线，并让部分道路穿过其他类别。
    for _ in range(py_rng.randint(1, 3)):
        points = np.asarray([(py_rng.randrange(size), py_rng.randrange(size)) for _ in range(py_rng.randint(2, 4))], np.int32)
        width = py_rng.randint(max(2, size // 48), max(4, size // 24))
        cv2.polylines(mask, [points], False, 2, thickness=width)
        road_color = py_rng.randint(130, 175)
        cv2.polylines(image, [points], False, (road_color, road_color, road_color), thickness=width)
        cv2.polylines(image, [points], False, (205, 198, 160), thickness=1)

    # 建筑使用旋转矩形、屋顶纹理和偏移阴影。
    for _ in range(py_rng.randint(5, 13)):
        width = py_rng.randint(max(6, size // 18), max(9, size // 7))
        height = py_rng.randint(max(6, size // 20), max(9, size // 8))
        center = (py_rng.randint(width, size - width), py_rng.randint(height, size - height))
        angle = py_rng.choice([0, 0, 0, 15, 30, 45, 75, 90, 120, 150])
        box = cv2.boxPoints((center, (width, height), angle)).astype(np.int32)
        shadow = box + np.asarray([max(1, size // 64), max(1, size // 64)])
        cv2.fillPoly(image, [shadow], (55, 60, 66))
        roof = py_rng.randint(160, 226)
        cv2.fillPoly(image, [box], (roof, max(120, roof - py_rng.randint(0, 25)), max(115, roof - py_rng.randint(0, 30))))
        cv2.polylines(image, [box], True, (235, 235, 228), 1)
        cv2.fillPoly(mask, [box], 1)

    # 光照、模糊和传感器噪声使类别边界更接近影像。
    gain = py_rng.uniform(0.85, 1.15)
    bias = py_rng.uniform(-12, 12)
    image = np.clip(image.astype(np.float32) * gain + bias + rng.normal(0, 4, image.shape), 0, 255).astype(np.uint8)
    if py_rng.random() < 0.35:
        image = cv2.GaussianBlur(image, (3, 3), 0)
    return image, mask


def generate_dataset(profile: str = "quick", output_dir: Path | None = None, seed: int = DEFAULT_SEED,
                     progress: Callable[[int, str], None] | None = None) -> dict:
    if profile not in PROFILES:
        raise ValueError(f"未知配置: {profile}")
    config = PROFILES[profile]
    root = Path(output_dir or DATA_DIR / "synthetic" / profile)
    manifest = {"profile": profile, "seed": seed, "classes": CLASS_NAMES, "samples": [], "config": config}
    split_names: list[str] = []
    for split, count in config["splits"].items():
        split_names.extend([split] * count)
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "masks").mkdir(parents=True, exist_ok=True)
    for index, split in enumerate(split_names):
        image, mask = generate_landcover_sample(index, config["image_size"], seed)
        name = f"sample_{index:05d}.png"
        Image.fromarray(image).save(root / split / "images" / name)
        Image.fromarray(mask).save(root / split / "masks" / name)
        counts = np.bincount(mask.reshape(-1), minlength=len(CLASS_NAMES)).tolist()
        manifest["samples"].append({"id": name[:-4], "split": split, "image": f"{split}/images/{name}", "mask": f"{split}/masks/{name}", "class_pixels": counts})
        if progress:
            progress(int((index + 1) / len(split_names) * 100), f"生成 {index + 1}/{len(split_names)}")
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"root": str(root), "manifest": str(root / "manifest.json"), "samples": len(split_names), "splits": config["splits"]}


def _read_image(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(path) as dataset:
            array = dataset.read(list(range(1, min(dataset.count, 3) + 1)))
        if array.shape[0] == 1:
            array = np.repeat(array, 3, axis=0)
        image = np.moveaxis(array[:3], 0, -1)
        low, high = np.percentile(image[np.isfinite(image)], [2, 98]) if np.isfinite(image).any() else (0, 1)
        image = np.clip((image - low) / max(high - low, 1e-6) * 255, 0, 255).astype(np.uint8)
        return image
    return np.asarray(Image.open(path).convert("RGB"))


def _read_mask(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(path) as dataset:
            return dataset.read(1).astype(np.uint8)
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


class SegmentationDataset(Dataset):
    def __init__(self, root: Path, split: str, image_size: int, augment: bool = False, binary: bool = False):
        self.image_dir = Path(root) / split / "images"
        self.mask_dir = Path(root) / split / "masks"
        self.image_size = image_size
        self.augment = augment
        self.binary = binary
        image_files = {path.stem: path for path in self.image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES}
        mask_files = {path.stem: path for path in self.mask_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES}
        missing_masks = sorted(set(image_files) - set(mask_files))
        missing_images = sorted(set(mask_files) - set(image_files))
        if missing_masks or missing_images:
            raise ValueError(f"影像标签未配对: 缺标签={missing_masks[:5]} 缺影像={missing_images[:5]}")
        self.pairs = [(image_files[key], mask_files[key]) for key in sorted(image_files)]
        if not self.pairs:
            raise ValueError(f"数据集为空: {self.image_dir}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[index]
        image = cv2.resize(_read_image(image_path), (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(_read_mask(mask_path), (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        if self.binary:
            mask = (mask > 0).astype(np.uint8)
        if self.augment:
            if random.random() < 0.5:
                image, mask = np.fliplr(image).copy(), np.fliplr(mask).copy()
            if random.random() < 0.5:
                image, mask = np.flipud(image).copy(), np.flipud(mask).copy()
            turns = random.randint(0, 3)
            if turns:
                image, mask = np.rot90(image, turns).copy(), np.rot90(mask, turns).copy()
            if random.random() < 0.35:
                image = np.clip(image.astype(np.float32) * random.uniform(0.85, 1.15), 0, 255).astype(np.uint8)
        image_tensor = torch.from_numpy(np.moveaxis(image.astype(np.float32) / 255.0, -1, 0))
        mask_tensor = torch.from_numpy(mask.astype(np.int64))
        return image_tensor, mask_tensor


def build_loaders(root: Path, profile: str, binary: bool = False, num_workers: int = 0) -> dict[str, DataLoader]:
    config = PROFILES[profile]
    loaders = {}
    for split in ("train", "val", "test"):
        dataset = SegmentationDataset(root, split, config["image_size"], augment=split == "train", binary=binary)
        loaders[split] = DataLoader(dataset, batch_size=config["batch_size"], shuffle=split == "train", num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return loaders


def prepare_real_building_dataset(output_dir: Path | None = None, image_size: int = 256) -> dict:
    images_dir = REAL_BUILDING_DIR / "images"
    masks_dir = REAL_BUILDING_DIR / "labels"
    image_files = sorted(images_dir.glob("*.tif"))
    mask_by_stem = {path.stem: path for path in masks_dir.glob("*.tif")}
    paired = [(path, mask_by_stem[path.stem]) for path in image_files if path.stem in mask_by_stem]
    if len(paired) < 4:
        raise FileNotFoundError(f"至少需要 4 组真实建筑样本，当前找到 {len(paired)} 组")
    root = Path(output_dir or DATA_DIR / "real_buildings")
    assignment = {0: "train", 1: "train", 2: "val", 3: "test"}
    for split in ("train", "val", "test"):
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "masks").mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, (image_path, mask_path) in enumerate(paired[:4]):
        split = assignment[index]
        image = cv2.resize(_read_image(image_path), (image_size, image_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize((_read_mask(mask_path) > 0).astype(np.uint8), (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        Image.fromarray(image).save(root / split / "images" / f"{image_path.stem}.png")
        Image.fromarray(mask).save(root / split / "masks" / f"{image_path.stem}.png")
        manifest.append({"id": image_path.stem, "split": split, "source_image": str(image_path), "source_mask": str(mask_path)})
    unlabeled = [str(path) for path in image_files if path.stem not in mask_by_stem]
    payload = {"root": str(root), "samples": manifest, "unlabeled": unlabeled}
    (root / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, value in CLASS_COLORS.items():
        color[mask == class_id] = value
    return color
