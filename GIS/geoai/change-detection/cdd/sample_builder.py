"""Training sample builders for the GeoAI change detection demo."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.windows import Window


def sample_counts(samples_dir: str | Path) -> Dict[str, int]:
    samples_dir = Path(samples_dir)
    return {
        "time_a": len(list((samples_dir / "time_a").glob("*.tif"))),
        "time_b": len(list((samples_dir / "time_b").glob("*.tif"))),
        "labels": len(list((samples_dir / "labels").glob("*.tif"))),
    }


def validate_samples(samples_dir: str | Path) -> Dict[str, object]:
    counts = sample_counts(samples_dir)
    ok = counts["time_a"] > 0 and counts["time_a"] == counts["time_b"] == counts["labels"]
    message = "样本可用于训练" if ok else (
        "训练样本不足或数量不匹配。请先生成样本，或检查 data/samples/time_a、"
        "data/samples/time_b、data/samples/labels 三个目录。"
    )
    return {"ok": ok, "counts": counts, "message": message}


def generate_synthetic_samples(
    samples_dir: str | Path,
    num_samples: int = 100,
    tile_size: int = 256,
    seed: int = 42,
    overwrite: bool = True,
) -> Dict[str, object]:
    samples_dir = Path(samples_dir)
    dir_a, dir_b, dir_l = _sample_dirs(samples_dir)
    if overwrite:
        _clear_tifs(dir_a, dir_b, dir_l)

    rng = np.random.RandomState(seed)
    transform = from_bounds(116.3, 39.8, 116.4, 39.9, tile_size, tile_size)
    crs = "EPSG:4326"

    for i in range(num_samples):
        base = rng.randint(50, 200, size=(3, tile_size, tile_size), dtype=np.uint8)

        for _ in range(rng.randint(3, 8)):
            cx, cy = rng.randint(20, tile_size - 20, 2)
            w, h = rng.randint(10, 40, 2)
            color = rng.randint(100, 255, 3)
            base[:, cy : cy + h, cx : cx + w] = color.reshape(3, 1, 1)

        img_b = base.copy()
        label = np.zeros((tile_size, tile_size), dtype=np.uint8)
        for _ in range(rng.randint(1, 5)):
            cx, cy = rng.randint(30, tile_size - 30, 2)
            w, h = rng.randint(8, 30, 2)
            img_b[:, cy : cy + h, cx : cx + w] = rng.randint(50, 255, 3).reshape(3, 1, 1)
            label[cy : cy + h, cx : cx + w] = 1
            if rng.random() > 0.5:
                base[:, cy : cy + h, cx : cx + w] = rng.randint(50, 255, 3).reshape(3, 1, 1)

        base = _add_noise(base, rng)
        img_b = _add_noise(img_b, rng)
        _write_triplet(dir_a, dir_b, dir_l, f"sample_{i:04d}", base, img_b, label, transform, crs)

    return {"mode": "synthetic", "samples_dir": str(samples_dir), "counts": sample_counts(samples_dir)}


def generate_weak_label_samples(
    image_a: str,
    image_b: str,
    samples_dir: str | Path,
    tile_size: int = 256,
    stride: int = 256,
    min_change_pixels: int = 20,
    max_tiles: Optional[int] = None,
    model_name: str = "s1_s1c1_vitb",
    threshold: float = 0.5,
    overwrite: bool = True,
) -> Dict[str, object]:
    from config import OUTPUT_DIR
    from cdd.geoai_change import run_geoai_change_detection

    samples_dir = Path(samples_dir)
    dir_a, dir_b, dir_l = _sample_dirs(samples_dir)
    if overwrite:
        _clear_tifs(dir_a, dir_b, dir_l)

    weak = run_geoai_change_detection(
        image_a=image_a,
        image_b=image_b,
        output_dir=OUTPUT_DIR,
        model_name=model_name,
        threshold=threshold,
        tile_size=max(512, tile_size),
        overlap=64,
        visualize=False,
    )
    created = _tile_from_mask(
        image_a,
        image_b,
        weak["mask"],
        dir_a,
        dir_b,
        dir_l,
        tile_size=tile_size,
        stride=stride,
        min_change_pixels=min_change_pixels,
        max_tiles=max_tiles,
        prefix="weak",
    )
    return {
        "mode": "weak-label",
        "label_source": weak["mask"],
        "created": created,
        "samples_dir": str(samples_dir),
        "counts": sample_counts(samples_dir),
    }


def generate_vector_label_samples(
    image_a: str,
    image_b: str,
    vector_label: str,
    samples_dir: str | Path,
    tile_size: int = 256,
    stride: int = 256,
    min_change_pixels: int = 20,
    max_tiles: Optional[int] = None,
    overwrite: bool = True,
) -> Dict[str, object]:
    import geopandas as gpd
    from config import OUTPUT_DIR

    samples_dir = Path(samples_dir)
    dir_a, dir_b, dir_l = _sample_dirs(samples_dir)
    if overwrite:
        _clear_tifs(dir_a, dir_b, dir_l)

    mask_path = Path(OUTPUT_DIR) / f"{Path(vector_label).stem}_rasterized_label.tif"
    with rasterio.open(image_a) as src:
        gdf = gpd.read_file(vector_label)
        if gdf.crs and src.crs and gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)
        shapes = [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty]
        mask = rasterize(
            shapes,
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0,
            dtype="uint8",
            all_touched=True,
        )
        profile = src.profile.copy()
        profile.update(count=1, dtype="uint8", compress="lzw")
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(mask_path, "w", **profile) as dst:
            dst.write(mask, 1)

    created = _tile_from_mask(
        image_a,
        image_b,
        mask_path,
        dir_a,
        dir_b,
        dir_l,
        tile_size=tile_size,
        stride=stride,
        min_change_pixels=min_change_pixels,
        max_tiles=max_tiles,
        prefix="vector",
    )
    return {
        "mode": "vector-label",
        "label_source": str(mask_path),
        "created": created,
        "samples_dir": str(samples_dir),
        "counts": sample_counts(samples_dir),
    }


def _tile_from_mask(
    image_a: str,
    image_b: str,
    mask_path: str | Path,
    dir_a: Path,
    dir_b: Path,
    dir_l: Path,
    tile_size: int,
    stride: int,
    min_change_pixels: int,
    max_tiles: Optional[int],
    prefix: str,
) -> int:
    created = 0
    with rasterio.open(image_a) as src_a, rasterio.open(image_b) as src_b, rasterio.open(mask_path) as src_l:
        width = min(src_a.width, src_b.width, src_l.width)
        height = min(src_a.height, src_b.height, src_l.height)
        for row in range(0, max(1, height - tile_size + 1), stride):
            for col in range(0, max(1, width - tile_size + 1), stride):
                row = min(row, max(0, height - tile_size))
                col = min(col, max(0, width - tile_size))
                win = Window(col, row, tile_size, tile_size)
                label = src_l.read(1, window=win).astype(np.uint8)
                if int(label.sum()) < min_change_pixels:
                    continue
                img_a = _ensure_three(src_a.read(window=win))
                img_b = _ensure_three(src_b.read(window=win))
                name = f"{prefix}_{created:04d}"
                _write_triplet(
                    dir_a,
                    dir_b,
                    dir_l,
                    name,
                    img_a,
                    img_b,
                    (label > 0).astype(np.uint8),
                    src_a.window_transform(win),
                    src_a.crs,
                )
                created += 1
                if max_tiles and created >= max_tiles:
                    return created
    return created


def _write_triplet(dir_a, dir_b, dir_l, name, image_a, image_b, label, transform, crs) -> None:
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": image_a.shape[2],
        "height": image_a.shape[1],
        "count": 3,
        "crs": crs,
        "transform": transform,
        "compress": "lzw",
    }
    with rasterio.open(dir_a / f"{name}.tif", "w", **profile) as dst:
        dst.write(_to_uint8(image_a))
    with rasterio.open(dir_b / f"{name}.tif", "w", **profile) as dst:
        dst.write(_to_uint8(image_b))
    label_profile = {**profile, "count": 1, "dtype": "uint8"}
    with rasterio.open(dir_l / f"{name}.tif", "w", **label_profile) as dst:
        dst.write((label > 0).astype(np.uint8), 1)


def _sample_dirs(samples_dir: Path):
    dirs = samples_dir / "time_a", samples_dir / "time_b", samples_dir / "labels"
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _clear_tifs(*dirs: Path) -> None:
    for d in dirs:
        for f in d.glob("*.tif"):
            f.unlink()


def _ensure_three(data: np.ndarray) -> np.ndarray:
    if data.shape[0] < 3:
        return np.repeat(data, 3, axis=0)[:3]
    return data[:3]


def _to_uint8(data: np.ndarray) -> np.ndarray:
    data = _ensure_three(data)
    if data.dtype == np.uint8:
        return data
    data = data.astype(np.float32)
    if data.max() <= 1:
        return np.clip(data * 255, 0, 255).astype(np.uint8)
    lo, hi = np.percentile(data, 2), np.percentile(data, 98)
    if hi <= lo:
        hi = lo + 1
    return np.clip((data - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def _add_noise(image: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    noise = rng.normal(0, 5, image.shape).astype(np.int16)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
