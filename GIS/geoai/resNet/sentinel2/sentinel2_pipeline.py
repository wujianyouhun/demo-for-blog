"""Sentinel-2 Level-2A SAFE 到 ResNet 分类格网。

处理顺序：读取 SAFE → B02/B03/B04/SCL 分别镶嵌 → 裁剪研究区 →
SCL 云/云影/雪掩膜 → 反射率缩放 → B04/B03/B02 RGB → 滑窗切片 →
ResNet 批量推理 → 分类格网和置信度格网。

仅支持 Level-2A SAFE：必须含 IMG_DATA/R10m 的 B02、B03、B04，
以及 IMG_DATA/R20m 的 SCL 场景分类层。
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, from_bounds, transform as window_transform
from rasterio.warp import transform_bounds
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# 默认沿用当前项目的研究区：min_lon, min_lat, max_lon, max_lat（WGS84）
DEFAULT_BBOX = (97.38, 36.48, 103.77, 39.73)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / "models" / "Classification" / "checkpoints" / "best_model.pth"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# 复用项目中 models/hub/checkpoints/ 已下载的 torchvision 权重。
torch.hub.set_dir(str(ROOT / "models" / "hub"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sentinel-2 SAFE 到 ResNet 分类格网")
    parser.add_argument("--input-safe", type=Path, required=True,
                        help="存放 *.SAFE 或 *.SAFE.zip 的目录；支持递归搜索")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX,
                        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tile-size", type=int, default=192,
                        help="原始 Sentinel-2 10m 像素边长；192 像素约为 1.92 km")
    parser.add_argument("--stride", type=int, default=None, help="默认等于 tile-size，即不重叠")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", help="auto、cpu 或 cuda")
    parser.add_argument("--target-crs", default="EPSG:3857",
                        help="跨 UTM 带镶嵌时使用的统一投影")
    parser.add_argument("--target-resolution", type=float, default=10.0,
                        help="输出 RGB、切片和格网的空间分辨率（米）")
    parser.add_argument("--reflectance-offset", type=float, default=0.0,
                        help="反射率偏移量，默认适用于未采用 offset 的 L2A 产品；若产品元数据 BOA_ADD_OFFSET=-1000，请设为 -1000")
    return parser.parse_args()


def unzip_safe_archives(root: Path, extracted_dir: Path) -> None:
    """解压 .SAFE.zip；已存在对应目录时跳过，避免重复解压。"""
    archives = [root] if root.name.lower().endswith(".safe.zip") else sorted(root.rglob("*.safe.zip"))
    extracted_dir.mkdir(parents=True, exist_ok=True)
    for archive_path in archives:
        # 去掉末尾 .zip 后保留完整的 *.SAFE 目录名。
        target = extracted_dir / archive_path.name[:-4]
        if target.exists():
            continue
        print(f"解压 SAFE: {archive_path.name}")
        with zipfile.ZipFile(archive_path) as archive:
            # 防止 ZIP 内以 ../ 形式写入到输出目录之外。
            destination = extracted_dir.resolve()
            for member in archive.infolist():
                member_path = (destination / member.filename).resolve()
                if not member_path.is_relative_to(destination):
                    raise ValueError(f"ZIP 包含不安全路径: {member.filename}")
            archive.extractall(extracted_dir)


def find_safe_products(root: Path) -> list[Path]:
    """返回 SAFE 产品目录；忽略 SAFE 内部的 manifest.safe 元数据文件。"""
    products = ([root] if root.name.upper().endswith(".SAFE") and root.is_dir()
                else sorted(path for path in root.rglob("*.SAFE") if path.is_dir()))
    if not products:
        raise FileNotFoundError(f"未在 {root} 下找到 *.SAFE 文件夹")
    return products


def find_band_files(safe_products: list[Path], band: str) -> list[Path]:
    """从 SAFE 中找对应 JP2。B02/B03/B04 用 10m，SCL 用 20m。"""
    suffix = f"_{band}_10M.JP2" if band != "SCL" else "_SCL_20M.JP2"
    result = []
    for product in safe_products:
        matches = sorted(path for path in product.rglob("*.jp2") if path.name.upper().endswith(suffix))
        if not matches:
            level = "Level-2A" if band == "SCL" else "10m 波段"
            raise FileNotFoundError(f"{product.name} 中未找到 {band}（需要 {level} 数据）")
        result.extend(matches)
    return result


def mosaic_and_clip(paths: list[Path], bbox_wgs84: tuple[float, float, float, float], output_path: Path,
                    target_crs: str, target_resolution: float) -> None:
    """先统一投影和分辨率，跨场景镶嵌，再裁剪；SCL 用最近邻重采样保留分类编码。"""
    sources = [rasterio.open(path) for path in paths]
    warped = []
    try:
        warped = [WarpedVRT(source, crs=target_crs, resampling=Resampling.nearest) for source in sources]
        left, bottom, right, top = transform_bounds("EPSG:4326", target_crs, *bbox_wgs84, densify_pts=21)
        # 只计算研究区内的跨场景镶嵌，避免将数百公里外的像元也读入内存。
        # 对该 bounds 而言，结果等同于“全景镶嵌后再裁剪”。
        mosaic, transform = merge(warped, bounds=(left, bottom, right, top), res=target_resolution)
        window = from_bounds(left, bottom, right, top, transform=transform).round_offsets().round_lengths()
        full = Window(0, 0, mosaic.shape[2], mosaic.shape[1])
        window = window.intersection(full)
        if window.width <= 0 or window.height <= 0:
            raise ValueError("研究区与 SAFE 影像没有重叠，请检查 --bbox 和输入数据。")
        row0, row1 = int(window.row_off), int(window.row_off + window.height)
        col0, col1 = int(window.col_off), int(window.col_off + window.width)
        clipped = mosaic[:, row0:row1, col0:col1]
        profile = sources[0].profile.copy()
        profile.update(driver="GTiff", crs=target_crs, height=clipped.shape[1], width=clipped.shape[2],
                       transform=window_transform(window, transform))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(clipped)
    finally:
        for source in warped:
            source.close()
        for source in sources:
            source.close()


def scl_invalid_mask(scl: np.ndarray) -> np.ndarray:
    """SCL 无效类别：无数据、饱和/坏像元、云影、中/高概率云、卷云、雪冰。"""
    invalid_classes = (0, 1, 3, 8, 9, 10, 11)
    return np.isin(scl, invalid_classes)


def make_rgb(b02_path: Path, b03_path: Path, b04_path: Path, scl_path: Path, output_path: Path,
             offset: float) -> None:
    """按 B04/B03/B02 写出反射率 RGB，并把 SCL 无效像元写为 NaN。"""
    with rasterio.open(b02_path) as b02_src, rasterio.open(b03_path) as b03_src, \
            rasterio.open(b04_path) as b04_src, rasterio.open(scl_path) as scl_src:
        b02, b03, b04, scl = (source.read(1) for source in (b02_src, b03_src, b04_src, scl_src))
        invalid = scl_invalid_mask(scl) | (b02 == 0) | (b03 == 0) | (b04 == 0)
        # Sentinel-2 L2A 默认量化值为 10000；offset 需与 SAFE 元数据保持一致。
        rgb = (np.stack([b04, b03, b02]).astype("float32") + offset) / 10000.0
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb[:, invalid] = np.nan
        profile = b02_src.profile.copy()
        profile.update(count=3, dtype="float32", nodata=np.nan, compress="deflate")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(rgb)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, list[str], int]:
    """只接受项目训练产生的 ResNet50/101 checkpoint；不存在时用于流程验证的 ImageNet 回退。"""
    if not checkpoint_path.exists():
        print("未找到遥感 checkpoint，改用 ImageNet ResNet50（仅验证处理流程，不可作为地物分类结果）。")
        weights = models.ResNet50_Weights.DEFAULT
        return models.resnet50(weights=weights).to(device).eval(), list(weights.meta["categories"]), 224
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get("model_name", "resnet50")
    class_names = list(checkpoint.get("class_names", []))
    if model_name == "resnet50":
        model = models.resnet50(weights=None)
    elif model_name == "resnet101":
        model = models.resnet101(weights=None)
    else:
        raise ValueError(f"仅支持 ResNet checkpoint，当前模型: {model_name}")
    if not class_names:
        class_names = [str(i) for i in range(checkpoint["model"]["fc.weight"].shape[0])]
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), class_names, int(checkpoint.get("image_size", 224))


def prepare_tile(tile: np.ndarray, image_size: int, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(tile.transpose(2, 0, 1))).float().unsqueeze(0)
    tensor = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return ((tensor - mean) / std).to(device)


def classify_tiles(rgb_path: Path, output_dir: Path, model: nn.Module, class_names: list[str], image_size: int,
                   device: torch.device, tile_size: int, stride: int, batch_size: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(rgb_path) as src:
        rows = list(range(0, src.height - tile_size + 1, stride))
        cols = list(range(0, src.width - tile_size + 1, stride))
        if not rows or not cols:
            raise ValueError("研究区小于切片大小；请减小 --tile-size。")
        labels = np.full((len(rows), len(cols)), -1, dtype=np.int16)
        confidence = np.full((len(rows), len(cols)), np.nan, dtype=np.float32)
        records, pending, positions = [], [], []

        def flush() -> None:
            if not pending:
                return
            with torch.inference_mode():
                probabilities = torch.softmax(model(torch.cat(pending)), dim=1)
            scores, indices = probabilities.max(dim=1)
            for (row, col), score, index in zip(positions, scores.cpu(), indices.cpu()):
                class_id = int(index)
                labels[row, col] = class_id
                confidence[row, col] = float(score)
                records.append({"row": row, "col": col, "class_id": class_id,
                                "class_name": class_names[class_id], "confidence": round(float(score), 6)})
            pending.clear()
            positions.clear()

        for row_index, row in enumerate(rows):
            for col_index, col in enumerate(cols):
                tile = src.read(window=Window(col, row, tile_size, tile_size)).transpose(1, 2, 0)
                if np.mean(~np.isfinite(tile).all(axis=2)) > 0.2:  # 云/无效像元超过 20% 时跳过
                    continue
                pending.append(prepare_tile(np.nan_to_num(tile, nan=0.0), image_size, device))
                positions.append((row_index, col_index))
                if len(pending) >= batch_size:
                    flush()
        flush()

        grid_transform = src.transform * Affine.scale(stride, stride)
        profile = src.profile.copy()
        profile.update(count=1, height=labels.shape[0], width=labels.shape[1], transform=grid_transform,
                       dtype="int16", nodata=-1, compress="deflate")
        with rasterio.open(output_dir / "classification_grid.tif", "w", **profile) as dst:
            dst.write(labels, 1)
        profile.update(dtype="float32", nodata=np.nan)
        with rasterio.open(output_dir / "confidence_grid.tif", "w", **profile) as dst:
            dst.write(confidence, 1)
        (output_dir / "tile_predictions.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"有效切片: {len(records)} / {len(rows) * len(cols)}")


def main() -> None:
    args = parse_args()
    if args.tile_size <= 0 or args.target_resolution <= 0 or (args.stride is not None and args.stride <= 0):
        raise ValueError("切片大小、步长和目标分辨率必须为正数")
    stride = args.stride or args.tile_size
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # 支持从 Copernicus 等平台下载的 *.SAFE.zip；原始 ZIP 不会被删除。
    unzip_safe_archives(args.input_safe, args.output_dir / "00_extracted_safe")
    search_root = args.output_dir / "00_extracted_safe" if any((args.output_dir / "00_extracted_safe").iterdir()) else args.input_safe
    safe_products = find_safe_products(search_root)
    mosaic_dir = args.output_dir / "01_mosaic_clipped"
    clipped = {}
    for band in ("B02", "B03", "B04", "SCL"):
        target = mosaic_dir / f"sentinel2_{band}_mosaic_clip.tif"
        mosaic_and_clip(find_band_files(safe_products, band), tuple(args.bbox), target,
                        args.target_crs, args.target_resolution)
        clipped[band] = target
        print(f"完成镶嵌和裁剪: {band}")
    rgb_path = args.output_dir / "02_rgb_reflectance.tif"
    make_rgb(clipped["B02"], clipped["B03"], clipped["B04"], clipped["SCL"], rgb_path, args.reflectance_offset)
    print(f"完成 RGB 反射率合成与 SCL 掩膜: {rgb_path}")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    model, class_names, image_size = load_model(args.checkpoint, device)
    classify_tiles(rgb_path, args.output_dir / "03_inference", model, class_names, image_size, device,
                   args.tile_size, stride, args.batch_size)
    print("完成。请查看 03_inference/classification_grid.tif 和 confidence_grid.tif")


if __name__ == "__main__":
    main()
