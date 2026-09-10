"""Landsat 8 Collection 2 Level-2：ZIP 场景到 ResNet 格网分类图。

处理顺序（适用于多个已下载的 Landsat 8 ZIP 场景）：
  1) 解压 ZIP；2) 分别镶嵌 B2/B3/B4/QA_PIXEL；3) 裁剪研究区；
  4) 用 QA_PIXEL 去除云、云影等无效像元；5) DN 转地表反射率；
  6) 组合 B4/B3/B2 为 RGB；7) 滑动切片；8) ResNet 批量推理；
  9) 输出分类格网和每个格网的置信度。

示例（在 geoai 根目录运行）：
  python resNet/landsat8_pipeline.py --input-zips D:/landsat_zip --output-dir outputs/l8

依赖：pip install rasterio numpy torch torchvision
可选本地模型：models/Classification/checkpoints/best_model.pth
没有该模型时，脚本会使用 ImageNet 预训练 ResNet50；此时类别不是遥感地物类别，
仅用于验证整条数据处理链是否可以运行。
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.transform import Affine
from rasterio.windows import Window, from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# 用户给出的研究区：min_lon, min_lat, max_lon, max_lat（WGS84 经纬度）
DEFAULT_BBOX = (97.38, 36.48, 103.77, 39.73)
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = ROOT / "models" / "Classification" / "checkpoints" / "best_model.pth"
LANDSAT_SCALE = 0.0000275
LANDSAT_OFFSET = -0.2
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Landsat 8 ZIP 到 ResNet 分类格网")
    parser.add_argument("--input-zips", type=Path, required=True,
                        help="存放 Landsat ZIP 的目录；支持递归查找 ZIP")
    parser.add_argument("--output-dir", type=Path, required=True, help="输出目录")
    parser.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX,
                        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
                        help="裁剪范围（WGS84 经纬度），默认是 97.38 36.48 103.77 39.73")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tile-size", type=int, default=64,
                        help="原始 Landsat 像素切片边长；64 像素约等于 1.92 km")
    parser.add_argument("--stride", type=int, default=None,
                        help="滑窗步长；默认等于 tile-size，即切片不重叠")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-crs", default="EPSG:3857",
                        help="镶嵌坐标系；研究区跨 UTM 带时应保持默认 EPSG:3857")
    parser.add_argument("--target-resolution", type=float, default=30.0,
                        help="镶嵌网格分辨率（米），默认与 Landsat 多光谱分辨率一致")
    parser.add_argument("--device", default="auto", help="auto、cpu 或 cuda")
    return parser.parse_args()


def unzip_scenes(zip_dir: Path, extracted_dir: Path) -> list[Path]:
    """解压每个 ZIP；已解压的 ZIP 不会重复解压。"""
    zip_paths = sorted(zip_dir.rglob("*.zip"))
    if not zip_paths:
        raise FileNotFoundError(f"未在 {zip_dir} 找到 ZIP 文件")
    extracted_dir.mkdir(parents=True, exist_ok=True)
    for zip_path in zip_paths:
        target = extracted_dir / zip_path.stem
        if not target.exists():
            print(f"解压: {zip_path.name}")
            target.mkdir(parents=True)
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(target)
    return zip_paths


def find_band_files(extracted_dir: Path, band: str) -> list[Path]:
    """寻找 Collection 2 L2 的单波段影像，例如 *_SR_B4.TIF、*_QA_PIXEL.TIF。"""
    suffix = f"_{band}.TIF"
    files = sorted(path for path in extracted_dir.rglob("*.TIF") if path.name.upper().endswith(suffix))
    if not files:
        raise FileNotFoundError(f"未找到 {band}。请确认 ZIP 是 Landsat Collection 2 Level-2 产品。")
    return files


def mosaic_and_clip(paths: list[Path], bbox_wgs84: tuple[float, float, float, float], output_path: Path,
                    target_crs: str, target_resolution: float) -> None:
    """统一投影后先镶嵌同一波段的全部场景，再按 WGS84 范围裁剪并保存。"""
    sources = [rasterio.open(path) for path in paths]
    warped_sources = []
    try:
        # 此研究区跨两个 UTM 带。先统一为 EPSG:3857，才能正确镶嵌不同投影的场景。
        warped_sources = [WarpedVRT(source, crs=target_crs, resampling=Resampling.nearest) for source in sources]
        mosaic, transform = merge(warped_sources, res=target_resolution)
        profile = sources[0].profile.copy()
        profile.update(crs=target_crs)
        left, bottom, right, top = transform_bounds("EPSG:4326", target_crs, *bbox_wgs84, densify_pts=21)
        full_window = from_bounds(left, bottom, right, top, transform=transform)
        # 整数化窗口并与镶嵌影像边界相交，防止裁剪范围略超出影像时出错。
        full = Window(0, 0, mosaic.shape[2], mosaic.shape[1])
        window = full_window.round_offsets().round_lengths().intersection(full)
        if window.width <= 0 or window.height <= 0:
            raise ValueError("研究区与已下载的影像没有重叠，请检查 ZIP 场景和 --bbox。")
        row0, row1 = int(window.row_off), int(window.row_off + window.height)
        col0, col1 = int(window.col_off), int(window.col_off + window.width)
        clipped = mosaic[:, row0:row1, col0:col1]
        profile.update(height=clipped.shape[1], width=clipped.shape[2], transform=rasterio.windows.transform(window, transform))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(clipped)
    finally:
        for source in warped_sources:
            source.close()
        for source in sources:
            source.close()


def qa_invalid_mask(qa: np.ndarray) -> np.ndarray:
    """Collection 2 QA_PIXEL：填充值、膨胀云、卷云、云、云影均视为无效。"""
    bad_bits = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)
    return (qa.astype(np.uint16) & bad_bits) != 0


def make_rgb(b2_path: Path, b3_path: Path, b4_path: Path, qa_path: Path, output_path: Path) -> None:
    """DN → 反射率，应用 QA 掩膜后以 B4/B3/B2 顺序写出 float32 RGB GeoTIFF。"""
    with rasterio.open(b2_path) as b2_src, rasterio.open(b3_path) as b3_src, \
            rasterio.open(b4_path) as b4_src, rasterio.open(qa_path) as qa_src:
        b2, b3, b4, qa = (src.read(1) for src in (b2_src, b3_src, b4_src, qa_src))
        invalid = qa_invalid_mask(qa) | (b2 == 0) | (b3 == 0) | (b4 == 0)
        # Landsat C2 L2 表面反射率缩放；保留 NaN 让后续切片可以跳过无效区域。
        rgb = np.stack([b4, b3, b2]).astype("float32") * LANDSAT_SCALE + LANDSAT_OFFSET
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb[:, invalid] = np.nan
        profile = b2_src.profile.copy()
        profile.update(count=3, dtype="float32", nodata=np.nan, compress="deflate")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(rgb)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, list[str], int]:
    """加载项目训练产物；没有 checkpoint 时回退到 ImageNet ResNet50。"""
    if not checkpoint_path.exists():
        print("未找到本地遥感 checkpoint，改用 ImageNet ResNet50（仅用于流程测试）。")
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
        raise ValueError(f"该示例仅支持 ResNet checkpoint，实际模型为: {model_name}")
    if not class_names:
        class_names = [str(index) for index in range(checkpoint["model"]["fc.weight"].shape[0])]
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), class_names, int(checkpoint.get("image_size", 224))


def tensor_from_tile(tile: np.ndarray, image_size: int, device: torch.device) -> torch.Tensor:
    """HWC 反射率切片 → NCHW ResNet 输入；缩放与 ImageNet 标准化在此处统一执行。"""
    tensor = torch.from_numpy(np.ascontiguousarray(tile.transpose(2, 0, 1))).float().unsqueeze(0)
    tensor = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return ((tensor - mean) / std).to(device)


def classify_tiles(rgb_path: Path, output_dir: Path, model: nn.Module, class_names: list[str], image_size: int,
                   device: torch.device, tile_size: int, stride: int, batch_size: int) -> None:
    """滑动窗口分类，并输出一个像素代表一个切片的空间分类格网。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(rgb_path) as src:
        rows = list(range(0, src.height - tile_size + 1, stride))
        cols = list(range(0, src.width - tile_size + 1, stride))
        if not rows or not cols:
            raise ValueError("研究区小于切片尺寸；请减小 --tile-size。")
        labels = np.full((len(rows), len(cols)), -1, dtype=np.int16)
        confidence = np.full((len(rows), len(cols)), np.nan, dtype=np.float32)
        records: list[dict] = []
        pending: list[torch.Tensor] = []
        positions: list[tuple[int, int]] = []

        def flush_batch() -> None:
            if not pending:
                return
            with torch.inference_mode():
                probabilities = torch.softmax(model(torch.cat(pending, dim=0)), dim=1)
            scores, indices = probabilities.max(dim=1)
            for (r, c), score, index in zip(positions, scores.cpu(), indices.cpu()):
                class_id = int(index)
                labels[r, c] = class_id
                confidence[r, c] = float(score)
                records.append({"row": r, "col": c, "class_id": class_id,
                                "class_name": class_names[class_id], "confidence": round(float(score), 6)})
            pending.clear()
            positions.clear()

        for r, row in enumerate(rows):
            for c, col in enumerate(cols):
                tile = src.read(window=Window(col, row, tile_size, tile_size)).transpose(1, 2, 0)
                # 云、NoData 或边缘无效像元超过 20% 的切片不输出类别。
                if np.mean(~np.isfinite(tile).all(axis=2)) > 0.2:
                    continue
                tile = np.nan_to_num(tile, nan=0.0)
                pending.append(tensor_from_tile(tile, image_size, device))
                positions.append((r, c))
                if len(pending) >= batch_size:
                    flush_batch()
        flush_batch()

        # 输出格网的一个像素代表一个切片，像素大小 = stride × 原始 Landsat 像元大小。
        grid_transform = src.transform * Affine.translation(0, 0) * Affine.scale(stride, stride)
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
    if args.tile_size <= 0 or (args.stride is not None and args.stride <= 0):
        raise ValueError("--tile-size 和 --stride 必须为正整数")
    stride = args.stride or args.tile_size
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extracted = args.output_dir / "01_extracted"
    mosaic_dir = args.output_dir / "02_mosaic_clipped"

    unzip_scenes(args.input_zips, extracted)
    # 注意顺序：这里每个波段都先跨场景镶嵌，之后才裁剪研究区。
    clipped_paths = {}
    for band in ("SR_B2", "SR_B3", "SR_B4", "QA_PIXEL"):
        target = mosaic_dir / f"landsat8_{band}_mosaic_clip.tif"
        mosaic_and_clip(find_band_files(extracted, band), tuple(args.bbox), target,
                        args.target_crs, args.target_resolution)
        clipped_paths[band] = target
        print(f"完成镶嵌和裁剪: {band}")

    rgb_path = args.output_dir / "03_rgb_reflectance.tif"
    make_rgb(clipped_paths["SR_B2"], clipped_paths["SR_B3"], clipped_paths["SR_B4"], clipped_paths["QA_PIXEL"], rgb_path)
    print(f"完成 RGB 反射率合成与云掩膜: {rgb_path}")

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    model, class_names, image_size = load_model(args.checkpoint, device)
    classify_tiles(rgb_path, args.output_dir / "04_inference", model, class_names, image_size, device,
                   args.tile_size, stride, args.batch_size)
    print("完成。请查看 04_inference/classification_grid.tif 和 confidence_grid.tif")


if __name__ == "__main__":
    main()
