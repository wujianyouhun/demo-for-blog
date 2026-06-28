#!/usr/bin/env python3
"""
建筑物提取脚本
===========================
从高分辨率遥感 TIF 影像中提取建筑物，执行正则化后输出 SHP 文件。

流程:
  1. 加载 DeepLabV3+ 语义分割模型
  2. 滑动窗口推理 → 分类栅格图
  3. 提取建筑物多边形（class_id=1）
  4. 正则化：简化 → 平滑 → 正交化 → 过滤 → 属性计算
  5. 导出 SHP 文件

用法:
  python building_extractor.py                           # 自动扫描 data/input/ 下所有 tif
  python building_extractor.py --image data/input/xx.tif # 指定影像
  python building_extractor.py --model data/models/best.pth --model-name deeplabv3p_resnet50
  python building_extractor.py --device cuda             # 使用 GPU
"""
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.features import shapes as rasterio_shapes
from scipy.ndimage import gaussian_filter
from shapely.geometry import Polygon, MultiPolygon, shape
from shapely.ops import unary_union
import geopandas as gpd

import torch
import torch.nn as nn

# ── 项目配置 ──
from config import (
    PROJECT_ROOT, INPUT_DIR, MODELS_DIR, OUTPUT_DIR,
    CLASS_NAMES, NUM_CLASSES, BUILDING_CLASS_ID,
    MODEL_CONFIG, DEFAULT_MODEL,
    INFERENCE_CONFIG, REGULARIZE_CONFIG,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════
#  模型构建 & 加载
# ════════════════════════════════════════════════════════════

def build_model(model_name: str, num_classes: int = NUM_CLASSES,
                in_channels: int = 3, pretrained: bool = True) -> nn.Module:
    """构建 DeepLabV3+ 语义分割模型。"""
    import segmentation_models_pytorch as smp

    registry = {
        "deeplabv3p_resnet50":  "resnet50",
        "deeplabv3p_resnet101": "resnet101",
        "deeplabv3p_mobilenet": "mobilenet_v2",
    }
    if model_name not in registry:
        raise ValueError(f"未知模型: {model_name}，可选: {list(registry.keys())}")

    encoder_name = registry[model_name]
    logger.info("构建模型: %s (encoder=%s, classes=%d)", model_name, encoder_name, num_classes)

    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,
    )
    return model


def load_model(path: Path, model_name: str, device: str = "cpu") -> nn.Module:
    """从 .pth 文件加载训练好的模型权重。"""
    logger.info("加载模型权重: %s", path)
    model = build_model(model_name, pretrained=False)

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    logger.info("模型加载完成 (device=%s)", device)
    return model


# ════════════════════════════════════════════════════════════
#  滑动窗口推理
# ════════════════════════════════════════════════════════════

@torch.no_grad()
def sliding_window_inference(
    model: nn.Module,
    image_path: Path,
    output_path: Path,
    tile_size: int = 256,
    overlap: int = 32,
    batch_size: int = 4,
    smoothing_sigma: float = 1.0,
    device: str = "cpu",
) -> Path:
    """
    对大幅遥感影像执行滑动窗口语义分割推理。

    Args:
        model: 已加载的分割模型
        image_path: 输入 TIF 影像路径
        output_path: 输出分类栅格路径
        tile_size: 滑窗尺寸
        overlap: 重叠像素
        batch_size: 批处理大小
        smoothing_sigma: 高斯平滑 sigma
        device: 推理设备

    Returns:
        输出文件路径
    """
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(image_path) as src:
        height, width = src.height, src.width
        num_bands = src.count
        profile = src.profile.copy()

        logger.info("读取影像: %s (%dx%d, %d bands)", image_path.name, height, width, num_bands)
        image = src.read().astype(np.float32)
        if image.max() > 1.0:
            image /= 255.0

    stride = tile_size - overlap

    # 从模型推断类别数
    num_classes = NUM_CLASSES
    for module in reversed(list(model.modules())):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            num_classes = module.out_channels
            break

    # 概率累积图
    prob_sum = np.zeros((num_classes, height, width), dtype=np.float64)
    count_map = np.zeros((1, height, width), dtype=np.float64)

    # 生成所有窗口位置
    windows = []
    for row in range(0, height, stride):
        for col in range(0, width, stride):
            r_end = min(row + tile_size, height)
            c_end = min(col + tile_size, width)
            r_start = max(0, r_end - tile_size)
            c_start = max(0, c_end - tile_size)
            windows.append((r_start, c_start, r_end, c_end))

    total_windows = len(windows)
    logger.info("滑窗推理: %d 个窗口 (tile=%d, stride=%d, batch=%d)",
                total_windows, tile_size, stride, batch_size)

    batch_tiles = []
    batch_wins = []
    processed = 0
    t0 = time.time()

    for i, (r0, c0, r1, c1) in enumerate(windows):
        tile = image[:, r0:r1, c0:c1]

        # 不足 tile_size 的边缘 tile 做 reflect padding
        pad_h = tile_size - tile.shape[1]
        pad_w = tile_size - tile.shape[2]
        if pad_h > 0 or pad_w > 0:
            tile = np.pad(tile, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")

        batch_tiles.append(tile)
        batch_wins.append((r0, c0, r1, c1))

        if len(batch_tiles) >= batch_size or i == total_windows - 1:
            tensor = torch.from_numpy(np.stack(batch_tiles)).to(dev)

            with torch.amp.autocast("cuda", enabled=(dev.type == "cuda")):
                outputs = model(tensor)

            probs = torch.softmax(outputs, dim=1).cpu().numpy()

            for j, (r0, c0, r1, c1) in enumerate(batch_wins):
                h_act, w_act = r1 - r0, c1 - c0
                prob_sum[:, r0:r1, c0:c1] += probs[j, :, :h_act, :w_act]
                count_map[:, r0:r1, c0:c1] += 1

            processed += len(batch_tiles)
            elapsed = time.time() - t0
            pct = processed / total_windows * 100
            speed = processed / elapsed if elapsed > 0 else 0
            logger.info("  进度: %d/%d (%.0f%%) - %.1f tiles/s", processed, total_windows, pct, speed)

            batch_tiles = []
            batch_wins = []

    # 概率平均
    count_map = np.maximum(count_map, 1)
    prob_avg = prob_sum / count_map

    # 高斯平滑
    if smoothing_sigma > 0:
        logger.info("高斯平滑 (sigma=%.1f)", smoothing_sigma)
        for c in range(num_classes):
            prob_avg[c] = gaussian_filter(prob_avg[c], sigma=smoothing_sigma)

    # argmax 分类
    classification = np.argmax(prob_avg, axis=0).astype(np.uint8)

    # 保存分类栅格
    profile.update(count=1, dtype="uint8", compress="deflate")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(classification, 1)

    logger.info("推理完成: %s (%dx%d, %d classes)", output_path.name, height, width, num_classes)
    return output_path


# ════════════════════════════════════════════════════════════
#  栅格 → 矢量（仅提取建筑物）
# ════════════════════════════════════════════════════════════

def mask_to_buildings(
    mask_path: Path,
    min_area_px: int = 30,
) -> gpd.GeoDataFrame:
    """
    从分类栅格中提取建筑物多边形。

    Args:
        mask_path: 分类结果 TIF
        min_area_px: 最小面积（像素单位），过滤碎斑

    Returns:
        GeoDataFrame（仅包含 building 要素）
    """
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs

    logger.info("矢量化: %s (%dx%d, CRS=%s)", mask_path.name, mask.shape[1], mask.shape[0], crs)

    # 只提取 building (class_id = 1)
    building_mask = (mask == BUILDING_CLASS_ID).astype(np.uint8)

    records = []
    for geom_json, value in rasterio_shapes(building_mask, mask=building_mask, transform=transform):
        if value == 0:
            continue
        geom = shape(geom_json)
        if not geom.is_valid:
            geom = geom.buffer(0)
        area = geom.area
        if area < min_area_px:
            continue
        records.append({
            "geometry": geom,
            "class_id": BUILDING_CLASS_ID,
            "class_name": "building",
            "area_px": area,
        })

    if not records:
        logger.warning("未提取到建筑物多边形")
        gdf = gpd.GeoDataFrame(
            columns=["class_id", "class_name", "area_px", "geometry"],
            geometry="geometry", crs=crs,
        )
    else:
        gdf = gpd.GeoDataFrame(records, crs=crs)

    logger.info("矢量化完成: %d 个建筑物多边形", len(gdf))
    return gdf


# ════════════════════════════════════════════════════════════
#  建筑物正则化
# ════════════════════════════════════════════════════════════

def _chaikin_smooth(coords, iterations=1):
    """Chaikin 曲线平滑。"""
    for _ in range(iterations):
        new_coords = [coords[0]]
        for i in range(len(coords) - 1):
            p0 = np.array(coords[i])
            p1 = np.array(coords[i + 1])
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1
            new_coords.extend([tuple(q), tuple(r)])
        new_coords.append(coords[-1])
        coords = new_coords
    return coords


def _smooth_polygon(geom, iters):
    if isinstance(geom, Polygon):
        ext = _chaikin_smooth(list(geom.exterior.coords), iters)
        holes = [_chaikin_smooth(list(h.coords), iters) for h in geom.interiors]
        try:
            return Polygon(ext, holes)
        except Exception:
            return geom
    elif isinstance(geom, MultiPolygon):
        return MultiPolygon([_smooth_polygon(p, iters) for p in geom.geoms])
    return geom


def _orthogonalize_polygon(geom: Polygon) -> Polygon:
    """正交化：使建筑物角点更接近直角。"""
    if not isinstance(geom, Polygon) or not geom.is_valid:
        return geom

    coords = np.array(geom.exterior.coords[:-1])
    if len(coords) < 4:
        return geom

    n = len(coords)
    new_coords = coords.copy()

    for i in range(n):
        p_prev = coords[(i - 1) % n]
        p_curr = coords[i]
        p_next = coords[(i + 1) % n]

        v1 = p_prev - p_curr
        v2 = p_next - p_curr

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

        if 70 < angle < 110:
            v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
            v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)
            v2_ortho = v2_norm - np.dot(v2_norm, v1_norm) * v1_norm
            v2_ortho = v2_ortho / (np.linalg.norm(v2_ortho) + 1e-10)
            dist = np.linalg.norm(p_next - p_curr)
            new_coords[(i + 1) % n] = p_curr + v2_ortho * dist

    closed = np.vstack([new_coords, new_coords[0]])
    try:
        result = Polygon(closed)
        if result.is_valid and result.area > 0:
            return result
    except Exception:
        pass
    return geom


def regularize_buildings(
    gdf: gpd.GeoDataFrame,
    config: dict = None,
) -> gpd.GeoDataFrame:
    """
    对建筑物多边形执行正则化处理链。

    处理步骤: 简化 → 平滑 → 正交化 → 面积过滤 → 属性计算

    Args:
        gdf: 输入建筑物 GeoDataFrame
        config: 正则化参数（默认使用 REGULARIZE_CONFIG）

    Returns:
        正则化后的 GeoDataFrame
    """
    if config is None:
        config = REGULARIZE_CONFIG

    if len(gdf) == 0:
        logger.warning("无建筑物可正则化")
        return gdf

    gdf = gdf.copy()
    logger.info("开始后处理: %d 个建筑物, 配置=%s", len(gdf), config)

    # 1. Douglas-Peucker 简化
    tol = config.get("simplify_tolerance", 0)
    if tol and tol > 0:
        before = len(gdf)
        gdf["geometry"] = gdf.geometry.simplify(tol, preserve_topology=True)
        gdf = gdf[~gdf.geometry.is_empty].copy()
        logger.info("  简化 (tol=%.1f): %d → %d", tol, before, len(gdf))

    # 2. Chaikin 平滑
    iters = config.get("smooth_iterations", 0)
    if iters and iters > 0:
        gdf["geometry"] = gdf.geometry.apply(lambda g: _smooth_polygon(g, iters))
        logger.info("  平滑: %d 次迭代", iters)

    # 3. 正交化（建筑物专用）
    if config.get("orthogonalize", False):
        gdf["geometry"] = gdf.geometry.apply(_orthogonalize_polygon)
        logger.info("  正交化完成")

    # 4. 面积过滤
    min_area = config.get("min_area", 0)
    if min_area and min_area > 0:
        before = len(gdf)
        gdf = gdf[gdf.geometry.area >= min_area].copy()
        logger.info("  面积过滤 (min=%.1f): %d → %d", min_area, before, len(gdf))

    # 5. 属性计算
    gdf["area"] = gdf.geometry.area
    gdf["perimeter"] = gdf.geometry.length
    perim_sq = gdf["perimeter"] ** 2
    perim_sq = perim_sq.replace(0, np.nan)
    gdf["compactness"] = (4 * np.pi * gdf["area"] / perim_sq).fillna(0.0)

    # 修复无效几何
    gdf["geometry"] = gdf.geometry.buffer(0)
    gdf = gdf[gdf.geometry.notnull()].copy()

    logger.info("后处理完成: %d 个建筑物", len(gdf))
    return gdf


# ════════════════════════════════════════════════════════════
#  导出 SHP
# ════════════════════════════════════════════════════════════

def export_shp(gdf: gpd.GeoDataFrame, output_path: Path) -> Path:
    """
    导出建筑物到 SHP 文件。

    SHP 要求字段名 ≤10 字符，这里做截断处理。

    Args:
        gdf: 建筑物 GeoDataFrame
        output_path: 输出 .shp 路径

    Returns:
        输出文件路径
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # SHP 字段名限制 ≤ 10 字符
    export_gdf = gdf.copy()
    rename_map = {}
    for col in export_gdf.columns:
        if col == "geometry":
            continue
        if len(col) > 10:
            rename_map[col] = col[:10]
    if rename_map:
        export_gdf = export_gdf.rename(columns=rename_map)
        logger.info("SHP 字段名截断: %s", rename_map)

    export_gdf.to_file(output_path, driver="ESRI Shapefile", encoding="utf-8")
    logger.info("SHP 导出完成: %s (%d 个建筑物)", output_path, len(export_gdf))
    return output_path


# ════════════════════════════════════════════════════════════
#  完整流水线
# ════════════════════════════════════════════════════════════

def find_model_file(models_dir: Path) -> Optional[Path]:
    """自动在 data/models/ 下查找最新的 .pth 模型文件。"""
    pth_files = sorted(models_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    if pth_files:
        return pth_files[0]
    return None


def find_input_images(input_dir: Path) -> list[Path]:
    """扫描 data/input/ 下所有 TIF 影像。"""
    seen = set()
    tifs = []
    for ext in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        for p in input_dir.glob(ext):
            resolved = p.resolve()
            if resolved not in seen:
                seen.add(resolved)
                tifs.append(p)
    return sorted(tifs)


def run_pipeline(
    image_path: Path,
    model: nn.Module,
    model_name: str,
    device: str = "cpu",
    inference_cfg: dict = None,
    regularize_cfg: dict = None,
) -> Path:
    """
    执行完整的建筑物提取流水线。

    Args:
        image_path: 输入 TIF 影像
        model: 已加载的模型
        model_name: 模型名称
        device: 推理设备
        inference_cfg: 推理参数
        regularize_cfg: 正则化参数

    Returns:
        输出的 SHP 文件路径
    """
    if inference_cfg is None:
        inference_cfg = INFERENCE_CONFIG
    if regularize_cfg is None:
        regularize_cfg = REGULARIZE_CONFIG

    stem = image_path.stem
    cls_path = OUTPUT_DIR / f"{stem}_classification.tif"
    shp_path = OUTPUT_DIR / f"{stem}_buildings.shp"

    logger.info("=" * 60)
    logger.info("建筑物提取流水线")
    logger.info("=" * 60)
    logger.info("输入影像: %s", image_path)
    logger.info("模型:     %s", model_name)
    logger.info("设备:     %s", device)
    logger.info("输出目录: %s", OUTPUT_DIR)

    # Step 1: 滑动窗口推理
    t0 = time.time()
    logger.info("[1/4] 滑动窗口推理 ...")
    sliding_window_inference(
        model=model,
        image_path=image_path,
        output_path=cls_path,
        tile_size=inference_cfg.get("tile_size", 256),
        overlap=inference_cfg.get("overlap", 32),
        batch_size=inference_cfg.get("batch_size", 4),
        smoothing_sigma=inference_cfg.get("smoothing_sigma", 1.0),
        device=device,
    )
    t_infer = time.time() - t0
    logger.info("推理耗时: %.1f s", t_infer)

    # Step 2: 矢量化（仅建筑物）
    t0 = time.time()
    logger.info("[2/4] 提取建筑物多边形 ...")
    gdf = mask_to_buildings(
        cls_path,
        min_area_px=inference_cfg.get("min_area_px", 30),
    )
    t_vec = time.time() - t0
    logger.info("矢量化耗时: %.1f s, 提取 %d 个建筑物", t_vec, len(gdf))

    if len(gdf) == 0:
        logger.warning("未检测到建筑物，跳过正则化和导出")
        return cls_path

    # Step 3: 正则化
    t0 = time.time()
    logger.info("[3/4] 建筑物正则化 ...")
    gdf = regularize_buildings(gdf, regularize_cfg)
    t_reg = time.time() - t0
    logger.info("正则化耗时: %.1f s, 剩余 %d 个建筑物", t_reg, len(gdf))

    if len(gdf) == 0:
        logger.warning("正则化后无建筑物剩余")
        return cls_path

    # Step 4: 导出 SHP
    logger.info("[4/4] 导出 SHP ...")
    export_shp(gdf, shp_path)

    # 汇总
    logger.info("=" * 60)
    logger.info("处理完成!")
    logger.info("  分类栅格: %s", cls_path)
    logger.info("  建筑物SHP: %s", shp_path)
    logger.info("  建筑物数量: %d", len(gdf))
    logger.info("  总面积: %.1f", gdf["area"].sum() if "area" in gdf.columns else 0)
    logger.info("=" * 60)

    return shp_path


# ════════════════════════════════════════════════════════════
#  CLI 入口
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="从遥感 TIF 影像中提取建筑物并输出 SHP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image", type=str, default=None,
                        help="输入 TIF 影像路径 (默认扫描 data/input/ 下所有 tif)")
    parser.add_argument("--model", type=str, default=None,
                        help="训练模型 .pth 路径 (默认自动查找 data/models/ 下最新 .pth)")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL,
                        choices=list(MODEL_CONFIG.keys()),
                        help=f"模型架构 (默认: {DEFAULT_MODEL})")
    parser.add_argument("--device", type=str, default="auto",
                        help="推理设备: cpu / cuda / auto (默认: auto)")
    parser.add_argument("--tile-size", type=int, default=None,
                        help="滑窗尺寸 (默认: 256)")
    parser.add_argument("--overlap", type=int, default=None,
                        help="重叠像素 (默认: 32)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="批处理大小 (默认: 4)")
    parser.add_argument("--simplify-tolerance", type=float, default=None,
                        help="简化容差 (默认: 2.0)")
    parser.add_argument("--smooth-iterations", type=int, default=None,
                        help="平滑迭代次数 (默认: 3)")
    parser.add_argument("--min-area", type=float, default=None,
                        help="最小面积过滤 (默认: 50.0)")
    parser.add_argument("--no-orthogonalize", action="store_true",
                        help="禁用正交化")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录 (默认: data/output/)")

    args = parser.parse_args()

    # ── 设备选择 ──
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info("推理设备: %s", device)

    # ── 覆盖配置 ──
    inference_cfg = INFERENCE_CONFIG.copy()
    if args.tile_size:
        inference_cfg["tile_size"] = args.tile_size
    if args.overlap:
        inference_cfg["overlap"] = args.overlap
    if args.batch_size:
        inference_cfg["batch_size"] = args.batch_size

    regularize_cfg = REGULARIZE_CONFIG.copy()
    if args.simplify_tolerance is not None:
        regularize_cfg["simplify_tolerance"] = args.simplify_tolerance
    if args.smooth_iterations is not None:
        regularize_cfg["smooth_iterations"] = args.smooth_iterations
    if args.min_area is not None:
        regularize_cfg["min_area"] = args.min_area
    if args.no_orthogonalize:
        regularize_cfg["orthogonalize"] = False

    if args.output_dir:
        global OUTPUT_DIR
        OUTPUT_DIR = Path(args.output_dir)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 查找模型 ──
    model_path = Path(args.model) if args.model else find_model_file(MODELS_DIR)
    if model_path and model_path.exists():
        logger.info("使用训练模型: %s (%.1f MB)", model_path, model_path.stat().st_size / 1024 / 1024)
        model = load_model(model_path, args.model_name, device="cpu")
    else:
        logger.warning("未找到训练模型 (.pth)，将使用 ImageNet 预训练权重 (未经微调，分割效果有限)")
        model = build_model(args.model_name, pretrained=True)
        model.eval()

    # ── 查找输入影像 ──
    if args.image:
        images = [Path(args.image)]
        if not images[0].exists():
            logger.error("影像文件不存在: %s", images[0])
            sys.exit(1)
    else:
        images = find_input_images(INPUT_DIR)
        if not images:
            logger.error(
                "未在 %s 下找到 TIF 影像。\n"
                "  请将 .tif 文件放入 data/input/ 目录，或使用 --image 参数指定路径。",
                INPUT_DIR,
            )
            sys.exit(1)

    logger.info("找到 %d 张输入影像:", len(images))
    for img in images:
        logger.info("  %s (%.1f MB)", img.name, img.stat().st_size / 1024 / 1024)

    # ── 逐张处理 ──
    results = []
    for img in images:
        try:
            result = run_pipeline(
                image_path=img,
                model=model,
                model_name=args.model_name,
                device=device,
                inference_cfg=inference_cfg,
                regularize_cfg=regularize_cfg,
            )
            results.append((img.name, result))
        except Exception as e:
            logger.error("处理 %s 时出错: %s", img.name, e, exc_info=True)

    # ── 汇总 ──
    logger.info("\n" + "=" * 60)
    logger.info("全部处理完成!")
    for name, path in results:
        logger.info("  %s → %s", name, path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
