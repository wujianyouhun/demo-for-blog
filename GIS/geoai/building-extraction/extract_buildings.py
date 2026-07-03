"""GeoAI 建筑物轮廓提取与正则化主流程。

脚本面向两类使用场景：
1. 公众号/教学演示：把原始 GeoTIFF 放到项目 `data/` 目录，直接运行命令生成成果；
2. 大幅遥感影像处理：自动切片推理、合并瓦片结果、去除重叠重复建筑物。

为了让 `--check-env` 能在依赖不完整的环境里也正常执行，GIS 和深度学习相关
依赖尽量放在函数内部延迟导入。这样用户可以先检查环境，再进入真正的模型推理。
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


# ── 项目目录约定 ──────────────────────────────────────────────
# 原始影像直接放在 data/ 根目录；模型与输出统一放在项目内，方便打包和复现。
PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = Path(os.getenv("GEOAI_MODELS_DIR", str(REPO_ROOT / "models"))).expanduser()
if not MODEL_DIR.is_absolute():
    MODEL_DIR = (REPO_ROOT / MODEL_DIR).resolve()
OUTPUT_DIR = PROJECT_DIR / "outputs"
SYNTHETIC_TIF = DATA_DIR / "synthetic_test.tif"
DEFAULT_SOURCE_TIF = None
EXISTING_SAM_MODEL_DIR = REPO_ROOT / "sam" / "models"

# 将 Hugging Face、PyTorch 等缓存固定到项目 models/ 下，避免默认写入系统盘。
os.environ.setdefault("HF_HOME", str(MODEL_DIR / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(MODEL_DIR / "huggingface" / "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(MODEL_DIR / "huggingface" / "hub"))
os.environ.setdefault("TORCH_HOME", str(MODEL_DIR / "torch"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# SAM 模型配置。expected_bytes 用于判断下载文件是否完整，避免半截权重参与推理。
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

# GeoAI 建筑物提取模型位于 Hugging Face 仓库，下载后缓存到根 models/geoai/。
GEOAI_BUILDING_MODEL = {
    "repo_id": "giswqs/geoai",
    "filename": "building_footprints_usa.pth",
}

# SAM 自动分割尝试两组参数，用于在召回率和噪声之间做一个轻量对比。
SAM_PARAM_GRID = [
    {"points_per_side": 16, "pred_iou_thresh": 0.86, "stability_score_thresh": 0.92},
    {"points_per_side": 24, "pred_iou_thresh": 0.88, "stability_score_thresh": 0.95},
]

# 运行主流程必须具备的依赖；可选依赖缺失时只跳过对应功能或预览。
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
    """单次模型运行的结果摘要，用于写统计表和挑选最佳成果。"""

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
    """统一日志前缀，方便在终端中区分本脚本输出。"""
    print(f"[building-extraction] {message}", flush=True)


def normalize_device(device: str) -> str:
    """规范化推理设备名称，并在 CUDA 不可用时自动回退到 CPU。"""
    value = (device or "cpu").strip().lower()
    aliases = {"gpu": "cuda", "nvidia": "cuda", "cuda0": "cuda:0"}
    value = aliases.get(value, value)
    if value == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    if value.startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                return value
            log(f"Requested device '{device}' but CUDA is not available; falling back to cpu")
            return "cpu"
        except Exception:
            log(f"Requested device '{device}' but torch is not importable; falling back to cpu")
            return "cpu"
    return value


def parse_models(value: str) -> List[str]:
    """解析命令行传入的模型列表，例如 `geoai,sam_vit_b`。"""
    models = [item.strip() for item in value.split(",") if item.strip()]
    valid = {"geoai", *SAM_MODELS.keys()}
    unknown = [m for m in models if m not in valid]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown model(s): {', '.join(unknown)}")
    return models


def check_env() -> int:
    """检查运行环境依赖，返回 0 表示可继续执行主流程。"""
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
    """创建项目运行所需目录，重复调用不会破坏已有文件。"""
    for path in (DATA_DIR, MODEL_DIR, OUTPUT_DIR, OUTPUT_DIR / "compare", OUTPUT_DIR / "preview"):
        path.mkdir(parents=True, exist_ok=True)


def create_synthetic_image(path: Path, size: int = 768) -> Path:
    """创建一张带有矩形建筑物的合成 GeoTIFF，用于无真实影像时的冒烟测试。"""
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


def same_path(left: Path, right: Path) -> bool:
    """比较两个路径是否指向同一个文件，兼容目标文件尚不存在的情况。"""
    try:
        return left.resolve() == right.resolve()
    except FileNotFoundError:
        return left.absolute() == right.absolute()


def find_data_tifs(data_dir: Path = DATA_DIR) -> List[Path]:
    """查找直接放在项目 data/ 目录下的 GeoTIFF 影像。"""
    seen = set()
    tifs: List[Path] = []
    for pattern in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        for path in data_dir.glob(pattern):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            tifs.append(path)
    return sorted(tifs, key=lambda item: item.name.lower())


def format_tif_choices(paths: Sequence[Path]) -> str:
    """把多张候选影像格式化成错误提示中的列表。"""
    return "\n".join(f"  - {path}" for path in paths)


def prepare_data(source_tif: Optional[Path], copy_data: bool, create_synthetic_if_missing: bool) -> Path:
    """确定本次要处理的输入影像。

    默认行为是扫描项目 `data/` 根目录：
    - 只有一张 tif/tiff 时直接使用；
    - 多张时要求用户用 `--source-tif` 明确指定；
    - 没有真实影像且启用 `--test-crop` 时生成合成测试图。

    如果用户传入外部路径，默认直接读取外部影像；加 `--copy-data` 后复制到本项目
    `data/` 目录，便于后续复现实验。
    """
    prepare_dirs()
    if source_tif is None:
        local_tifs = find_data_tifs()
        if len(local_tifs) == 1:
            log(f"Using project data image: {local_tifs[0]}")
            return local_tifs[0]
        if len(local_tifs) > 1:
            raise ValueError(
                "Found multiple GeoTIFF files under data/. Use --source-tif to choose one:\n"
                f"{format_tif_choices(local_tifs)}"
            )
        if create_synthetic_if_missing:
            log("No GeoTIFF found under data/. Creating a project-local synthetic test image.")
            return create_synthetic_image(SYNTHETIC_TIF)
        raise FileNotFoundError(
            "No .tif/.tiff file found directly under data/. "
            "Put the source GeoTIFF in data/ or pass --source-tif <path>."
        )

    source_tif = Path(source_tif)
    if source_tif.exists():
        local_tif = DATA_DIR / source_tif.name
        if same_path(source_tif, local_tif):
            return local_tif
        if not copy_data:
            log(f"Using source image directly: {source_tif}")
            return source_tif
        if local_tif.exists() and local_tif.stat().st_size == source_tif.stat().st_size:
            log(f"Data already copied: {local_tif}")
            return local_tif
        log(f"Copying image to project data: {local_tif}")
        shutil.copy2(source_tif, local_tif)
        return local_tif

    if create_synthetic_if_missing:
        log(f"Source image not found: {source_tif}. Creating a project-local synthetic test image.")
        return create_synthetic_image(SYNTHETIC_TIF)

    raise FileNotFoundError(f"Source image not found: {source_tif}")


def safe_name(value: str) -> str:
    """将任意名称转换成适合用作文件夹/文件名前缀的安全字符串。"""
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)
    return cleaned.strip("_") or "image"


def tile_cache_dir(image_path: Path, tile_size: int, overlap: int) -> Path:
    """生成瓦片缓存目录，目录名包含影像名、瓦片大小和重叠宽度。"""
    return OUTPUT_DIR / "tiles" / safe_name(f"{image_path.stem}_ts{tile_size}_ov{overlap}")


def tile_vector_path(image_path: Path, tile_size: int, overlap: int, tile_index: int) -> Path:
    """单个瓦片矢量结果的缓存路径。"""
    return tile_cache_dir(image_path, tile_size, overlap) / "vectors" / f"tile_{tile_index:05d}.gpkg"


def tile_empty_marker_path(image_path: Path, tile_size: int, overlap: int, tile_index: int) -> Path:
    """空瓦片标记路径，用于避免重复处理没有建筑物的瓦片。"""
    return tile_cache_dir(image_path, tile_size, overlap) / "vectors" / f"tile_{tile_index:05d}.empty.json"


def valid_model_file(path: Path, expected_bytes: int = 0) -> bool:
    """判断模型文件是否存在且大小合理，降低断点下载残留文件误用风险。"""
    if not path.exists() or not path.is_file():
        return False
    if expected_bytes <= 0:
        return path.stat().st_size > 0
    return path.stat().st_size >= expected_bytes * 0.95


def download_url(url: str, dst: Path, expected_bytes: int = 0, retries: int = 5) -> Path:
    """下载普通 URL 模型文件，支持 .part 临时文件和 Range 断点续传。"""
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
    """确保指定 SAM 权重存在，不存在时按需下载或复用相邻 sam/models 目录。"""
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
    """确保 GeoAI 建筑物模型存在，不存在时从 Hugging Face 下载。"""
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
    """根据用户选择的模型列表准备权重文件，并返回模型名到路径的映射。"""
    prepare_dirs()
    paths = {}
    for model in models:
        if model == "geoai":
            paths[model] = ensure_geoai_model(allow_download)
        elif model.startswith("sam_"):
            paths[model] = ensure_sam_model(model, allow_download)
    return paths


def make_test_crop(image_path: Path, tile_size: int) -> Path:
    """从影像中心裁剪一个 tile_size 大小的小图，用于快速验证流程。"""
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


def iter_tile_windows(image_path: Path, tile_size: int, overlap: int) -> List[Tuple[int, Any]]:
    """按固定瓦片大小和重叠宽度生成整幅影像的窗口列表。"""
    import rasterio
    from rasterio.windows import Window

    stride = max(1, tile_size - overlap)
    windows: List[Tuple[int, Any]] = []
    with rasterio.open(image_path) as src:
        idx = 0
        for row_off in range(0, src.height, stride):
            height = min(tile_size, src.height - row_off)
            if height <= 0:
                continue
            for col_off in range(0, src.width, stride):
                width = min(tile_size, src.width - col_off)
                if width <= 0:
                    continue
                windows.append((idx, Window(col_off, row_off, width, height)))
                idx += 1
    return windows


def write_tile(image_path: Path, window: Any, tile_path: Path) -> Path:
    """把原始影像的一个窗口写成独立 GeoTIFF，并保持空间参考和仿射变换。"""
    import rasterio

    tile_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(image_path) as src:
        data = src.read(window=window)
        profile = src.profile.copy()
        profile.update(
            width=int(window.width),
            height=int(window.height),
            transform=src.window_transform(window),
        )
        with rasterio.open(tile_path, "w", **profile) as dst:
            dst.write(data)
    return tile_path


def merge_tile_gdfs(gdfs: Sequence[Any], crs: Any, min_area: float, dedupe_iou: float) -> Any:
    """合并所有瓦片的建筑物矢量，并用 IoU 去除重叠区域重复多边形。"""
    import geopandas as gpd
    import pandas as pd

    valid = [gdf for gdf in gdfs if gdf is not None and len(gdf) > 0]
    if not valid:
        return empty_gdf_like(crs)

    merged = gpd.GeoDataFrame(pd.concat(valid, ignore_index=True), geometry="geometry", crs=valid[0].crs)
    if crs and merged.crs and merged.crs != crs:
        merged = merged.to_crs(crs)
    elif crs and merged.crs is None:
        merged = merged.set_crs(crs)

    merged["geometry"] = merged.geometry.make_valid()
    merged = merged[merged.geometry.notna() & ~merged.geometry.is_empty].copy()
    merged = add_area_m2(merged)
    if len(merged) > 0:
        merged = merged[merged["area_m2"] >= min_area].copy()
    if len(merged) <= 1:
        return merged.reset_index(drop=True)

    work = merged.sort_values("area_m2", ascending=False).reset_index(drop=True)
    keep_indices: List[int] = []
    dropped = set()
    spatial_index = work.sindex
    for idx, geom in enumerate(work.geometry):
        if idx in dropped:
            continue
        keep_indices.append(idx)
        for other_idx in spatial_index.intersection(geom.bounds):
            if other_idx <= idx or other_idx in dropped:
                continue
            other = work.geometry.iloc[other_idx]
            union_area = geom.union(other).area
            if union_area <= 0:
                continue
            if geom.intersection(other).area / union_area >= dedupe_iou:
                dropped.add(other_idx)
    return work.iloc[keep_indices].reset_index(drop=True)


def empty_gdf_like(crs: Any = None):
    """创建统一字段结构的空 GeoDataFrame，方便空结果继续走后续流程。"""
    import geopandas as gpd

    return gpd.GeoDataFrame(
        columns=["method", "model", "params", "area_m2", "regularized", "geometry"],
        geometry="geometry",
        crs=crs,
    )


def add_area_m2(gdf: Any) -> Any:
    """为矢量结果添加平方米面积字段；经纬度坐标会先估算 UTM 投影再算面积。"""
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
    """统一不同模型输出的字段、几何有效性和面积过滤规则。"""
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
    """动态加载相邻 building-regularize 项目的正则化流水线。"""
    path = REPO_ROOT / "building-regularize" / "backend" / "regularize.py"
    spec = importlib.util.spec_from_file_location("building_regularize_pipeline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load regularizer from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.RegularizeConfig, module.RegularizePipeline


def regularize_gdf(gdf: Any, min_area: float, simplify: float) -> Any:
    """调用正则化模块，让建筑物边界更规整，并保留原始 CRS。"""
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
    """保存单次运行的 raw.shp、regularized.shp 和 regularized.gpkg。"""
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


def extract_geoai_gdf(image_path: Path, model_path: Path, args: argparse.Namespace, params: Dict[str, Any]) -> Any:
    """调用 GeoAI BuildingFootprintExtractor，返回标准化后的建筑物 GeoDataFrame。"""
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
    return normalize_gdf(gdf, "geoai", "geoai_building", params, args.min_area)


def run_geoai(image_path: Path, model_path: Path, args: argparse.Namespace) -> RunResult:
    """直接对整幅影像运行 GeoAI；适合小图或测试裁剪图。"""
    start = time.time()
    run_dir = OUTPUT_DIR / "geoai"
    params = {"tile_size": args.tile_size, "overlap": args.overlap, "min_area": args.min_area}
    try:
        gdf = extract_geoai_gdf(image_path, model_path, args, params)
        raw_path, reg_path, gpkg_path, reg = save_outputs(gdf, run_dir, args.regularize, args.min_area, args.simplify)
        stats_gdf = reg if reg_path else gdf
        return build_result("geoai", "geoai_building", params, raw_path, reg_path, gpkg_path, stats_gdf, start, "ok")
    except Exception as exc:
        return build_result("geoai", "geoai_building", params, None, None, None, None, start, "failed", str(exc))


def run_geoai_tiled(image_path: Path, model_path: Path, args: argparse.Namespace) -> RunResult:
    """对大幅影像执行切片 GeoAI 推理，再合并为最终建筑物结果。"""
    import geopandas as gpd
    import rasterio

    start = time.time()
    run_dir = OUTPUT_DIR / "geoai_tiled"
    params = {
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "min_area": args.min_area,
        "tiled": True,
        "dedupe_iou": args.dedupe_iou,
    }
    try:
        windows = iter_tile_windows(image_path, args.tile_size, args.overlap)
        with rasterio.open(image_path) as src:
            crs = src.crs
        log(f"GeoAI tiled extraction: {len(windows)} tiles, tile_size={args.tile_size}, overlap={args.overlap}")

        tile_gdfs = []
        cache_dir = tile_cache_dir(image_path, args.tile_size, args.overlap)
        tile_dir = cache_dir / "rasters"
        for idx, window in windows:
            vector_path = tile_vector_path(image_path, args.tile_size, args.overlap, idx)
            empty_marker = tile_empty_marker_path(image_path, args.tile_size, args.overlap, idx)
            if vector_path.exists():
                log(f"GeoAI tile {idx + 1}/{len(windows)} already extracted: {vector_path.name}")
                tile_gdfs.append(gpd.read_file(vector_path))
                continue
            if empty_marker.exists():
                log(f"GeoAI tile {idx + 1}/{len(windows)} already extracted: empty")
                tile_gdfs.append(empty_gdf_like(crs))
                continue

            tile_path = tile_dir / f"tile_{idx:05d}.tif"
            write_tile(image_path, window, tile_path)
            log(f"GeoAI tile {idx + 1}/{len(windows)}: {tile_path.name}")
            tile_params = {**params, "tile_index": idx}
            try:
                tile_gdf = extract_geoai_gdf(tile_path, model_path, args, tile_params)
                if len(tile_gdf) > 0:
                    tile_gdf["tile_id"] = idx
                    vector_path.parent.mkdir(parents=True, exist_ok=True)
                    tile_gdf.to_file(vector_path, driver="GPKG")
                    log(f"GeoAI tile {idx + 1}/{len(windows)} saved: {vector_path}")
                else:
                    empty_marker.parent.mkdir(parents=True, exist_ok=True)
                    empty_marker.write_text(json.dumps({"tile_index": idx, "empty": True}), encoding="utf-8")
                tile_gdfs.append(tile_gdf)
            except Exception as exc:
                log(f"GeoAI tile {idx} failed: {exc}")

        gdf = merge_tile_gdfs(tile_gdfs, crs, args.min_area, args.dedupe_iou)
        raw_path, reg_path, gpkg_path, reg = save_outputs(gdf, run_dir, args.regularize, args.min_area, args.simplify)
        stats_gdf = reg if reg_path else gdf
        return build_result("geoai", "geoai_building_tiled", params, raw_path, reg_path, gpkg_path, stats_gdf, start, "ok")
    except Exception as exc:
        return build_result("geoai", "geoai_building_tiled", params, None, None, None, None, start, "failed", str(exc))


def combine_sam_masks(masks: Any, shape: Tuple[int, int]) -> Any:
    """把 SAM 返回的多个 mask 合并成一张二值建筑候选图。"""
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
    """根据面积、矩形度和长宽比过滤明显不像建筑物的 SAM 候选面。"""
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
    """读取影像前三个波段并拉伸为 uint8 RGB，供 SAM 自动分割使用。"""
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
    """运行 SAM 自动分割参数网格，并保存每组参数的矢量化结果。"""
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
    """根据输出矢量和耗时组装统一的 RunResult。"""
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
    """把所有模型运行结果写成 CSV，便于公众号文章或实验报告引用。"""
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
    """从成功结果中挑选一个默认最佳成果，优先选择 GeoAI 的正则化输出。"""
    ok = [r for r in results if r.status == "ok" and r.count > 0 and r.regularized_path]
    if not ok:
        return None
    geoai = [r for r in ok if r.method == "geoai"]
    if geoai:
        return max(geoai, key=lambda r: r.count)
    return max(ok, key=lambda r: (r.count, r.total_area_m2))


def copy_best(best: Optional[RunResult]) -> Optional[Path]:
    """把最佳正则化 Shapefile 复制到 outputs/compare/best_regularized.*。"""
    if best is None or best.regularized_path is None:
        log("No non-empty regularized result available for best_regularized.shp")
        return None
    target_base = OUTPUT_DIR / "compare" / "best_regularized"
    for src in best.regularized_path.parent.glob(best.regularized_path.stem + ".*"):
        shutil.copy2(src, target_base.with_suffix(src.suffix))
    log(f"Best result copied from {best.regularized_path}")
    return target_base.with_suffix(".shp")


def copy_shapefile_to_root(src_shp: Optional[Path], root_stem: str) -> Optional[Path]:
    """复制 Shapefile 及其配套文件到 outputs/ 根目录，方便用户直接找到成果。"""
    if src_shp is None or not src_shp.exists():
        return None
    target_base = OUTPUT_DIR / root_stem
    for src in src_shp.parent.glob(src_shp.stem + ".*"):
        if src.suffix.lower() == ".gpkg":
            continue
        shutil.copy2(src, target_base.with_suffix(src.suffix))
    return target_base.with_suffix(".shp")


def export_root_shapefiles(results: Sequence[RunResult], best: Optional[RunResult]) -> List[Path]:
    """为每个模型结果导出一份根目录 Shapefile 快捷副本。"""
    exported: List[Path] = []
    for item in results:
        safe_model = safe_name(item.model)
        safe_method = safe_name(item.method)
        raw = copy_shapefile_to_root(item.raw_path, f"{safe_model}_{safe_method}_raw")
        if raw:
            exported.append(raw)
        regularized = copy_shapefile_to_root(item.regularized_path, f"{safe_model}_{safe_method}_regularized")
        if regularized:
            exported.append(regularized)
    best_path = copy_shapefile_to_root(best.regularized_path if best else None, "best_regularized")
    if best_path:
        exported.append(best_path)
    if exported:
        log("Root Shapefile exports: " + ", ".join(str(path) for path in exported))
    return exported


def make_preview(image_path: Path, result: RunResult) -> Optional[Path]:
    """生成影像叠加建筑物边界的 PNG 预览图。"""
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
    """重新读取输出矢量并检查要素数、无效几何数量和坐标参考。"""
    import geopandas as gpd

    for item in results:
        for path in (item.raw_path, item.regularized_path):
            if path is None:
                continue
            gdf = gpd.read_file(path)
            invalid = 0 if len(gdf) == 0 else int((~gdf.geometry.is_valid).sum())
            log(f"Validated {path}: {len(gdf)} features, invalid geometries={invalid}, crs={gdf.crs}")


def build_parser() -> argparse.ArgumentParser:
    """定义命令行参数，所有默认值都集中在这里维护。"""
    parser = argparse.ArgumentParser(description="Extract and regularize building footprints from GeoTIFF imagery.")
    parser.add_argument(
        "--source-tif",
        type=Path,
        default=DEFAULT_SOURCE_TIF,
        help="Source GeoTIFF path. By default the script auto-detects one .tif/.tiff directly under data/.",
    )
    parser.add_argument("--copy-data", action="store_true", help="Copy source GeoTIFF into the project data/ directory.")
    parser.add_argument("--models", type=parse_models, default=parse_models("geoai"))
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-area", type=float, default=20.0)
    parser.add_argument("--simplify", type=float, default=0.5)
    add_bool_flag(parser, "regularize", True, "Run polygon regularization.")
    add_bool_flag(parser, "test-crop", False, "Run on a center crop for a quick smoke test.")
    add_bool_flag(parser, "tile-full-image", True, "Split full images into tiles, extract each tile, then merge results.")
    parser.add_argument("--dedupe-iou", type=float, default=0.6, help="IoU threshold for removing duplicate polygons from overlapping tiles.")
    add_bool_flag(parser, "download-models", True, "Download missing models into the shared workspace models directory.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--check-env", action="store_true", help="Only check Python dependencies.")
    add_bool_flag(parser, "validate", True, "Validate generated vector files.")
    add_bool_flag(parser, "preview", True, "Create PNG preview images.")
    return parser


def add_bool_flag(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    """添加同时支持 `--xxx` 和 `--no-xxx` 的布尔参数。"""
    attr = name.replace("-", "_")
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(f"--{name}", action=argparse.BooleanOptionalAction, default=default, help=help_text)
        return
    parser.set_defaults(**{attr: default})
    parser.add_argument(f"--{name}", dest=attr, action="store_true", help=help_text)
    parser.add_argument(f"--no-{name}", dest=attr, action="store_false", help=f"Disable: {help_text}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """命令行入口：检查环境、准备数据和模型、执行提取、写出成果。"""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.check_env:
        return check_env()

    # 主流程执行前先检查依赖，避免跑到一半才发现环境缺包。
    if check_env() != 0:
        return 2

    prepare_dirs()
    args.device = normalize_device(args.device)
    log(f"Using device: {args.device}")

    # 确定输入影像和模型权重。这里不会触发推理，只做路径和文件准备。
    image = prepare_data(args.source_tif, args.copy_data, create_synthetic_if_missing=args.test_crop)
    model_paths = ensure_models(args.models, args.download_models)

    # --test-crop 用于快速验证流程；正式运行时处理整幅影像。
    if args.test_crop:
        run_image = make_test_crop(image, args.tile_size)
        log(f"Processing test crop only: {run_image}")
    else:
        run_image = image
        log(f"Processing full image extent: {run_image}")

    results: List[RunResult] = []
    if "geoai" in args.models:
        # 大图默认走切片流程，小图/测试裁剪图可以直接整图推理。
        if args.tile_full_image and not args.test_crop:
            result = run_geoai_tiled(run_image, model_paths["geoai"], args)
        else:
            result = run_geoai(run_image, model_paths["geoai"], args)
        log(f"GeoAI result: {result.status}, count={result.count}, message={result.message}")
        results.append(result)

    for model in args.models:
        if model.startswith("sam_"):
            # SAM 作为对比模型执行，通常更适合小图或裁剪区域。
            for result in run_sam(run_image, model, model_paths[model], args):
                log(f"SAM result: {result.model}, {result.status}, count={result.count}, message={result.message}")
                results.append(result)

    # 汇总、挑选最佳成果、生成快捷 Shapefile 和预览图。
    write_summary(results)
    best = choose_best(results)
    copy_best(best)
    export_root_shapefiles(results, best)
    if args.preview:
        for item in results:
            make_preview(run_image, item)
    if args.validate:
        validate_outputs(results)

    # 如果所有模型都失败，返回非零退出码，便于脚本/CI 判断运行失败。
    failed = [r for r in results if r.status != "ok"]
    if failed and len(failed) == len(results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
