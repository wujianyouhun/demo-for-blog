"""加载五分类 U-Net 权重，对整幅遥感影像滑窗推理并输出分类与概率 GeoTIFF。"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.features import shapes
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling, calculate_default_transform
from rasterio.windows import Window

from unet_models import build_model


ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE = ROOT / "DATA" / "Qilian.tif"
DEFAULT_WEIGHTS = ROOT.parent / "models" / "Unet" / "qilian_multiclass_5class"
DEFAULT_OUTPUT = ROOT / "out" / "qilian_multiclass_5class"
IGNORE_VALUE = 255
# classification.tif 内嵌的类别色表。GIS 支持 GeoTIFF 色表时会自动彩色显示。
CLASS_COLORS = {
    0: (220, 50, 47, 255),    # road：红色
    1: (255, 165, 0, 255),    # built_up：橙色
    2: (76, 175, 80, 255),    # cropland：绿色
    3: (156, 112, 69, 255),   # bare_land：棕色
    4: (33, 150, 243, 255),   # solar_farm：蓝色
    255: (0, 0, 0, 0),        # NoData：透明
}

TARGET_EXTRACTS = {
    # 当前五分类样本中，建筑对应 built_up / 居民地。
    "building": {
        "class_names": ("built_up",),
        "mask": "building_mask.tif",
        "probability": "building_probability.tif",
        "note": "building 使用 class_id=1 / built_up / 居民地。",
        "colors": {0: (0, 0, 0, 0), 1: (255, 165, 0, 255), 255: (0, 0, 0, 0)},
    },
    # 当前样本没有单独的 vegetation 类，这里先用 cropland 作为绿色地物提取。
    "vegetation": {
        "class_names": ("cropland",),
        "mask": "vegetation_mask.tif",
        "probability": "vegetation_probability.tif",
        "note": "vegetation 当前映射到 class_id=2 / cropland / 耕地；若要真正植被类，需要新增样本并重训。",
        "colors": {0: (0, 0, 0, 0), 1: (76, 175, 80, 255), 255: (0, 0, 0, 0)},
    },
}


def parse_args(model_name: str, defaults: dict) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"{model_name} 五分类整图推理")
    parser.add_argument("--input", type=Path, default=defaults["input"], help="默认 Unet/DATA/Qilian.tif")
    parser.add_argument("--weights", type=Path, default=defaults["weights"], help="默认使用本模型 best.pth")
    parser.add_argument("--output-dir", type=Path, default=None, help="模型结果目录")
    parser.add_argument("--tile-size", type=int, default=defaults["tile_size"])
    parser.add_argument("--stride", type=int, default=defaults["stride"], help="小于 tile-size 时采用重叠平均")
    parser.add_argument("--target-crs", default=defaults["target_crs"])
    parser.add_argument("--target-resolution", type=float, default=defaults["target_resolution"])
    parser.add_argument("--device", default=defaults["device"], help="auto、cpu 或 cuda")
    parser.add_argument(
        "--task",
        choices=("multiclass", "building", "vegetation", "both"),
        default="multiclass",
        help="multiclass 输出五分类；building/vegetation/both 额外输出二值提取结果。",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖同名输出目录")
    parser.add_argument("--no-vectors", action="store_true", help="不导出 GeoPackage 矢量结果")
    parser.add_argument("--min-vector-area", type=float, default=0.0,
                        help="删除面积小于此值（平方米）的面要素，默认保留全部")
    args = parser.parse_args()
    args.model_name = model_name
    args.output_dir = args.output_dir or defaults["output"]
    if args.task != "multiclass" and args.output_dir == defaults["output"]:
        args.output_dir = args.output_dir / args.task
    if args.tile_size < 16 or args.tile_size % 16 or not 0 < args.stride <= args.tile_size:
        parser.error("tile-size 必须能被 16 整除，stride 必须位于 1 到 tile-size。")
    if args.target_resolution <= 0:
        parser.error("target-resolution 必须为正数。")
    if args.min_vector_area < 0:
        parser.error("--min-vector-area 不能小于 0。")
    return args


def choose_device(value: str) -> torch.device:
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else ("cpu" if value == "auto" else value))


def normalise(image: np.ndarray) -> np.ndarray:
    """必须与训练阶段的 per_tile_p99 归一化保持一致。"""
    image = np.nan_to_num(image.astype("float32"), nan=0.0)
    upper = np.percentile(image, 99, axis=(1, 2), keepdims=True)
    return np.clip(image / np.maximum(upper, 1e-6), 0.0, 1.0)


def load_model(args: argparse.Namespace, device: torch.device):
    if not args.weights.is_file():
        raise FileNotFoundError(f"找不到权重：{args.weights}。请先运行 train_multiclass_models.py。")
    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    if checkpoint.get("architecture") != args.model_name:
        raise ValueError(f"权重架构为 {checkpoint.get('architecture')}，不能用于 {args.model_name}。")
    model = build_model(args.model_name, in_channels=int(checkpoint["in_channels"]),
                        base=int(checkpoint["base_channels"]), num_classes=int(checkpoint["num_classes"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def build_vrt(path: Path, target_crs: str, resolution: float):
    source = rasterio.open(path)
    if source.crs is None:
        source.close()
        raise ValueError("输入影像没有 CRS。")
    transform, width, height = calculate_default_transform(
        source.crs, target_crs, source.width, source.height, *source.bounds,
        resolution=(resolution, resolution),
    )
    vrt = WarpedVRT(source, crs=target_crs, transform=transform, width=width, height=height,
                    resampling=Resampling.bilinear)
    return source, vrt


def output_starts(length: int, stride: int) -> list[int]:
    return list(range(0, length, stride))


def task_names(task: str) -> tuple[str, ...]:
    if task == "both":
        return ("building", "vegetation")
    if task == "multiclass":
        return ()
    return (task,)


def write_vector(
    values: np.ndarray,
    profile: dict,
    output_path: Path,
    *,
    valid: np.ndarray,
    class_names: tuple[str, ...] | list[str] | None,
    min_area: float,
    layer: str,
) -> str:
    """把分类或二值栅格转为带 CRS 的 GeoPackage 面图层。

    ``rasterio.features.shapes`` 直接复用推理输出的 transform，因此生成的面与
    classification.tif 像元严格对齐；这里不做平滑或拓扑修复，以避免无意改变模型结果。
    """
    try:
        import geopandas as gpd
        from shapely.geometry import shape
    except ImportError as exc:
        raise SystemExit("导出矢量结果需要 geopandas 和 shapely。请安装 requirements.txt 中的依赖后重试。") from exc

    records = []
    for geometry, value in shapes(values.astype("int16"), mask=valid, transform=profile["transform"]):
        class_id = int(value)
        record = {"class_id": class_id, "geometry": shape(geometry)}
        if class_names is not None:
            record["class_name"] = class_names[class_id]
        records.append(record)
    columns = ["class_id", *( ["class_name"] if class_names is not None else [] ), "geometry"]
    vector = gpd.GeoDataFrame(records, columns=columns, crs=profile["crs"])
    if min_area:
        vector = vector[vector.geometry.area >= min_area].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vector.to_file(output_path, layer=layer, driver="GPKG")
    return str(output_path)


def write_target_extracts(args: argparse.Namespace, checkpoint: dict, probability_sum: np.ndarray,
                          classification: np.ndarray, profile: dict) -> dict[str, dict[str, str]]:
    class_names = list(checkpoint["class_names"])
    outputs = {}
    for target in task_names(args.task):
        config = TARGET_EXTRACTS[target]
        class_ids = []
        for class_name in config["class_names"]:
            if class_name not in class_names:
                raise ValueError(f"权重中没有类别 {class_name!r}，无法输出 {target}。现有类别：{class_names}")
            class_ids.append(class_names.index(class_name))

        target_mask = np.isin(classification, class_ids).astype("uint8")
        target_probability = probability_sum[class_ids].sum(axis=0).astype("float32")

        mask_profile = profile.copy()
        mask_profile.update(dtype="uint8", nodata=255)
        mask_path = args.output_dir / config["mask"]
        with rasterio.open(mask_path, "w", **mask_profile) as destination:
            destination.write(target_mask, 1)
            destination.write_colormap(1, config["colors"])

        probability_profile = profile.copy()
        probability_profile.update(dtype="float32", nodata=np.nan)
        probability_path = args.output_dir / config["probability"]
        with rasterio.open(probability_path, "w", **probability_profile) as destination:
            destination.write(target_probability, 1)

        vector_path = None
        if not args.no_vectors:
            vector_path = write_vector(
                target_mask,
                profile,
                args.output_dir / f"{target}_vector.gpkg",
                valid=target_mask == 1,
                class_names=("background", target),
                min_area=args.min_vector_area,
                layer=target,
            )

        outputs[target] = {
            "mask": str(mask_path),
            "probability": str(probability_path),
            "vector": vector_path,
            "source_class_ids": class_ids,
            "source_class_names": list(config["class_names"]),
            "note": config["note"],
        }
    return outputs


def run_inference(args: argparse.Namespace) -> None:
    if not args.input.is_file():
        raise FileNotFoundError(f"找不到输入影像：{args.input}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"输出目录已存在：{args.output_dir}。确认后增加 --overwrite。")
        try:
            shutil.rmtree(args.output_dir)
        except PermissionError as exc:
            raise SystemExit("输出文件正在被 GIS 打开。请关闭对应图层后重试，或改用 --output-dir。") from exc
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    model, checkpoint = load_model(args, device)
    source, vrt = build_vrt(args.input, args.target_crs, args.target_resolution)
    cache_dir = Path(tempfile.mkdtemp(prefix="unet_prob_", dir=args.output_dir))
    probability_sum = None
    visits = None
    try:
        if vrt.count < checkpoint["in_channels"]:
            raise ValueError(f"输入仅有 {vrt.count} 个波段，权重需要 {checkpoint['in_channels']} 个波段。")
        num_classes = int(checkpoint["num_classes"])
        height, width = vrt.height, vrt.width
        probability_sum = np.memmap(cache_dir / "probabilities.dat", mode="w+", dtype="float32",
                                    shape=(num_classes, height, width))
        probability_sum[:] = 0
        visits = np.memmap(cache_dir / "visits.dat", mode="w+", dtype="float32", shape=(height, width))
        visits[:] = 0
        rows, cols = output_starts(height, args.stride), output_starts(width, args.stride)
        total, finished = len(rows) * len(cols), 0
        print(f"推理 {args.model_name}：{width}×{height}，{total} 个窗口，设备={device}")
        for row in rows:
            for col in cols:
                valid_height, valid_width = min(args.tile_size, height - row), min(args.tile_size, width - col)
                window = Window(col, row, valid_width, valid_height)
                patch = vrt.read(indexes=list(range(1, checkpoint["in_channels"] + 1)), window=window)
                padded = np.zeros((checkpoint["in_channels"], args.tile_size, args.tile_size), dtype="float32")
                padded[:, :valid_height, :valid_width] = patch
                with torch.inference_mode():
                    logits = model(torch.from_numpy(normalise(padded)[None]).to(device))
                    probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
                probability_sum[:, row:row + valid_height, col:col + valid_width] += probabilities[:, :valid_height, :valid_width]
                visits[row:row + valid_height, col:col + valid_width] += 1
                finished += 1
                if finished % 50 == 0 or finished == total:
                    print(f"  {finished}/{total} ({finished / total:.0%})")
        probability_sum /= np.maximum(visits[None], 1)
        classification = np.argmax(probability_sum, axis=0).astype("uint8")
        # ``vrt.profile`` 的 driver 是 VRT（只读虚拟栅格）；写出时必须明确改为
        # GTiff，否则 Rasterio 会报 “Writing through VRTSourcedRasterBand”。
        profile = vrt.profile.copy()
        profile.update(
            driver="GTiff", height=height, width=width, transform=vrt.transform,
            crs=vrt.crs, count=1, dtype="uint8", nodata=IGNORE_VALUE,
            compress="deflate",
        )
        classification_path = args.output_dir / "classification.tif"
        with rasterio.open(classification_path, "w", **profile) as destination:
            destination.write(classification, 1)
            destination.write_colormap(1, CLASS_COLORS)
        classification_vector = None
        if not args.no_vectors:
            classification_vector = write_vector(
                classification,
                profile,
                args.output_dir / "classification_vector.gpkg",
                valid=np.ones(classification.shape, dtype=bool),
                class_names=list(checkpoint["class_names"]),
                min_area=args.min_vector_area,
                layer="classification",
            )
        probability_paths = {}
        profile.update(dtype="float32", nodata=np.nan)
        for index, class_name in enumerate(checkpoint["class_names"]):
            path = args.output_dir / f"probability_{index}_{class_name}.tif"
            with rasterio.open(path, "w", **profile) as destination:
                destination.write(probability_sum[index], 1)
            probability_paths[class_name] = str(path)
        target_outputs = write_target_extracts(args, checkpoint, probability_sum, classification, profile)
        report = {
            "model": args.model_name, "input": str(args.input), "weights": str(args.weights),
            "classification": str(classification_path), "classification_vector": classification_vector,
            "probabilities": probability_paths,
            "task": args.task, "target_outputs": target_outputs,
            "crs": str(vrt.crs), "resolution": list(vrt.res), "shape": [height, width],
            "class_names": list(checkpoint["class_names"]), "normalisation": checkpoint.get("normalisation"),
            "class_id_mapping": {str(index): name for index, name in enumerate(checkpoint["class_names"])},
            "class_colors_rgba": {str(index): CLASS_COLORS[index] for index in range(num_classes)},
        }
        (args.output_dir / "inference_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"完成：{classification_path}")
    finally:
        # Windows 上 numpy.memmap 存活时文件句柄仍被占用，必须在删临时目录前
        # 主动释放映射；否则每次推理都会在输出目录遗留 unet_prob_* 缓存。
        if probability_sum is not None:
            probability_sum.flush()
            del probability_sum
        if visits is not None:
            visits.flush()
            del visits
        gc.collect()
        vrt.close()
        source.close()
        shutil.rmtree(cache_dir, ignore_errors=True)


def main(
    model_name: str,
    *,
    input_path: Path = DEFAULT_IMAGE,
    weights_path: Path | None = None,
    output_dir: Path | None = None,
    tile_size: int = 256,
    stride: int = 192,
    target_crs: str = "EPSG:3857",
    target_resolution: float = 10.0,
    device: str = "auto",
) -> None:
    """以调用脚本给出的可见默认参数启动；命令行参数仍可覆盖它们。"""
    defaults = {
        "input": input_path,
        "weights": weights_path or DEFAULT_WEIGHTS / model_name / "best.pth",
        "output": output_dir or DEFAULT_OUTPUT / model_name,
        "tile_size": tile_size,
        "stride": stride,
        "target_crs": target_crs,
        "target_resolution": target_resolution,
        "device": device,
    }
    run_inference(parse_args(model_name, defaults))
