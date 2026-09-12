"""五个 U-Net 模型共用的批量 GeoTIFF 推理流程。"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
import torch

from unet_models import build_model

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "out"
MODEL_DIR = ROOT.parent / "models" / "Unet"
RASTER_SUFFIXES = {".tif", ".tiff"}


def choose_device(value: str) -> torch.device:
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else ("cpu" if value == "auto" else value))


def checkpoint_path(model_name: str, task: str) -> Path:
    return MODEL_DIR / f"{model_name}_{task}.pth"


def fetch_weights(path: Path, url: str | None, sha256: str | None) -> None:
    if path.exists():
        print(f"使用本地模型：{path}")
        return
    if not url:
        raise FileNotFoundError(f"未找到 {path}。请放入 models/Unet，或提供对应任务的模型下载地址。")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".download")
    print(f"模型不存在，开始下载：{url}")
    urllib.request.urlretrieve(url, temporary)
    if sha256 and hashlib.sha256(temporary.read_bytes()).hexdigest().lower() != sha256.lower():
        temporary.unlink(missing_ok=True)
        raise ValueError("下载模型的 SHA256 校验失败，文件未保留。")
    temporary.replace(path)


def normalise(image: np.ndarray) -> np.ndarray:
    image = np.nan_to_num(image.astype("float32"), nan=0.0)
    high = np.percentile(image, 99, axis=(1, 2), keepdims=True)
    return np.clip(image / np.maximum(high, 1e-6), 0.0, 1.0)


def read_image(path: Path, bands: list[int]) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as source:
        if max(bands) > source.count:
            raise ValueError(f"{path} 只有 {source.count} 个波段，但模型需要 {bands}。")
        return source.read(bands).astype("float32"), source.profile.copy()


def load_model(model_name: str, task: str, url: str | None, sha256: str | None, target_device: torch.device):
    path = checkpoint_path(model_name, task)
    fetch_weights(path, url, sha256)
    checkpoint = torch.load(path, map_location=target_device, weights_only=False)
    if checkpoint.get("model") != model_name or checkpoint.get("task") != task:
        raise ValueError(f"{path.name} 与当前 {model_name}/{task} 不匹配。")
    bands = list(checkpoint["bands"])
    model = build_model(model_name, len(bands), checkpoint.get("base", 32)).to(target_device)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), bands


def predict_one(image_path: Path, output_path: Path, model, bands: list[int], args, target_device: torch.device) -> dict:
    image, profile = read_image(image_path, bands)
    image = normalise(image)
    height, width = image.shape[1:]
    probability = np.zeros((height, width), dtype=np.float32)
    visits = np.zeros((height, width), dtype=np.float32)
    for row in range(0, height, args.stride):
        for col in range(0, width, args.stride):
            patch = image[:, row:min(row + args.tile_size, height), col:min(col + args.tile_size, width)]
            patch = np.pad(patch, ((0, 0), (0, args.tile_size - patch.shape[1]), (0, args.tile_size - patch.shape[2])))
            with torch.inference_mode():
                score = torch.sigmoid(model(torch.from_numpy(patch[None]).to(target_device)))[0, 0].cpu().numpy()
            patch_height, patch_width = min(args.tile_size, height - row), min(args.tile_size, width - col)
            probability[row:row + patch_height, col:col + patch_width] += score[:patch_height, :patch_width]
            visits[row:row + patch_height, col:col + patch_width] += 1
    probability /= np.maximum(visits, 1)
    mask = (probability >= args.threshold).astype("uint8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile.update(count=1, dtype="uint8", nodata=0, compress="deflate")
    with rasterio.open(output_path, "w", **profile) as destination:
        destination.write(mask, 1)
    probability_path = output_path.with_name(output_path.stem + "_probability.tif")
    profile.update(dtype="float32", nodata=np.nan)
    with rasterio.open(probability_path, "w", **profile) as destination:
        destination.write(probability, 1)
    return {"input": str(image_path), "mask": str(output_path), "probability": str(probability_path)}


def find_images(data_dir: Path) -> list[Path]:
    ignored = {"mask", "masks", "label", "labels", "out", "output", "outputs"}
    images = [p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in RASTER_SUFFIXES
              and not any(part.lower() in ignored for part in p.relative_to(data_dir).parts)]
    if not images:
        raise FileNotFoundError(f"{data_dir} 中未找到待推理的 .tif 或 .tiff 影像。")
    return sorted(images)


def main(model_name: str, *, default_task: str = "both", default_data_dir: Path = DATA_DIR,
         default_out_dir: Path = OUTPUT_DIR, default_tile_size: int = 512,
         default_stride: int = 384, default_threshold: float = 0.5,
         default_device: str = "auto") -> None:
    parser = argparse.ArgumentParser(description=f"{model_name} 批量建筑/植被提取")
    parser.add_argument("--task", choices=("building", "vegetation", "both"), default=default_task)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument("--out-dir", type=Path, default=default_out_dir)
    parser.add_argument("--tile-size", type=int, default=default_tile_size)
    parser.add_argument("--stride", type=int, default=default_stride)
    parser.add_argument("--threshold", type=float, default=default_threshold)
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--building-weights-url")
    parser.add_argument("--vegetation-weights-url")
    parser.add_argument("--building-sha256")
    parser.add_argument("--vegetation-sha256")
    args = parser.parse_args()
    if args.tile_size < 16 or args.stride <= 0 or args.stride > args.tile_size:
        parser.error("--tile-size 至少为 16，且 --stride 必须位于 1 到 tile-size 之间。")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold 必须在 0 到 1 之间。")
    images = find_images(args.data_dir)
    target_device = choose_device(args.device)
    tasks = ("building", "vegetation") if args.task == "both" else (args.task,)
    records = []
    for task in tasks:
        url, digest = ((args.building_weights_url, args.building_sha256) if task == "building"
                       else (args.vegetation_weights_url, args.vegetation_sha256))
        model, bands = load_model(model_name, task, url, digest, target_device)
        for image_path in images:
            relative = image_path.relative_to(args.data_dir).with_suffix("")
            output_path = args.out_dir / model_name / task / relative.parent / f"{relative.name}_mask.tif"
            records.append({"model": model_name, "task": task,
                            **predict_one(image_path, output_path, model, bands, args, target_device)})
            print(f"完成 {model_name}/{task}：{image_path.name}")
    report = args.out_dir / model_name / "inference_report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"全部完成，结果目录：{report.parent}")
