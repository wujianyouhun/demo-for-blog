"""
ResNet 图像分类推理示例
=======================

这个脚本可以：
1. 优先加载项目 models/hub/checkpoints/resnet50-11ad3fa6.pth 的 ImageNet ResNet50 权重；
2. 如果本地权重不存在，则自动下载 torchvision 官方 ImageNet 预训练 ResNet50；
3. 对普通图片、目录中的图片，或多波段 GeoTIFF 进行推理，并输出 Top-K 结果。

运行示例（在仓库根目录执行）：
    python resNet/resnet_inference_example.py --image path/to/image.jpg
    python resNet/resnet_inference_example.py --dir path/to/images --topk 3
    python resNet/resnet_inference_example.py --geotiff resNet/data.tif --bands 3 2 1

依赖：
    pip install torch torchvision pillow

说明：
    本示例默认输入 RGB 图片，并按照 ImageNet 的均值/标准差做归一化。
    如果通过 --checkpoint 使用自己训练的 EuroSAT 模型，模型 checkpoint 中应包含：
    model、model_name（可选）、class_names（可选）、image_size（可选）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
import torch
from PIL import Image
from torchvision import models, transforms


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_CACHE_DIR = REPO_ROOT / "models"
IMAGENET_WEIGHT_PATH = MODEL_CACHE_DIR / "hub" / "checkpoints" / "resnet50-11ad3fa6.pth"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# GeoAI 的全部模型统一保存在项目根目录 models；不存在时 torchvision 自动下载到此处。
torch.hub.set_dir(str(MODEL_CACHE_DIR / "hub"))


def choose_device(name: str) -> torch.device:
    """选择推理设备；auto 会优先使用可用的 CUDA。"""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_imagenet_model() -> tuple[torch.nn.Module, list[str], int]:
    """加载官方 ResNet50；本地缓存不存在时 torchvision 会自动下载并缓存。"""
    weights = models.ResNet50_Weights.DEFAULT
    if IMAGENET_WEIGHT_PATH.exists():
        print(f"加载本地 ImageNet ResNet50 权重: {IMAGENET_WEIGHT_PATH}")
    else:
        print("本地 ImageNet ResNet50 权重不存在，正在从 torchvision 官方地址下载...")
    model = models.resnet50(weights=weights)
    return model, list(weights.meta["categories"]), 224


def build_local_model(checkpoint_path: Path, device: torch.device):
    """加载本项目训练脚本保存的 checkpoint。"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get("model_name", "resnet50")
    class_names = checkpoint.get("class_names")
    image_size = checkpoint.get("image_size", 224)

    # 复用训练脚本中的模型构造函数，保证结构与训练时一致。
    import sys
    sys.path.insert(0, str(checkpoint_path.parents[3] / "Classification" / "scripts"))
    from train import build_model

    if not class_names:
        class_names = [str(i) for i in range(checkpoint["model"]["fc.weight"].shape[0])]
    model = build_model(model_name, len(class_names))
    model.load_state_dict(checkpoint["model"])
    return model, list(class_names), image_size


def load_model(checkpoint_path: Path | None, device: torch.device):
    """显式提供分类 checkpoint 时加载它，否则使用或下载 ImageNet ResNet50。"""
    if checkpoint_path is not None and checkpoint_path.exists():
        print(f"加载本地模型: {checkpoint_path}")
        return build_local_model(checkpoint_path, device)
    if checkpoint_path is not None:
        print(f"未找到指定分类模型: {checkpoint_path}")
    return build_imagenet_model()


def predict(image_path: Path, model, class_names: list[str], image_size: int, device: torch.device, topk: int):
    """读取一张图片并返回 Top-K 分类结果。"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    image = Image.open(image_path).convert("RGB")
    batch = transform(image).unsqueeze(0).to(device)

    return predict_batch(batch, model, class_names, topk)


def predict_batch(batch: torch.Tensor, model, class_names: list[str], topk: int):
    """对已标准化的单张 NCHW Tensor 计算 Top-K。"""
    with torch.inference_mode():
        probabilities = torch.softmax(model(batch), dim=1)[0]
    scores, indices = probabilities.topk(min(topk, len(class_names)))
    return [{"class_id": int(i), "class_name": class_names[int(i)], "probability": round(float(s), 6)}
            for s, i in zip(scores.cpu(), indices.cpu())]


def predict_geotiff(geotiff_path: Path, bands: tuple[int, int, int], scale: float, offset: float,
                    model, class_names: list[str], image_size: int, device: torch.device, topk: int):
    """选取 GeoTIFF 的三个波段，直接降采样到模型大小后进行整幅影像分类测试。"""
    with rasterio.open(geotiff_path) as src:
        if min(bands) < 1 or max(bands) > src.count:
            raise ValueError(f"请求波段 {bands} 超出影像有效范围 1-{src.count}")
        # 读取时降采样，不会把 data.tif 的全部像元读入内存。
        values = src.read(indexes=bands, out_shape=(3, image_size, image_size),
                          resampling=Resampling.bilinear, masked=True).filled(0).astype("float32")
        metadata = {
            "source_bands": list(bands), "source_shape": [src.height, src.width],
            "crs": str(src.crs), "bounds": list(src.bounds),
        }
    reflectance = np.clip(values * scale + offset, 0.0, 1.0)
    batch = torch.from_numpy(reflectance).unsqueeze(0)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return predict_batch(((batch - mean) / std).to(device), model, class_names, topk), metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="ResNet 单张/批量图像推理示例")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=Path, help="单张图片路径")
    group.add_argument("--dir", type=Path, help="图片目录，会递归搜索常见图片格式")
    group.add_argument("--geotiff", type=Path, help="多波段 GeoTIFF；用于整幅影像缩放后的单图测试")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="可选：自己训练的 ResNet 分类 checkpoint；不填则使用或下载 ImageNet ResNet50")
    parser.add_argument("--device", default="auto", help="auto、cpu 或 cuda")
    parser.add_argument("--topk", type=int, default=5, help="输出前几个类别")
    parser.add_argument("--bands", type=int, nargs=3, default=(3, 2, 1), metavar=("R", "G", "B"),
                        help="--geotiff 的 RGB 波段序号，默认 3 2 1")
    parser.add_argument("--scale", type=float, default=0.0001,
                        help="--geotiff 的 DN 缩放系数；Sentinel-2 L2A 常用 0.0001")
    parser.add_argument("--offset", type=float, default=0.0,
                        help="--geotiff 的反射率偏移量；BOA_ADD_OFFSET=-1000 时请设为 -0.1")
    parser.add_argument("--output", type=Path, help="可选：将 JSON 结果保存到文件")
    args = parser.parse_args()

    device = choose_device(args.device)
    model, class_names, image_size = load_model(args.checkpoint, device)
    model.to(device).eval()
    results = []
    if args.geotiff:
        topk, metadata = predict_geotiff(args.geotiff, tuple(args.bands), args.scale, args.offset,
                                          model, class_names, image_size, device, args.topk)
        results.append({"geotiff": str(args.geotiff), **metadata, "topk": topk})
        best = topk[0]
        print(f"\n{args.geotiff}\n  预测: {best['class_name']} ({best['probability']:.2%})")
        print(f"  RGB 波段: {args.bands}，原始尺寸: {metadata['source_shape']}，模型输入: {image_size}×{image_size}")
        for rank, item in enumerate(topk, 1):
            print(f"  Top{rank}: {item['class_name']:<28} {item['probability']:.2%}")
    else:
        image_paths = [args.image] if args.image else sorted(
            p for p in args.dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        for path in image_paths:
            try:
                topk = predict(path, model, class_names, image_size, device, args.topk)
                result = {"image": str(path), "topk": topk}
                results.append(result)
                best = topk[0]
                print(f"\n{path}\n  预测: {best['class_name']} ({best['probability']:.2%})")
                for rank, item in enumerate(topk, 1):
                    print(f"  Top{rank}: {item['class_name']:<28} {item['probability']:.2%}")
            except Exception as exc:
                results.append({"image": str(path), "error": str(exc)})
                print(f"\n推理失败: {path}\n  原因: {exc}")

    if args.output:
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已保存: {args.output}")


if __name__ == "__main__":
    main()
