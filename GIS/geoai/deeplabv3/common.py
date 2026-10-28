"""DeepLabV3-ResNet50 共用的模型、预处理和 checkpoint 约定。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT.parent / "models" / "deeplabv3"
DEFAULT_CHECKPOINT = MODEL_DIR / "remote_sensing_best.pth"
WEIGHTS = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IGNORE = 255


def choose_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA 不可用，请使用 --device cpu。")
    return device


def pretrained_model():
    """权重直接缓存到 models/deeplabv3，不使用用户目录下的默认缓存。"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    state = torch.hub.load_state_dict_from_url(
        WEIGHTS.url, model_dir=str(MODEL_DIR), map_location="cpu", check_hash=True,
    )
    model = deeplabv3_resnet50(weights=None, weights_backbone=None, aux_loss=True)
    model.load_state_dict(state)
    return model


def training_model(num_classes: int, pretrained: bool = True, in_channels: int = 3, aspp_rates=None):
    if not 2 <= num_classes <= 255:
        raise ValueError("类别数必须在 2–255 之间；255 保留给无效像元。")
    # 无 aux head 的模型与下面的遥感 checkpoint 完全一致。
    model = deeplabv3_resnet50(
        weights=None, weights_backbone=None, num_classes=num_classes, aux_loss=False,
    )
    if aspp_rates is not None:
        if len(aspp_rates) != 3 or any(not isinstance(r, int) or r < 1 for r in aspp_rates):
            raise ValueError("aspp_rates 需要三个正整数。")
        for rate, branch in zip(aspp_rates, model.classifier[0].convs[1:4]):
            branch[0].dilation = (rate, rate)
            branch[0].padding = (rate, rate)
    if in_channels not in (3, 4):
        raise ValueError("当前支持明确指定的 3 或 4 个输入波段。")
    if in_channels != 3:
        original = model.backbone.conv1
        model.backbone.conv1 = torch.nn.Conv2d(
            in_channels, original.out_channels, original.kernel_size, original.stride,
            original.padding, bias=False,
        )
    if pretrained:
        source = pretrained_model().state_dict()
        target = model.state_dict()
        compatible = {
            k: v for k, v in source.items()
            if k in target and v.shape == target[k].shape and not k.startswith("classifier.4.")
        }
        model.load_state_dict(compatible, strict=False)
        if in_channels == 4:
            # 保留三通道预训练结构，额外通道以 RGB 卷积核均值初始化。
            rgb = source["backbone.conv1.weight"]
            expanded = torch.cat((rgb, rgb.mean(1, keepdim=True)), dim=1) * (3 / 4)
            with torch.no_grad():
                model.backbone.conv1.weight.copy_(expanded)
    return model


def normalization_for_channels(channels: int) -> dict:
    if channels not in (3, 4):
        raise ValueError("归一化仅支持三通道或四通道。")
    return dict(mean=list(MEAN) + ([sum(MEAN) / 3] if channels == 4 else []),
                std=list(STD) + ([sum(STD) / 3] if channels == 4 else []))


def to_tensor(image: np.ndarray, scale: float, normalization: dict | None = None) -> torch.Tensor:
    """CHW 固定缩放；RGB 默认 ImageNet 参数，四通道实验保存扩展参数。"""
    if image.ndim != 3 or not np.isfinite(scale) or scale <= 0 or image.shape[0] not in (3, 4):
        raise ValueError("必须提供 3 或 4 个有序波段和大于 0 的有限 scale。")
    normalization = normalization or normalization_for_channels(image.shape[0])
    mean, std = np.asarray(normalization["mean"]), np.asarray(normalization["std"])
    if (mean.shape != (image.shape[0],) or std.shape != mean.shape
            or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any()):
        raise ValueError("normalization 的 mean/std 必须与通道数一致且 std 为正。")
    values = np.clip(np.nan_to_num(image.astype("float32") / scale), 0, 1)
    tensor = torch.from_numpy(np.ascontiguousarray(values))
    return (tensor - tensor.new_tensor(mean)[:, None, None]) / tensor.new_tensor(std)[:, None, None]


def load_checkpoint(path: Path, device: torch.device):
    if not path.is_file():
        raise FileNotFoundError(f"未找到遥感权重：{path}。请先运行 train_remote_sensing.py。")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "deeplabv3_resnet50":
        raise ValueError("需要本示例训练脚本保存的 DeepLabV3-ResNet50 checkpoint。")
    names = checkpoint["class_names"]
    bands, scale = checkpoint["bands"], checkpoint["scale"]
    channels = checkpoint.get("in_channels", 3)
    if (channels not in (3, 4) or len(bands) != channels or min(bands) < 1
            or not np.isfinite(scale) or scale <= 0):
        raise ValueError("checkpoint 中的 bands / scale 无效。")
    to_tensor(np.zeros((channels, 1, 1)), scale, checkpoint.get("normalization"))
    model = training_model(len(names), pretrained=False, in_channels=channels,
                           aspp_rates=checkpoint.get("aspp_rates"))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), checkpoint


def palette(count: int) -> list[int]:
    """VOC 位运算调色板，可覆盖全部 uint8 类别。"""
    result = []
    for index in range(count):
        r = g = b = 0
        value = index
        for shift in range(8):
            r |= (value & 1) << (7 - shift)
            g |= ((value >> 1) & 1) << (7 - shift)
            b |= ((value >> 2) & 1) << (7 - shift)
            value >>= 3
        result.extend((r, g, b))
    return result
