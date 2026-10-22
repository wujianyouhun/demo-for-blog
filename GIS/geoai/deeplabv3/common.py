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


def training_model(num_classes: int, pretrained: bool = True):
    if not 2 <= num_classes <= 255:
        raise ValueError("类别数必须在 2–255 之间；255 保留给无效像元。")
    # 无 aux head 的模型与下面的遥感 checkpoint 完全一致。
    model = deeplabv3_resnet50(
        weights=None, weights_backbone=None, num_classes=num_classes, aux_loss=False,
    )
    if pretrained:
        source = pretrained_model().state_dict()
        target = model.state_dict()
        compatible = {
            k: v for k, v in source.items()
            if k in target and v.shape == target[k].shape and not k.startswith("classifier.4.")
        }
        model.load_state_dict(compatible, strict=False)
    return model


def to_tensor(image: np.ndarray, scale: float) -> torch.Tensor:
    """输入 CHW，先按固定 scale 缩放至 [0,1]，再做 ImageNet 归一化。"""
    if not np.isfinite(scale) or scale <= 0 or image.shape[0] != 3:
        raise ValueError("必须提供 3 个有序 RGB 波段和大于 0 的有限 scale。")
    values = np.clip(np.nan_to_num(image.astype("float32") / scale), 0, 1)
    tensor = torch.from_numpy(np.ascontiguousarray(values))
    return (tensor - tensor.new_tensor(MEAN)[:, None, None]) / tensor.new_tensor(STD)[:, None, None]


def load_checkpoint(path: Path, device: torch.device):
    if not path.is_file():
        raise FileNotFoundError(f"未找到遥感权重：{path}。请先运行 train_remote_sensing.py。")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "deeplabv3_resnet50":
        raise ValueError("需要本示例训练脚本保存的 DeepLabV3-ResNet50 checkpoint。")
    names = checkpoint["class_names"]
    bands, scale = checkpoint["bands"], checkpoint["scale"]
    if len(bands) != 3 or min(bands) < 1 or not np.isfinite(scale) or scale <= 0:
        raise ValueError("checkpoint 中的 bands / scale 无效。")
    model = training_model(len(names), pretrained=False)
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
