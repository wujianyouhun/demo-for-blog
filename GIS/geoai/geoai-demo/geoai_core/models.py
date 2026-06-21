"""GeoAI Core - 语义分割模型构建与加载"""
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# 模型注册表
# --------------------------------------------------------------------------
MODEL_REGISTRY = {
    "deeplabv3p_resnet50": {
        "arch": "DeepLabV3Plus",
        "encoder": "resnet50",
    },
    "deeplabv3p_resnet101": {
        "arch": "DeepLabV3Plus",
        "encoder": "resnet101",
    },
    "deeplabv3p_mobilenet": {
        "arch": "DeepLabV3Plus",
        "encoder": "mobilenet_v2",
    },
}


def build_model(
    model_name: str,
    num_classes: int = 6,
    in_channels: int = 3,
    pretrained: bool = True,
) -> nn.Module:
    """
    根据名称构建语义分割模型。

    Args:
        model_name: 模型名称, 如 "deeplabv3p_resnet50"
        num_classes: 分类类别数 (默认 6)
        in_channels: 输入通道数 (默认 3, RGB)
        pretrained: 是否使用 ImageNet 预训练编码器

    Returns:
        构建好的 PyTorch 模型
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise ImportError(
            "需要安装 segmentation-models-pytorch: "
            "pip install segmentation-models-pytorch"
        )

    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"未知模型: {model_name}. 可选: {available}"
        )

    cfg = MODEL_REGISTRY[model_name]
    encoder_name = cfg["encoder"]
    encoder_weights = "imagenet" if pretrained else None

    logger.info(
        "构建模型: %s (encoder=%s, classes=%d, in_channels=%d, pretrained=%s)",
        model_name, encoder_name, num_classes, in_channels, pretrained,
    )

    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,
    )

    return model


def load_model(
    path: str | Path,
    model_name: str,
    num_classes: int = 6,
    device: str | torch.device = "cpu",
    in_channels: int = 3,
) -> nn.Module:
    """
    从检查点加载模型权重。

    Args:
        path: 检查点文件路径 (.pth)
        model_name: 模型名称
        num_classes: 分类类别数
        device: 目标设备
        in_channels: 输入通道数

    Returns:
        加载权重后的模型
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"模型文件不存在: {path}")

    device = torch.device(device)

    # 先构建模型结构
    model = build_model(
        model_name, num_classes=num_classes,
        in_channels=in_channels, pretrained=False,
    )

    # 加载权重
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

    logger.info("模型已加载: %s <- %s (device=%s)", model_name, path, device)
    return model
