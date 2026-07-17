from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _resize_like(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)


class ClassicUNet(nn.Module):
    """经典四级 U-Net，返回未经 softmax 的 logits。"""

    def __init__(self, in_channels: int = 3, num_classes: int = 6, base_channels: int = 32):
        super().__init__()
        widths = [base_channels * (2**index) for index in range(5)]
        self.encoder = nn.ModuleList([
            DoubleConv(in_channels, widths[0]),
            DoubleConv(widths[0], widths[1]),
            DoubleConv(widths[1], widths[2]),
            DoubleConv(widths[2], widths[3]),
        ])
        self.bottleneck = DoubleConv(widths[3], widths[4])
        self.pool = nn.MaxPool2d(2)
        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(widths[4], widths[3], 2, stride=2),
            nn.ConvTranspose2d(widths[3], widths[2], 2, stride=2),
            nn.ConvTranspose2d(widths[2], widths[1], 2, stride=2),
            nn.ConvTranspose2d(widths[1], widths[0], 2, stride=2),
        ])
        self.decoder = nn.ModuleList([
            DoubleConv(widths[3] * 2, widths[3]),
            DoubleConv(widths[2] * 2, widths[2]),
            DoubleConv(widths[1] * 2, widths[1]),
            DoubleConv(widths[0] * 2, widths[0]),
        ])
        self.head = nn.Conv2d(widths[0], num_classes, 1)

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        skips: list[torch.Tensor] = []
        features: dict[str, torch.Tensor] = {}
        for index, block in enumerate(self.encoder):
            x = block(x)
            skips.append(x)
            features[f"encoder_{index + 1}"] = x
            x = self.pool(x)
        x = self.bottleneck(x)
        features["bottleneck"] = x
        for index, (up, block, skip) in enumerate(zip(self.upconvs, self.decoder, reversed(skips))):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = _resize_like(x, skip)
            x = block(torch.cat([skip, x], dim=1))
            features[f"decoder_{index + 1}"] = x
        return self.head(x), features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)[0]


class NoSkipEncoderDecoder(nn.Module):
    """与经典 U-Net 深度和宽度一致，但完全移除跳跃连接。"""

    def __init__(self, in_channels: int = 3, num_classes: int = 6, base_channels: int = 32):
        super().__init__()
        widths = [base_channels * (2**index) for index in range(5)]
        self.encoder = nn.ModuleList([
            DoubleConv(in_channels, widths[0]),
            DoubleConv(widths[0], widths[1]),
            DoubleConv(widths[1], widths[2]),
            DoubleConv(widths[2], widths[3]),
        ])
        self.bottleneck = DoubleConv(widths[3], widths[4])
        self.pool = nn.MaxPool2d(2)
        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(widths[4], widths[3], 2, stride=2),
            nn.ConvTranspose2d(widths[3], widths[2], 2, stride=2),
            nn.ConvTranspose2d(widths[2], widths[1], 2, stride=2),
            nn.ConvTranspose2d(widths[1], widths[0], 2, stride=2),
        ])
        self.decoder = nn.ModuleList([
            DoubleConv(widths[3], widths[3]),
            DoubleConv(widths[2], widths[2]),
            DoubleConv(widths[1], widths[1]),
            DoubleConv(widths[0], widths[0]),
        ])
        self.head = nn.Conv2d(widths[0], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.encoder:
            x = self.pool(block(x))
        x = self.bottleneck(x)
        for up, block in zip(self.upconvs, self.decoder):
            x = block(up(x))
        return self.head(x)


class UNetPlusPlus(nn.Module):
    """四级密集跳连 U-Net++，与教学 U-Net 使用相同基础卷积块。"""

    def __init__(self, in_channels: int = 3, num_classes: int = 6, base_channels: int = 32):
        super().__init__()
        w = [base_channels * (2**index) for index in range(5)]
        self.pool = nn.MaxPool2d(2)
        self.x00 = DoubleConv(in_channels, w[0])
        self.x10 = DoubleConv(w[0], w[1])
        self.x20 = DoubleConv(w[1], w[2])
        self.x30 = DoubleConv(w[2], w[3])
        self.x40 = DoubleConv(w[3], w[4])
        self.x01 = DoubleConv(w[0] + w[1], w[0])
        self.x11 = DoubleConv(w[1] + w[2], w[1])
        self.x21 = DoubleConv(w[2] + w[3], w[2])
        self.x31 = DoubleConv(w[3] + w[4], w[3])
        self.x02 = DoubleConv(w[0] * 2 + w[1], w[0])
        self.x12 = DoubleConv(w[1] * 2 + w[2], w[1])
        self.x22 = DoubleConv(w[2] * 2 + w[3], w[2])
        self.x03 = DoubleConv(w[0] * 3 + w[1], w[0])
        self.x13 = DoubleConv(w[1] * 3 + w[2], w[1])
        self.x04 = DoubleConv(w[0] * 4 + w[1], w[0])
        self.head = nn.Conv2d(w[0], num_classes, 1)

    @staticmethod
    def up(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return _resize_like(x, reference)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x00 = self.x00(x)
        x10 = self.x10(self.pool(x00))
        x20 = self.x20(self.pool(x10))
        x30 = self.x30(self.pool(x20))
        x40 = self.x40(self.pool(x30))
        x01 = self.x01(torch.cat([x00, self.up(x10, x00)], 1))
        x11 = self.x11(torch.cat([x10, self.up(x20, x10)], 1))
        x21 = self.x21(torch.cat([x20, self.up(x30, x20)], 1))
        x31 = self.x31(torch.cat([x30, self.up(x40, x30)], 1))
        x02 = self.x02(torch.cat([x00, x01, self.up(x11, x00)], 1))
        x12 = self.x12(torch.cat([x10, x11, self.up(x21, x10)], 1))
        x22 = self.x22(torch.cat([x20, x21, self.up(x31, x20)], 1))
        x03 = self.x03(torch.cat([x00, x01, x02, self.up(x12, x00)], 1))
        x13 = self.x13(torch.cat([x10, x11, x12, self.up(x22, x10)], 1))
        x04 = self.x04(torch.cat([x00, x01, x02, x03, self.up(x13, x00)], 1))
        return self.head(x04)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        rates = (1, 3, 6, 9)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1 if rate == 1 else 3,
                          padding=0 if rate == 1 else rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            ) for rate in rates
        ])
        self.project = DoubleConv(out_channels * len(rates), out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class LiteDeepLabV3Plus(nn.Module):
    """不依赖下载权重的轻量 DeepLabV3+ 基线。"""

    def __init__(self, in_channels: int = 3, num_classes: int = 6, base_channels: int = 32):
        super().__init__()
        self.low = DoubleConv(in_channels, base_channels)
        self.encoder = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(base_channels, base_channels * 2),
            nn.MaxPool2d(2), DoubleConv(base_channels * 2, base_channels * 4),
            nn.MaxPool2d(2), DoubleConv(base_channels * 4, base_channels * 8),
        )
        self.aspp = ASPP(base_channels * 8, base_channels * 4)
        self.low_project = nn.Conv2d(base_channels, base_channels, 1)
        self.decoder = DoubleConv(base_channels * 5, base_channels * 2)
        self.head = nn.Conv2d(base_channels * 2, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        low = self.low(x)
        encoded = self.aspp(self.encoder(low))
        encoded = _resize_like(encoded, low)
        decoded = self.decoder(torch.cat([encoded, self.low_project(low)], dim=1))
        return F.interpolate(self.head(decoded), size=size, mode="bilinear", align_corners=False)


MODEL_NAMES = ("unet", "no_skip", "unetpp", "deeplabv3plus")


def build_model(name: str, in_channels: int = 3, num_classes: int = 6, base_channels: int = 32) -> nn.Module:
    factories = {
        "unet": ClassicUNet,
        "no_skip": NoSkipEncoderDecoder,
        "unetpp": UNetPlusPlus,
        "deeplabv3plus": LiteDeepLabV3Plus,
    }
    if name not in factories:
        raise ValueError(f"未知模型 {name}，可选: {', '.join(MODEL_NAMES)}")
    return factories[name](in_channels=in_channels, num_classes=num_classes, base_channels=base_channels)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


@dataclass(frozen=True)
class ArchitectureStage:
    name: str
    scale: str
    role: str
    skip: bool


ARCHITECTURE = [
    ArchitectureStage("Encoder 1", "1×", "保留边缘和纹理", True),
    ArchitectureStage("Encoder 2", "1/2", "组合局部结构", True),
    ArchitectureStage("Encoder 3", "1/4", "学习地物部件", True),
    ArchitectureStage("Encoder 4", "1/8", "形成高级语义", True),
    ArchitectureStage("Bottleneck", "1/16", "获得大感受野", False),
    ArchitectureStage("Decoder", "1×", "融合语义与跳连细节", True),
]
