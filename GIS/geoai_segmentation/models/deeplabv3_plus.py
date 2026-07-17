import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1 if rate == 1 else 3,
                                    padding=0 if rate == 1 else rate, dilation=rate, bias=False),
                          nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
            for rate in (1, 3, 6, 9)
        ])
        self.project = ConvBlock(out_channels * 4, out_channels)

    def forward(self, x):
        return self.project(torch.cat([branch(x) for branch in self.branches], 1))


class DeepLabV3Plus(nn.Module):
    """无需下载预训练权重的轻量 DeepLabV3+。"""

    def __init__(self, in_ch=3, out_ch=2, base_channels=32):
        super().__init__()
        self.low = ConvBlock(in_ch, base_channels)
        self.encoder = nn.Sequential(
            ConvBlock(base_channels, base_channels * 2, stride=2),
            ConvBlock(base_channels * 2, base_channels * 4, stride=2),
            ConvBlock(base_channels * 4, base_channels * 8, stride=2),
        )
        self.aspp = ASPP(base_channels * 8, base_channels * 4)
        self.low_project = nn.Conv2d(base_channels, base_channels, 1)
        self.decoder = ConvBlock(base_channels * 5, base_channels * 2)
        self.head = nn.Conv2d(base_channels * 2, out_ch, 1)

    def forward(self, x):
        size = x.shape[-2:]
        low = self.low(x)
        high = self.aspp(self.encoder(low))
        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        out = self.head(self.decoder(torch.cat([high, self.low_project(low)], 1)))
        return F.interpolate(out, size=size, mode="bilinear", align_corners=False)
