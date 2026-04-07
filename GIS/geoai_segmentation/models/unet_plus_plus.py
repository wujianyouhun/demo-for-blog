import torch
import torch.nn as nn
import torch.nn.functional as F

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class UNetPlusPlus(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        n1 = 64
        filters = [n1, n1*2, n1*4, n1*8, n1*16]

        self.conv0_0 = VGGBlock(in_ch, filters[0])
        self.conv1_0 = VGGBlock(filters[0], filters[1])
        self.conv2_0 = VGGBlock(filters[1], filters[2])
        self.conv3_0 = VGGBlock(filters[2], filters[3])
        self.conv4_0 = VGGBlock(filters[3], filters[4])

        self.conv0_1 = VGGBlock(filters[0]+filters[1], filters[0])
        self.conv1_1 = VGGBlock(filters[1]+filters[2], filters[1])
        self.conv2_1 = VGGBlock(filters[2]+filters[3], filters[2])
        self.conv3_1 = VGGBlock(filters[3]+filters[4], filters[3])

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final = nn.Conv2d(filters[0], out_ch, 1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        out = self.final(x0_1)
        return torch.sigmoid(out)