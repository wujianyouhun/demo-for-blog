import torch
import torch.nn as nn
import torchvision.models as models

class ASPP(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, 256, 3, padding=6, dilation=6, bias=False)
        self.conv3 = nn.Conv2d(in_channels, 256, 3, padding=12, dilation=12, bias=False)
        self.conv4 = nn.Conv2d(in_channels, 256, 3, padding=18, dilation=18, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Conv2d(256*5, 256, 1, bias=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.pool(x)
        x5 = self.conv1(x5)
        x5 = F.interpolate(x5, size=x.shape[2:], mode='bilinear', align_corners=False)
        return self.project(torch.cat([x1,x2,x3,x4,x5], dim=1))

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.aspp = ASPP(1024)
        self.final = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        feat = self.backbone(x)
        aspp = self.aspp(feat)
        out = F.interpolate(aspp, size=x.shape[2:], mode='bilinear', align_corners=False)
        return torch.sigmoid(self.final(out))