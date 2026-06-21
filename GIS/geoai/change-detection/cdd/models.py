"""变化检测模型 - Siamese U-Net / BiT"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class _Encoder(nn.Module):
    def __init__(self, in_channels=3, encoder_name="resnet34"):
        super().__init__()
        import torchvision.models as models
        if encoder_name == "resnet34":
            bb = models.resnet34(weights="DEFAULT"); self.channels = [64, 64, 128, 256, 512]
        elif encoder_name == "resnet50":
            bb = models.resnet50(weights="DEFAULT"); self.channels = [64, 256, 512, 1024, 2048]
        elif encoder_name == "mobilenet_v2":
            mb = models.mobilenet_v2(weights="DEFAULT")
            self.channels = [16, 24, 32, 96, 320]
            if in_channels != 3:
                mb.features[0][0] = nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False)
            self.features = mb.features
            return
        else:
            raise ValueError(f"不支持: {encoder_name}")
        if in_channels != 3:
            bb.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.conv1 = nn.Sequential(bb.conv1, bb.bn1, bb.relu)
        self.maxpool = bb.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = bb.layer1, bb.layer2, bb.layer3, bb.layer4

    def forward(self, x):
        if hasattr(self, "features"):
            feats = []
            for i, block in enumerate(self.features):
                x = block(x)
                if i in [1, 3, 6, 13, 17]:
                    feats.append(x)
            return feats
        feats = [self.conv1(x)]
        x = self.maxpool(feats[-1])
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            x = layer(x)
            feats.append(x)
        return feats


class _Decoder(nn.Module):
    def __init__(self, enc_channels, dec_channels):
        super().__init__()
        enc_rev = list(reversed(enc_channels))
        self.up_convs = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()
        in_ch = enc_rev[0] * 2  # siamese: deepest features doubled
        for enc_ch, dec_ch in zip(enc_rev[1:], dec_channels):
            skip_ch = enc_ch * 2  # siamese: skip connections also doubled
            self.up_convs.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch, dec_ch, 2, stride=2),
                nn.BatchNorm2d(dec_ch), nn.ReLU(inplace=True),
            ))
            self.conv_blocks.append(nn.Sequential(
                nn.Conv2d(dec_ch + skip_ch, dec_ch, 3, padding=1),
                nn.BatchNorm2d(dec_ch), nn.ReLU(inplace=True),
            ))
            in_ch = dec_ch
        self.final_up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.final_conv = nn.Conv2d(in_ch, 2, 1)

    def forward(self, fa, fb):
        merged = [torch.cat([a, b], dim=1) for a, b in zip(fa, fb)]
        x = merged[-1]
        for i, (up, conv) in enumerate(zip(self.up_convs, self.conv_blocks)):
            x = up(x)
            skip = merged[-(i + 2)]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = conv(x)
        return self.final_conv(self.final_up(x))


class SiameseUNet(nn.Module):
    def __init__(self, in_channels=3, encoder_name="resnet34"):
        super().__init__()
        self.encoder = _Encoder(in_channels, encoder_name)
        self.decoder = _Decoder(self.encoder.channels, [256, 128, 64, 32])

    def forward(self, a, b):
        return {"out": self.decoder(self.encoder(a), self.encoder(b))}


class _TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        return x + self.mlp(self.norm2(x))


class BiT(nn.Module):
    def __init__(self, in_channels=3, encoder_name="resnet50", embed_dim=256, num_heads=8, depth=4):
        super().__init__()
        self.encoder = _Encoder(in_channels, encoder_name)
        self.proj = nn.Conv2d(self.encoder.channels[-1] * 2, embed_dim, 1)
        self.transformer = nn.Sequential(*[_TransformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 128, 4, stride=4), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=4), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=4), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 2, 1))

    def forward(self, a, b):
        fa, fb = self.encoder(a)[-1], self.encoder(b)[-1]
        x = self.proj(torch.cat([fa, fb], dim=1))
        B, C, H, W = x.shape
        x = self.transformer(x.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(B, C, H, W)
        return {"out": F.interpolate(self.decoder(x), size=a.shape[2:], mode="bilinear", align_corners=False)}


_REGISTRY = {"siamese_unet": SiameseUNet, "bit": BiT}


def build_model(model_name="siamese_unet", in_channels=3, **kw):
    if model_name not in _REGISTRY:
        raise ValueError(f"未知模型: {model_name}")
    Cls = _REGISTRY[model_name]
    if model_name == "siamese_unet":
        model = Cls(in_channels, kw.get("encoder", "resnet34"))
    elif model_name == "bit":
        model = Cls(in_channels, kw.get("encoder", "resnet50"), kw.get("embed_dim", 256), kw.get("num_heads", 8))
    else:
        model = Cls(in_channels)
    p = sum(x.numel() for x in model.parameters())
    logger.info(f"模型 {model_name}: {p:,} 参数")
    return model


def load_model(path, model_name="siamese_unet", in_channels=3, device="cpu", **kw):
    model = build_model(model_name, in_channels, **kw)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device).eval()
    return model
