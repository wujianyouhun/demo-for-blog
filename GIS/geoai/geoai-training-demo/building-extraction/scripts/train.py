"""
GeoAI 建筑提取训练完整示例
============================
任务：使用 U-Net + Attention Gate 从遥感影像中提取建筑物轮廓
数据集：WHU Building Dataset（武汉大学发布，8188 张航空影像）
模型：Attention U-Net（注意力门控 U-Net，聚焦建筑区域）
硬件：NVIDIA RTX 3060 (12GB VRAM)
作者：GeoAI Learning Lab
"""

import os
import sys
import time
import random
import warnings
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════
# 一、全局配置
# ══════════════════════════════════════════════════════════════════════

class Config:
    """集中管理所有超参数"""

    # --- 路径 ---
    project_dir = Path(__file__).parent.parent
    data_dir    = project_dir / "data"
    model_dir   = project_dir / "models"
    output_dir  = project_dir / "outputs"

    # --- 数据集 ---
    img_size    = 256          # 裁剪尺寸
    num_classes = 2            # 建筑 / 非建筑（二分类）
    in_channels = 3            # RGB

    # --- 训练 ---
    batch_size  = 8
    epochs      = 30
    lr          = 1e-4
    weight_decay = 1e-4
    patience    = 8

    # --- 数据增强 ---
    use_augmentation = True

    # --- 其他 ---
    num_workers = 4
    seed        = 42

    @classmethod
    def setup_dirs(cls):
        for d in [cls.data_dir, cls.model_dir, cls.output_dir]:
            d.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# 二、工具函数
# ══════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"[INFO] 使用 GPU: {gpu_name} ({vram:.1f} GB)")
    else:
        device = torch.device("cpu")
        print("[WARN] 未检测到 GPU，使用 CPU 训练")
    return device


# ══════════════════════════════════════════════════════════════════════
# 三、合成建筑数据生成器
# ══════════════════════════════════════════════════════════════════════

class SyntheticBuildingDataset:
    """
    生成模拟建筑提取的合成数据集。

    设计思路：
    - 影像：模拟城市航空影像，包含建筑（矩形/多边形）、植被、道路、水体
    - 标签：二值掩膜（1=建筑, 0=非建筑）
    - 建筑形状：矩形、L 型、U 型等，模拟真实建筑轮廓

    与语义分割的区别：
    - 这里只关注"建筑 vs 非建筑"的二分类问题
    - 建筑形状更加规则化（矩形、多边形）
    - 背景更加复杂（阴影、遮挡模拟）
    """

    def __init__(self, root_dir, num_train=300, num_val=60, img_size=256):
        self.root_dir = Path(root_dir)
        self.num_train = num_train
        self.num_val = num_val
        self.img_size = img_size
        self._generate()

    def _generate(self):
        for split, count in [("train", self.num_train), ("val", self.num_val)]:
            img_dir = self.root_dir / split / "images"
            lbl_dir = self.root_dir / split / "labels"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            existing = list(img_dir.glob("*.npy"))
            if len(existing) >= count:
                print(f"[INFO] {split} 数据已存在（{len(existing)} 张），跳过")
                continue

            print(f"[INFO] 生成 {count} 张合成建筑样本 ({split})...")
            for i in range(count):
                img, label = self._make_one_sample()
                np.save(img_dir / f"building_{i:04d}.npy", img)
                np.save(lbl_dir / f"building_{i:04d}.npy", label)

    def _make_one_sample(self):
        """
        生成一张包含多个建筑的合成航空影像。

        步骤：
        1. 生成背景（植被 + 道路 + 空地）
        2. 在背景上随机放置多个不同形状的建筑
        3. 添加建筑阴影效果
        4. 添加传感器噪声
        """
        h, w = self.img_size, self.img_size

        # 1. 生成背景
        background = self._make_background(h, w)

        # 2. 初始化标签（全 0 = 非建筑）
        label = np.zeros((h, w), dtype=np.int64)

        # 3. 生成建筑（3~12 个）
        image = background.copy()
        n_buildings = random.randint(3, 12)

        for _ in range(n_buildings):
            bldg_mask = self._make_building_mask(h, w)
            # 建筑颜色：灰白色调，带轻微变化
            base_color = np.array([
                random.uniform(0.55, 0.75),
                random.uniform(0.55, 0.70),
                random.uniform(0.50, 0.65),
            ])

            for c in range(3):
                noise = np.random.normal(0, 0.03, (h, w))
                channel = base_color[c] + noise
                image[:, :, c] = np.where(bldg_mask, channel, image[:, :, c])

            label[bldg_mask] = 1

            # 添加阴影（建筑右下方）
            shadow_mask = np.zeros((h, w), dtype=bool)
            shift_x = random.randint(3, 8)
            shift_y = random.randint(3, 8)
            if shift_y < h and shift_x < w:
                shifted = np.zeros_like(bldg_mask)
                shifted[shift_y:, shift_x:] = bldg_mask[:-shift_y, :-shift_x]
                shadow_mask = shifted & ~bldg_mask
                # 阴影变暗
                image[shadow_mask] *= random.uniform(0.6, 0.8)

        # 4. 传感器噪声
        noise = np.random.normal(0, 0.015, (h, w, 3))
        image = np.clip(image + noise, 0, 1).astype(np.float32)

        return image, label

    def _make_background(self, h, w):
        """生成包含植被、道路、空地的背景"""
        bg = np.full((h, w, 3), [0.35, 0.55, 0.25], dtype=np.float32)  # 基础植被色

        # 添加空地/裸土区域
        for _ in range(random.randint(2, 5)):
            cx, cy = random.randint(0, w - 30), random.randint(0, h - 30)
            rw, rh = random.randint(20, 80), random.randint(20, 80)
            earth_color = np.array([0.55, 0.50, 0.35])
            bg[cy:cy + rh, cx:cx + rw] = earth_color

        # 添加道路
        for _ in range(random.randint(1, 4)):
            road_color = np.array([0.45, 0.45, 0.42])
            if random.random() > 0.5:
                y = random.randint(0, h - 8)
                thickness = random.randint(5, 12)
                bg[y:y + thickness, :] = road_color
            else:
                x = random.randint(0, w - 8)
                thickness = random.randint(5, 12)
                bg[:, x:x + thickness] = road_color

        # 植被颜色变化
        veg_noise = np.random.normal(0, 0.06, (h, w, 3))
        bg = np.clip(bg + veg_noise, 0, 1)

        return bg

    def _make_building_mask(self, h, w):
        """生成一个随机建筑的二值掩膜"""
        mask = np.zeros((h, w), dtype=bool)
        cx = random.randint(10, w - 50)
        cy = random.randint(10, h - 50)

        shape_type = random.choice(["rect", "L", "cross"])

        if shape_type == "rect":
            bw = random.randint(15, 45)
            bh = random.randint(15, 45)
            mask[cy:cy + bh, cx:cx + bw] = True

        elif shape_type == "L":
            bw = random.randint(20, 40)
            bh = random.randint(20, 40)
            aw = random.randint(8, bw // 2)
            ah = random.randint(8, bh // 2)
            # L 型 = 两个矩形的并
            mask[cy:cy + bh, cx:cx + aw] = True
            mask[cy:cy + ah, cx:cx + bw] = True

        elif shape_type == "cross":
            bw = random.randint(25, 45)
            bh = random.randint(25, 45)
            aw = random.randint(8, bw // 2)
            ah = random.randint(8, bh // 2)
            mid_x = cx + bw // 2 - aw // 2
            mid_y = cy + bh // 2 - ah // 2
            mask[cy:cy + bh, mid_x:mid_x + aw] = True
            mask[mid_y:mid_y + ah, cx:cx + bw] = True

        return mask


# ══════════════════════════════════════════════════════════════════════
# 四、PyTorch 数据集类
# ══════════════════════════════════════════════════════════════════════

class BuildingDataset(Dataset):
    """
    建筑提取数据集。

    数据组织：
    data/
    ├── train/
    │   ├── images/    ← 航空影像（RGB）
    │   └── labels/    ← 二值标签（0=非建筑, 1=建筑）
    └── val/
        ├── images/
        └── labels/
    """

    def __init__(self, root_dir, split="train", img_size=256, augment=True):
        self.img_dir = Path(root_dir) / split / "images"
        self.lbl_dir = Path(root_dir) / split / "labels"
        self.img_size = img_size
        self.augment = augment and (split == "train")

        self.samples = []
        for ext in ["*.npy", "*.png", "*.tif", "*.jpg"]:
            for img_path in sorted(self.img_dir.glob(ext)):
                stem = img_path.stem
                for lbl_ext in [".npy", ".png", ".tif"]:
                    lbl_path = self.lbl_dir / f"{stem}{lbl_ext}"
                    if lbl_path.exists():
                        self.samples.append((img_path, lbl_path))
                        break

        if not self.samples:
            raise FileNotFoundError(
                f"在 {self.img_dir} 中未找到数据。\n"
                f"请先运行合成数据生成器，或下载 WHU Building Dataset。"
            )

        print(f"[INFO] {split} 集：{len(self.samples)} 个样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]

        if img_path.suffix == ".npy":
            image = np.load(img_path).astype(np.float32)
        else:
            image = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0

        if lbl_path.suffix == ".npy":
            label = np.load(lbl_path).astype(np.int64)
        else:
            label = np.array(Image.open(lbl_path)).astype(np.int64)
            if label.ndim == 3:
                label = label[:, :, 0]
            label = (label > 127).astype(np.int64)

        label = np.clip(label, 0, 1)

        if self.augment:
            image, label = self._augment(image, label)

        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        label = torch.from_numpy(label).long()

        return image, label

    def _augment(self, image, label):
        if random.random() > 0.5:
            image = np.flip(image, axis=1).copy()
            label = np.flip(label, axis=0).copy()
        if random.random() > 0.5:
            image = np.flip(image, axis=0).copy()
            label = np.flip(label, axis=1).copy()
        k = random.randint(0, 3)
        if k > 0:
            image = np.rot90(image, k, axes=(0, 1)).copy()
            label = np.rot90(label, k, axes=(0, 1)).copy()
        if random.random() > 0.5:
            brightness = random.uniform(0.85, 1.15)
            contrast = random.uniform(0.9, 1.1)
            image = np.clip((image - 0.5) * contrast + 0.5, 0, 1) * brightness
            image = np.clip(image, 0, 1)
        return image, label


# ══════════════════════════════════════════════════════════════════════
# 五、Attention U-Net 模型
# ══════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    """
    注意力门控机制（Attention Gate）

    这是 Attention U-Net 相比标准 U-Net 的关键创新：

    工作原理：
    1. 从解码器取上采样后的特征（引导信号 g）
    2. 从编码器取跳跃连接特征（输入特征 x）
    3. 将两者融合后通过 1×1 卷积 + Sigmoid 生成注意力权重
    4. 用注意力权重对编码器特征加权——只保留与目标相关的区域

    在建筑提取中，这让模型学会"忽略"植被、道路等非建筑区域，
    只关注建筑轮廓相关的特征，显著提升边界精度。
    """

    def __init__(self, g_channels, x_channels, inter_channels):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(g_channels, inter_channels, 1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_channels, inter_channels, 1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        g: 来自解码器的引导信号（上采样后的特征）
        x: 来自编码器的跳跃连接特征
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # 确保空间尺寸匹配
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=True)

        # 注意力权重计算
        attention = self.relu(g1 + x1)
        attention = self.psi(attention)  # (B, 1, H, W) 范围 [0, 1]

        # 用注意力权重加权编码器特征
        return x * attention


class AttentionUNet(nn.Module):
    """
    Attention U-Net 建筑提取网络

    与标准 U-Net 的区别：在每个跳跃连接处加入了 Attention Gate，
    让模型能够自动学习"关注哪里"。

    架构流程：

    输入 (3, 256, 256)
    │
    ├─ Enc1 (64) ──AG──┐
    │   ↓               │
    ├─ Enc2 (128)─AG──┐│
    │   ↓              ││
    ├─ Enc3 (256)─AG─┐││
    │   ↓             │││
    ├─ Enc4 (512)─AG┐│││
    │   ↓           ││││
    ├─ Bottleneck    ││││
    │   ↑            ││││
    ├─ Dec4 ← AG ────┘│││
    │   ↑             │││
    ├─ Dec3 ← AG ─────┘││
    │   ↑              ││
    ├─ Dec2 ← AG ──────┘│
    │   ↑               │
    ├─ Dec1 ← AG ───────┘
    │
    └─ 1×1 Conv → (2, 256, 256)
    """

    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()

        # 编码器
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)

        self.pool = nn.MaxPool2d(2)

        # 瓶颈层
        self.bottleneck = ConvBlock(512, 1024)

        # 上采样
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # 注意力门控
        self.ag4 = AttentionGate(g_channels=1024, x_channels=512, inter_channels=256)
        self.ag3 = AttentionGate(g_channels=512,  x_channels=256, inter_channels=128)
        self.ag2 = AttentionGate(g_channels=256,  x_channels=128, inter_channels=64)
        self.ag1 = AttentionGate(g_channels=128,  x_channels=64,  inter_channels=32)

        # 解码器
        self.dec4 = ConvBlock(1024 + 512, 512)
        self.dec3 = ConvBlock(512 + 256, 256)
        self.dec2 = ConvBlock(256 + 128, 128)
        self.dec1 = ConvBlock(128 + 64, 64)

        # 分类输出
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # 编码
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # 瓶颈
        b = self.bottleneck(self.pool(e4))

        # 解码 + 注意力门控
        d4 = self.up4(b)
        e4_attn = self.ag4(g=d4, x=e4)
        d4 = self.dec4(torch.cat([d4, e4_attn], dim=1))

        d3 = self.up3(d4)
        e3_attn = self.ag3(g=d3, x=e3)
        d3 = self.dec3(torch.cat([d3, e3_attn], dim=1))

        d2 = self.up2(d3)
        e2_attn = self.ag2(g=d2, x=e2)
        d2 = self.dec2(torch.cat([d2, e2_attn], dim=1))

        d1 = self.up1(d2)
        e1_attn = self.ag1(g=d1, x=e1)
        d1 = self.dec1(torch.cat([d1, e1_attn], dim=1))

        return self.final(d1)


# ══════════════════════════════════════════════════════════════════════
# 六、损失函数与评价指标
# ══════════════════════════════════════════════════════════════════════

class BuildingLoss(nn.Module):
    """
    建筑提取专用损失：Binary Cross Entropy + Dice Loss

    建筑提取的挑战：
    - 建筑通常只占图像面积的 10%~30%（类别不平衡）
    - 建筑边界需要高精度（边界像素损失更大）

    组合损失策略：
    - BCE：像素级的二分类损失，提供稳定的梯度
    - Dice Loss：区域级损失，对不平衡问题鲁棒
    - 可选：边界损失（Boundary Loss），加强边界精度
    """

    def __init__(self, pos_weight=3.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )

    def forward(self, pred, target):
        """
        pred:   (B, 2, H, W) logits
        target: (B, H, W)    0/1 标签
        """
        # BCE Loss（用建筑类别的 logit）
        pred_building = pred[:, 1:2, :, :]  # (B, 1, H, W)
        target_float = target.unsqueeze(1).float()  # (B, 1, H, W)
        bce_loss = self.bce(pred_building, target_float)

        # Dice Loss
        pred_prob = F.softmax(pred, dim=1)[:, 1]  # (B, H, W) 建筑概率
        target_bin = target.float()
        inter = (pred_prob * target_bin).sum()
        union = pred_prob.sum() + target_bin.sum()
        dice_loss = 1 - (2 * inter + 1) / (union + 1)

        return bce_loss + dice_loss


def compute_building_metrics(pred, target):
    """
    计算建筑提取评价指标。

    关键指标：
    - F1 Score：精确率与召回率的平衡
    - IoU：预测与真值的交集/并集
    - Precision：预测为建筑中真正的建筑比例
    - Recall：实际建筑中被正确检测的比例
    - Boundary F1：建筑边界的检测精度
    """
    pred_bin = pred.argmax(dim=1)

    tp = ((pred_bin == 1) & (target == 1)).sum().float()
    fp = ((pred_bin == 1) & (target == 0)).sum().float()
    fn = ((pred_bin == 0) & (target == 1)).sum().float()
    tn = ((pred_bin == 0) & (target == 0)).sum().float()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    oa = (tp + tn) / (tp + fp + fn + tn + 1e-8)

    return {
        "F1": f1.item(),
        "IoU": iou.item(),
        "OA": oa.item(),
        "Precision": precision.item(),
        "Recall": recall.item(),
    }


# ══════════════════════════════════════════════════════════════════════
# 七、训练引擎
# ══════════════════════════════════════════════════════════════════════

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config

        self.criterion = BuildingLoss(pos_weight=3.0).to(device)
        self.optimizer = optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs, eta_min=config.lr * 0.01
        )

        self.history = {"train_loss": [], "val_loss": [], "val_f1": []}
        self.best_f1 = 0
        self.patience_counter = 0

    def train_one_epoch(self):
        self.model.train()
        total_loss, n = 0, 0

        for idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            output = self.model(images)
            loss = self.criterion(output, labels)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n += 1

            if (idx + 1) % 5 == 0:
                print(f"  Batch {idx + 1}/{len(self.train_loader)} | Loss: {loss.item():.4f}")

        return total_loss / n

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss, all_metrics = 0, []

        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            output = self.model(images)
            loss = self.criterion(output, labels)
            total_loss += loss.item()

            metrics = compute_building_metrics(output, labels)
            all_metrics.append(metrics)

        avg_loss = total_loss / len(self.val_loader)
        avg_f1 = np.mean([m["F1"] for m in all_metrics])
        avg_iou = np.mean([m["IoU"] for m in all_metrics])
        avg_oa = np.mean([m["OA"] for m in all_metrics])
        avg_prec = np.mean([m["Precision"] for m in all_metrics])
        avg_rec = np.mean([m["Recall"] for m in all_metrics])

        return avg_loss, avg_f1, avg_iou, avg_oa, avg_prec, avg_rec

    def fit(self):
        print(f"\n{'=' * 60}")
        print(f"开始训练 | Epochs: {self.config.epochs} | "
              f"Batch Size: {self.config.batch_size}")
        print(f"{'=' * 60}\n")

        for epoch in range(self.config.epochs):
            t0 = time.time()

            train_loss = self.train_one_epoch()
            val_loss, val_f1, val_iou, val_oa, val_prec, val_rec = self.validate()
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_f1"].append(val_f1)

            print(f"Epoch {epoch + 1:02d}/{self.config.epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"F1: {val_f1:.4f} | IoU: {val_iou:.4f} | "
                  f"P: {val_prec:.4f} | R: {val_rec:.4f} | "
                  f"{time.time() - t0:.1f}s")

            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.patience_counter = 0
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "best_f1": val_f1,
                }, self.config.model_dir / "best_attention_unet.pth")
                print(f"  ✓ 保存最优模型 (F1={val_f1:.4f})")
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.config.patience:
                print(f"\n[EARLY STOP] 连续 {self.config.patience} 轮未提升")
                break

        print(f"\n训练完成！最佳 F1: {self.best_f1:.4f}")
        return self.history


# ══════════════════════════════════════════════════════════════════════
# 八、推理与可视化
# ══════════════════════════════════════════════════════════════════════

def visualize_building_results(dataset, model, device, output_dir, num_samples=4):
    """
    可视化建筑提取结果。

    四列展示：原始影像 | 真值建筑 | 预测建筑 | 轮廓叠加
    用红色轮廓高亮建筑边界，展示提取精度。
    """
    if not HAS_MPL:
        return

    model.eval()
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):
        image, label = dataset[idx]
        img_t = image.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_t)
        pred = output.argmax(dim=1).squeeze(0).cpu().numpy()

        img_np = image.permute(1, 2, 0).numpy()

        # 原始影像
        axes[row, 0].imshow(img_np)
        axes[row, 0].set_title("航空影像", fontsize=13)
        axes[row, 0].axis("off")

        # 真值建筑（蓝色半透明）
        gt_vis = img_np.copy()
        gt_vis[label == 1] = gt_vis[label == 1] * 0.5 + np.array([0.2, 0.4, 0.9]) * 0.5
        axes[row, 1].imshow(gt_vis)
        axes[row, 1].set_title("真值建筑（蓝色）", fontsize=13)
        axes[row, 1].axis("off")

        # 预测建筑（红色半透明）
        pred_vis = img_np.copy()
        pred_vis[pred == 1] = pred_vis[pred == 1] * 0.5 + np.array([0.9, 0.2, 0.1]) * 0.5
        axes[row, 2].imshow(pred_vis)
        axes[row, 2].set_title("预测建筑（红色）", fontsize=13)
        axes[row, 2].axis("off")

        # 轮廓叠加对比
        overlay = img_np.copy()
        # 真值边界（蓝色）
        from scipy import ndimage as ndi
        try:
            label_border = label.numpy() ^ ndi.binary_erosion(label.numpy())
            overlay[label_border == 1] = [0.2, 0.4, 0.9]
            pred_border = pred ^ ndi.binary_erosion(pred)
            overlay[pred_border == 1] = [0.9, 0.2, 0.1]
        except Exception:
            overlay[pred == 1] = overlay[pred == 1] * 0.5 + np.array([0.9, 0.2, 0.1]) * 0.5
        axes[row, 3].imshow(np.clip(overlay, 0, 1))
        axes[row, 3].set_title("轮廓对比", fontsize=13)
        axes[row, 3].axis("off")

    plt.tight_layout()
    save_path = output_dir / "building_extraction_results.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] 可视化结果已保存: {save_path}")


def plot_training_history(history, output_dir):
    if not HAS_MPL:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["train_loss"], label="训练损失", linewidth=2)
    ax1.plot(history["val_loss"], label="验证损失", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("损失曲线")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["val_f1"], label="验证 F1", linewidth=2, color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("F1 精度曲线")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


def export_building_vector(model, dataset, device, output_dir, num_samples=3):
    """
    将建筑提取结果导出为矢量数据（GeoJSON）。

    这是建筑提取的下游应用：
    - 栅格预测 → 多边形提取 → 简化 → 导出
    - 可用于城市规划、GIS 分析等场景
    """
    if not HAS_RASTERIO:
        print("[INFO] 安装 rasterio 后可导出矢量数据")
        return

    from rasterio import features
    from rasterio.transform import from_bounds
    import json

    model.eval()
    all_polygons = []

    for i in range(min(num_samples, len(dataset))):
        image, _ = dataset[i]
        img_t = image.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_t)
        pred = output.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # 模拟地理变换（实际使用时替换为真实地理坐标）
        transform = from_bounds(108.9, 34.2, 109.0, 34.3, pred.shape[1], pred.shape[0])

        # 提取多边形
        for geom, value in features.shapes(pred, transform=transform):
            if value == 1:  # 建筑
                all_polygons.append({
                    "type": "Feature",
                    "geometry": geom,
                    "properties": {"class": "building", "source": f"sample_{i}"}
                })

    if all_polygons:
        geojson = {
            "type": "FeatureCollection",
            "features": all_polygons
        }
        save_path = output_dir / "buildings.geojson"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 建筑矢量数据已导出: {save_path} ({len(all_polygons)} 个建筑)")


# ══════════════════════════════════════════════════════════════════════
# 九、WHU 数据集信息
# ══════════════════════════════════════════════════════════════════════

def print_dataset_info():
    print("""
╔══════════════════════════════════════════════════════════════╗
║  WHU Building Dataset 建筑提取数据集                         ║
║                                                              ║
║  来源：武汉大学测绘遥感信息工程国家重点实验室                    ║
║  规模：8,188 张 512×512 航空影像 + 二值标签                    ║
║  覆盖：新西兰 Christchurch 城区                               ║
║  分辨率：0.3m/像素                                            ║
║  特点：建筑形态多样，包含住宅、商业、工业建筑                    ║
║                                                              ║
║  下载：http://study.rsgis.whu.edu.cn/pages/download/          ║
║                                                              ║
║  当前使用合成数据进行演示。                                     ║
╚══════════════════════════════════════════════════════════════╝
    """)


# ══════════════════════════════════════════════════════════════════════
# 十、主函数
# ══════════════════════════════════════════════════════════════════════

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║          GeoAI 建筑提取训练完整示例                            ║
║          任务：遥感建筑物轮廓提取                               ║
║          模型：Attention U-Net                               ║
║          数据集：WHU Building / 合成数据                       ║
╚══════════════════════════════════════════════════════════════╝
    """)

    config = Config
    config.setup_dirs()
    set_seed(config.seed)
    device = get_device()

    # ── 数据准备 ──
    print("\n" + "=" * 60)
    print("步骤 1/5：数据准备")
    print("=" * 60)

    train_img_dir = config.data_dir / "train" / "images"
    if not train_img_dir.exists() or len(list(train_img_dir.glob("*"))) == 0:
        print("[INFO] 生成合成建筑数据集...")
        print_dataset_info()
        SyntheticBuildingDataset(
            config.data_dir,
            num_train=300, num_val=60,
            img_size=config.img_size
        )

    train_dataset = BuildingDataset(
        config.data_dir, "train", config.img_size, config.use_augmentation
    )
    val_dataset = BuildingDataset(
        config.data_dir, "val", config.img_size, augment=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True
    )

    # ── 模型构建 ──
    print("\n" + "=" * 60)
    print("步骤 2/5：构建 Attention U-Net 模型")
    print("=" * 60)

    model = AttentionUNet(in_channels=config.in_channels, num_classes=config.num_classes)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量：{total_params:,} ({total_params * 4 / 1024 / 1024:.1f} MB)")
    print(f"注意力门控数：4 层")

    # ── 训练 ──
    print("\n" + "=" * 60)
    print("步骤 3/5：模型训练")
    print("=" * 60)

    trainer = Trainer(model, train_loader, val_loader, device, config)
    history = trainer.fit()

    # ── 评估 ──
    print("\n" + "=" * 60)
    print("步骤 4/5：模型评估")
    print("=" * 60)

    best_path = config.model_dir / "best_attention_unet.pth"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"加载最优模型 (Epoch {ckpt['epoch']}, F1={ckpt['best_f1']:.4f})")

    val_loss, val_f1, val_iou, val_oa, val_prec, val_rec = trainer.validate()
    print(f"\n最终评估：")
    print(f"  F1:        {val_f1:.4f}")
    print(f"  IoU:       {val_iou:.4f}")
    print(f"  OA:        {val_oa:.4f}")
    print(f"  Precision: {val_prec:.4f}")
    print(f"  Recall:    {val_rec:.4f}")

    # ── 可视化 ──
    print("\n" + "=" * 60)
    print("步骤 5/5：推理与可视化")
    print("=" * 60)

    visualize_building_results(val_dataset, model, device, config.output_dir, num_samples=4)
    plot_training_history(history, config.output_dir)
    export_building_vector(model, val_dataset, device, config.output_dir, num_samples=3)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  训练完成！                                                   ║
║  • 最优模型：{config.model_dir}/best_attention_unet.pth                   ║
║  • 结果可视化：{config.output_dir}/building_extraction_results.png        ║
║  • 训练曲线：  {config.output_dir}/training_curves.png                    ║
║  • 建筑矢量：  {config.output_dir}/buildings.geojson (需安装rasterio)     ║
╚══════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
