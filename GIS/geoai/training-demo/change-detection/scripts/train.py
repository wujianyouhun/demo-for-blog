"""
GeoAI 变化检测训练完整示例
============================
任务：使用孪生 U-Net (Siam-UNet) 检测两时相遥感影像之间的变化区域
数据集：LEVIR-CD（双时相建筑变化检测基准，637 对 1024×1024 影像）
模型：孪生 U-Net（共享编码器 + 差异融合 + 解码器）
硬件：NVIDIA RTX 3050 (6GB VRAM)
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
    num_classes = 2            # 变化/未变化（二分类）
    in_channels = 3            # RGB

    # --- 训练 ---
    batch_size  = 3            # 孪生网络双倍输入，6GB 显存适配
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
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[INFO] 使用 GPU: {gpu_name} ({vram:.1f} GB)")
    else:
        device = torch.device("cpu")
        print("[WARN] 未检测到 GPU，将使用 CPU 训练")
    return device


# ══════════════════════════════════════════════════════════════════════
# 三、合成变化检测数据生成器
# ══════════════════════════════════════════════════════════════════════

class SyntheticChangeDataset:
    """
    生成模拟双时相变化检测的合成数据集。

    设计思路：
    - 时相 A：随机生成带有建筑、道路、植被的城市场景
    - 时相 B：在时相 A 基础上，随机添加/移除一些建筑，模拟城市扩张
    - 标签：标记发生变化的像素区域（1=变化，0=未变化）

    变化类型模拟：
    1. 新增建筑（植被/空地 → 建筑）
    2. 建筑拆除（建筑 → 空地）
    3. 道路扩建
    """

    COLORS = {
        "background": np.array([180, 180, 160]) / 255.0,
        "building":   np.array([190, 90,  60])  / 255.0,
        "vegetation": np.array([50,  160, 50])  / 255.0,
        "road":       np.array([130, 130, 130]) / 255.0,
        "water":      np.array([40,  100, 190]) / 255.0,
    }

    def __init__(self, root_dir, num_train=200, num_val=50, img_size=256):
        self.root_dir = Path(root_dir)
        self.num_train = num_train
        self.num_val = num_val
        self.img_size = img_size
        self._generate()

    def _generate(self):
        for split, count in [("train", self.num_train), ("val", self.num_val)]:
            a_dir = self.root_dir / split / "A"
            b_dir = self.root_dir / split / "B"
            lbl_dir = self.root_dir / split / "label"
            for d in [a_dir, b_dir, lbl_dir]:
                d.mkdir(parents=True, exist_ok=True)

            existing = list(a_dir.glob("*.npy"))
            if len(existing) >= count:
                print(f"[INFO] {split} 数据已存在（{len(existing)} 对），跳过")
                continue

            print(f"[INFO] 生成 {count} 对合成变化检测样本 ({split})...")
            for i in range(count):
                img_a, img_b, label = self._make_one_pair()
                np.save(a_dir / f"pair_{i:04d}.npy", img_a)
                np.save(b_dir / f"pair_{i:04d}.npy", img_b)
                np.save(lbl_dir / f"pair_{i:04d}.npy", label)

    def _make_one_pair(self):
        """
        生成一对双时相图像和变化标签。

        步骤：
        1. 生成基础场景（时相 A）
        2. 复制并在若干区域做变化（时相 B）
        3. 记录变化区域为二值标签
        """
        h, w = self.img_size, self.img_size

        # 生成时相 A 的基础场景
        img_a = self._make_scene(h, w)
        # 复制为时相 B 的基础
        img_b = img_a.copy()
        # 变化标签（全 0 = 未变化）
        label = np.zeros((h, w), dtype=np.int64)

        # 随机选择 1~4 个区域做变化
        n_changes = random.randint(1, 4)
        for _ in range(n_changes):
            # 随机矩形区域
            cx = random.randint(20, w - 60)
            cy = random.randint(20, h - 60)
            rw = random.randint(20, 60)
            rh = random.randint(20, 60)

            # 变化类型
            change_type = random.choice(["build", "demolish"])

            if change_type == "build":
                # 新建建筑：在 B 中将该区域变为建筑色
                img_b[cy:cy + rh, cx:cx + rw] = (
                    self.COLORS["building"] + np.random.normal(0, 0.03, (rh, rw, 3))
                ).clip(0, 1)
            else:
                # 拆除建筑：在 B 中将该区域变为空地
                img_b[cy:cy + rh, cx:cx + rw] = (
                    self.COLORS["background"] + np.random.normal(0, 0.03, (rh, rw, 3))
                ).clip(0, 1)

            # 标记变化区域
            label[cy:cy + rh, cx:cx + rw] = 1

        return img_a.astype(np.float32), img_b.astype(np.float32), label

    def _make_scene(self, h, w):
        """生成一个随机的城市场景"""
        scene = np.full((h, w, 3), self.COLORS["background"], dtype=np.float32)

        # 随机放置植被区域
        for _ in range(random.randint(3, 8)):
            cx, cy = random.randint(0, w - 40), random.randint(0, h - 40)
            rw, rh = random.randint(20, 80), random.randint(20, 80)
            color = self.COLORS["vegetation"]
            scene[cy:cy + rh, cx:cx + rw] = (
                color + np.random.normal(0, 0.04, (rh, rw, 3))
            ).clip(0, 1)

        # 随机放置建筑
        for _ in range(random.randint(5, 15)):
            cx, cy = random.randint(0, w - 30), random.randint(0, h - 30)
            rw, rh = random.randint(15, 40), random.randint(15, 40)
            color = self.COLORS["building"]
            scene[cy:cy + rh, cx:cx + rw] = (
                color + np.random.normal(0, 0.03, (rh, rw, 3))
            ).clip(0, 1)

        # 随机放置道路（水平或垂直长条）
        for _ in range(random.randint(1, 3)):
            if random.random() > 0.5:
                y = random.randint(0, h - 10)
                scene[y:y + 8, :] = self.COLORS["road"]
            else:
                x = random.randint(0, w - 10)
                scene[:, x:x + 8] = self.COLORS["road"]

        # 添加高斯噪声模拟传感器噪声
        noise = np.random.normal(0, 0.02, (h, w, 3))
        scene = np.clip(scene + noise, 0, 1)

        return scene.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# 四、PyTorch 数据集类
# ══════════════════════════════════════════════════════════════════════

class ChangeDetectionDataset(Dataset):
    """
    双时相变化检测数据集。

    数据组织方式：
    data/
    ├── train/
    │   ├── A/        ← 时相 A 图像
    │   ├── B/        ← 时相 B 图像
    │   └── label/    ← 变化标签（0=未变, 1=变化）
    └── val/
        ├── A/
        ├── B/
        └── label/

    支持 .npy（合成数据）和 .png/.tif（真实数据）两种格式。
    """

    def __init__(self, root_dir, split="train", img_size=256, augment=True):
        self.a_dir   = Path(root_dir) / split / "A"
        self.b_dir   = Path(root_dir) / split / "B"
        self.lbl_dir = Path(root_dir) / split / "label"
        self.img_size = img_size
        self.augment = augment and (split == "train")

        # 收集配对样本
        self.pairs = []
        for ext in ["*.npy", "*.png", "*.tif"]:
            for a_path in sorted(self.a_dir.glob(ext)):
                stem = a_path.stem
                b_path = self.b_dir / a_path.name
                # 尝试不同的标签扩展名
                lbl_path = None
                for lbl_ext in [".npy", ".png", ".tif"]:
                    candidate = self.lbl_dir / f"{stem}{lbl_ext}"
                    if candidate.exists():
                        lbl_path = candidate
                        break
                if b_path.exists() and lbl_path:
                    self.pairs.append((a_path, b_path, lbl_path))
                    break  # 找到一组就跳过其他扩展名

        if not self.pairs:
            raise FileNotFoundError(
                f"在 {self.a_dir} 中未找到配对的 A/B/label 数据。\n"
                f"请先运行合成数据生成器，或下载 LEVIR-CD 数据集。"
            )

        print(f"[INFO] {split} 集：加载 {len(self.pairs)} 对样本")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a_path, b_path, lbl_path = self.pairs[idx]

        # 加载图像 A
        if a_path.suffix == ".npy":
            img_a = np.load(a_path).astype(np.float32)
        else:
            img_a = np.array(Image.open(a_path).convert("RGB")).astype(np.float32) / 255.0

        # 加载图像 B
        if b_path.suffix == ".npy":
            img_b = np.load(b_path).astype(np.float32)
        else:
            img_b = np.array(Image.open(b_path).convert("RGB")).astype(np.float32) / 255.0

        # 加载标签
        if lbl_path.suffix == ".npy":
            label = np.load(lbl_path).astype(np.int64)
        else:
            label = np.array(Image.open(lbl_path)).astype(np.int64)
            if label.ndim == 3:
                label = label[:, :, 0]
            # 二值化（有些标签图用 255 表示变化）
            label = (label > 127).astype(np.int64)

        # 同步数据增强
        if self.augment:
            img_a, img_b, label = self._augment(img_a, img_b, label)

        # HWC → CHW
        img_a = torch.from_numpy(img_a.transpose(2, 0, 1)).float()
        img_b = torch.from_numpy(img_b.transpose(2, 0, 1)).float()
        label = torch.from_numpy(label).long()

        return img_a, img_b, label

    def _augment(self, img_a, img_b, label):
        """同步增强：对 A、B、标签做相同的几何变换"""
        if random.random() > 0.5:
            img_a = np.flip(img_a, axis=1).copy()
            img_b = np.flip(img_b, axis=1).copy()
            label = np.flip(label, axis=0).copy()

        if random.random() > 0.5:
            img_a = np.flip(img_a, axis=0).copy()
            img_b = np.flip(img_b, axis=0).copy()
            label = np.flip(label, axis=1).copy()

        k = random.randint(0, 3)
        if k > 0:
            img_a = np.rot90(img_a, k, axes=(0, 1)).copy()
            img_b = np.rot90(img_b, k, axes=(0, 1)).copy()
            label = np.rot90(label, k, axes=(0, 1)).copy()

        # 独立颜色抖动（A、B 各自独立，模拟不同拍摄条件）
        for img in [img_a, img_b]:
            if random.random() > 0.5:
                brightness = random.uniform(0.9, 1.1)
                img[:] = np.clip(img * brightness, 0, 1)

        return img_a, img_b, label


# ══════════════════════════════════════════════════════════════════════
# 五、孪生 U-Net 模型（Siam-UNet with Difference Fusion）
# ══════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """双卷积块"""
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


class Encoder(nn.Module):
    """
    共享编码器（孪生结构的核心）

    两个时相的图像通过同一个编码器提取特征。
    共享权重意味着模型对"相同内容在不同时间拍摄"有一致的理解。
    """

    def __init__(self, in_channels=3):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)

    def forward(self, x):
        """返回各层特征用于跳跃连接"""
        e1 = self.enc1(x)          # (B, 64, H, W)
        e2 = self.enc2(self.pool1(e1))  # (B, 128, H/2, W/2)
        e3 = self.enc3(self.pool2(e2))  # (B, 256, H/4, W/4)
        e4 = self.enc4(self.pool3(e3))  # (B, 512, H/8, W/8)
        b = self.pool4(e4)              # (B, 512, H/16, W/16)
        return e1, e2, e3, e4, b


class ChangeDecoder(nn.Module):
    """
    变化检测解码器

    融合策略：对两个时相的编码器特征做差值 (B - A)，
    然后拼接原始特征，让解码器同时看到"差异"和"原始内容"。

    这种差异融合的思想是变化检测网络的核心：
    - 差值特征：直接反映变化信息
    - 原始特征：提供上下文，帮助区分真实变化和噪声
    """

    def __init__(self, num_classes=2):
        super().__init__()
        # 上采样层
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # 解码卷积块（输入通道 = 上一层输出 + 差异特征跳跃连接）
        # dec4: diff_b(512) + b_b(512) = 1024 → 256
        self.dec4 = ConvBlock(512 + 512, 256)
        # dec3: dec4_out(256) + diff_e4(512) = 768 → 128
        self.dec3 = ConvBlock(256 + 512, 128)
        # dec2: dec3_out(128) + diff_e3(256) = 384 → 64
        self.dec2 = ConvBlock(128 + 256, 64)
        # dec1: dec2_out(64)  + diff_e2(128) = 192 → 32
        self.dec1 = ConvBlock(64 + 128, 32)

        # 最终分类
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, feats_a, feats_b):
        """
        feats_a: 时相 A 的编码器特征元组 (e1, e2, e3, e4, bottleneck)
        feats_b: 时相 B 的编码器特征元组
        """
        e1_a, e2_a, e3_a, e4_a, b_a = feats_a
        e1_b, e2_b, e3_b, e4_b, b_b = feats_b

        # 差异融合：|B - A| + 拼接
        # 使用绝对差值，因为变化是无方向的
        diff_b = torch.abs(b_b - b_a)
        x = torch.cat([diff_b, b_b], dim=1)  # (B, 1024, H/16, W/16)
        x = self.dec4(x)                      # (B, 256, H/16, W/16)

        # 逐层上采样 + 跳跃连接
        x = self.up4(x)
        diff_4 = torch.abs(e4_b - e4_a)
        x = torch.cat([x, diff_4], dim=1)
        x = self.dec3(x)

        x = self.up3(x)
        diff_3 = torch.abs(e3_b - e3_a)
        x = torch.cat([x, diff_3], dim=1)
        x = self.dec2(x)

        x = self.up2(x)
        diff_2 = torch.abs(e2_b - e2_a)
        x = torch.cat([x, diff_2], dim=1)
        x = self.dec1(x)

        x = self.up1(x)

        return self.final(x)


class SiamUNet(nn.Module):
    """
    孪生 U-Net 变化检测网络

    完整架构：

    时相 A ──┐                     ┌── 差异特征
             ├── 共享编码器 ──→ │   │      │
    时相 B ──┘                 └── 原始特征 ──→ 解码器 ──→ 变化图

    关键设计：
    1. 共享编码器：两时相使用相同权重提取特征
    2. 差异融合：在每一层计算 |B - A| 的绝对差
    3. 跳跃连接：传递高分辨率的差异细节
    """

    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        self.encoder = Encoder(in_channels)
        self.decoder = ChangeDecoder(num_classes)

    def forward(self, img_a, img_b):
        feats_a = self.encoder(img_a)
        feats_b = self.encoder(img_b)
        output = self.decoder(feats_a, feats_b)
        return output


# ══════════════════════════════════════════════════════════════════════
# 六、损失函数与评价指标
# ══════════════════════════════════════════════════════════════════════

class ChangeLoss(nn.Module):
    """
    变化检测专用损失函数：加权交叉熵 + Focal Loss

    变化检测的核心挑战是类别极度不平衡——
    通常只有 5%~20% 的像素是"变化"的。

    解决方案：
    - 加权交叉熵：给变化类别更高的权重
    - Focal Loss：让模型专注于难以分类的样本
    """

    def __init__(self, pos_weight=5.0):
        super().__init__()
        # 变化类别权重更高
        self.ce = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, pos_weight])
        )
        self.gamma = 2.0  # Focal Loss 的聚焦参数

    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)

        # Focal Loss
        pred_prob = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, 2).permute(0, 3, 1, 2).float()
        focal_weight = (1 - pred_prob) ** self.gamma
        focal_loss = -(focal_weight * target_onehot * torch.log(pred_prob + 1e-8)).mean()

        return ce_loss + focal_loss * 0.5


def compute_change_metrics(pred, target, num_classes=2):
    """
    计算变化检测评价指标。

    关键指标：
    - F1 Score：精确率和召回率的调和平均
    - IoU：交并比
    - OA：总体精度
    - Kappa：考虑随机一致性的修正精度
    """
    pred_binary = pred.argmax(dim=1)

    tp = ((pred_binary == 1) & (target == 1)).sum().float()
    fp = ((pred_binary == 1) & (target == 0)).sum().float()
    fn = ((pred_binary == 0) & (target == 1)).sum().float()
    tn = ((pred_binary == 0) & (target == 0)).sum().float()

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
    """变化检测训练管理器"""

    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config

        self.criterion = ChangeLoss(pos_weight=5.0).to(device)
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

        for idx, (img_a, img_b, label) in enumerate(self.train_loader):
            img_a = img_a.to(self.device)
            img_b = img_b.to(self.device)
            label = label.to(self.device)

            output = self.model(img_a, img_b)
            loss = self.criterion(output, label)

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

        for img_a, img_b, label in self.val_loader:
            img_a = img_a.to(self.device)
            img_b = img_b.to(self.device)
            label = label.to(self.device)

            output = self.model(img_a, img_b)
            loss = self.criterion(output, label)
            total_loss += loss.item()

            metrics = compute_change_metrics(output, label)
            all_metrics.append(metrics)

        avg_loss = total_loss / len(self.val_loader)
        avg_f1 = np.mean([m["F1"] for m in all_metrics])
        avg_iou = np.mean([m["IoU"] for m in all_metrics])
        avg_oa = np.mean([m["OA"] for m in all_metrics])

        return avg_loss, avg_f1, avg_iou, avg_oa

    def fit(self):
        print(f"\n{'=' * 60}")
        print(f"开始训练 | Epochs: {self.config.epochs} | "
              f"Batch Size: {self.config.batch_size}")
        print(f"{'=' * 60}\n")

        for epoch in range(self.config.epochs):
            t0 = time.time()

            train_loss = self.train_one_epoch()
            val_loss, val_f1, val_iou, val_oa = self.validate()
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_f1"].append(val_f1)

            print(f"Epoch {epoch + 1:02d}/{self.config.epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"F1: {val_f1:.4f} | IoU: {val_iou:.4f} | "
                  f"OA: {val_oa:.4f} | LR: {lr:.6f} | "
                  f"{time.time() - t0:.1f}s")

            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.patience_counter = 0
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "best_f1": val_f1,
                }, self.config.model_dir / "best_siamunet.pth")
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

def visualize_change_results(dataset, model, device, output_dir, num_samples=4):
    """
    可视化变化检测结果：时相A | 时相B | 真值 | 预测 | 叠加

    用红色高亮标记检测到的变化区域，方便直观评估效果。
    """
    if not HAS_MPL:
        return

    model.eval()
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    fig, axes = plt.subplots(num_samples, 5, figsize=(25, 5 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):
        img_a, img_b, label = dataset[idx]
        img_a_t = img_a.unsqueeze(0).to(device)
        img_b_t = img_b.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_a_t, img_b_t)
        pred = output.argmax(dim=1).squeeze(0).cpu().numpy()

        img_a_np = img_a.permute(1, 2, 0).numpy()
        img_b_np = img_b.permute(1, 2, 0).numpy()

        # 时相 A
        axes[row, 0].imshow(img_a_np)
        axes[row, 0].set_title("时相 A", fontsize=13)
        axes[row, 0].axis("off")

        # 时相 B
        axes[row, 1].imshow(img_b_np)
        axes[row, 1].set_title("时相 B", fontsize=13)
        axes[row, 1].axis("off")

        # 真值变化图（红=变化，绿=未变化）
        gt_vis = np.zeros((*label.shape, 3))
        gt_vis[label == 0] = [0.2, 0.6, 0.2]   # 未变化-绿
        gt_vis[label == 1] = [0.9, 0.2, 0.1]   # 变化-红
        axes[row, 2].imshow(gt_vis)
        axes[row, 2].set_title("真值变化", fontsize=13)
        axes[row, 2].axis("off")

        # 预测变化图
        pred_vis = np.zeros((*pred.shape, 3))
        pred_vis[pred == 0] = [0.2, 0.6, 0.2]
        pred_vis[pred == 1] = [0.9, 0.2, 0.1]
        axes[row, 3].imshow(pred_vis)
        axes[row, 3].set_title("预测变化", fontsize=13)
        axes[row, 3].axis("off")

        # 叠加对比（预测变化叠加到时相 B 上）
        overlay = img_b_np.copy()
        change_mask = pred == 1
        overlay[change_mask] = overlay[change_mask] * 0.4 + np.array([0.9, 0.2, 0.1]) * 0.6
        axes[row, 4].imshow(np.clip(overlay, 0, 1))
        axes[row, 4].set_title("变化叠加", fontsize=13)
        axes[row, 4].axis("off")

    plt.tight_layout()
    save_path = output_dir / "change_detection_results.png"
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


# ══════════════════════════════════════════════════════════════════════
# 九、LEVIR-CD 数据集信息
# ══════════════════════════════════════════════════════════════════════

def print_dataset_info():
    print("""
╔══════════════════════════════════════════════════════════════╗
║  LEVIR-CD 建筑变化检测数据集                                  ║
║                                                              ║
║  数据规模：637 对高分辨率双时相影像（1024×1024）               ║
║  覆盖范围：美国德克萨斯州多个城市区域                          ║
║  时间跨度：2002-2018 年                                       ║
║  分辨率：0.5m/像素                                            ║
║  标注类型：二值变化标签（建筑增减变化）                         ║
║                                                              ║
║  下载地址：https://github.com/chenhaozipeng/LEVIR-CD-Dataset  ║
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
║          GeoAI 变化检测训练完整示例                            ║
║          任务：双时相建筑变化检测                               ║
║          模型：孪生 U-Net (Siam-UNet)                         ║
║          数据集：LEVIR-CD / 合成数据                           ║
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

    train_a_dir = config.data_dir / "train" / "A"
    if not train_a_dir.exists() or len(list(train_a_dir.glob("*"))) == 0:
        print("[INFO] 生成合成变化检测数据集...")
        print_dataset_info()
        SyntheticChangeDataset(
            config.data_dir,
            num_train=200, num_val=50,
            img_size=config.img_size
        )

    train_dataset = ChangeDetectionDataset(
        config.data_dir, "train", config.img_size, config.use_augmentation
    )
    val_dataset = ChangeDetectionDataset(
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
    print("步骤 2/5：构建孪生 U-Net 模型")
    print("=" * 60)

    model = SiamUNet(in_channels=config.in_channels, num_classes=config.num_classes)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量：{total_params:,} ({total_params * 4 / 1024 / 1024:.1f} MB)")
    print(f"编码器参数（共享）：{sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"解码器参数：{sum(p.numel() for p in model.decoder.parameters()):,}")

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

    best_path = config.model_dir / "best_siamunet.pth"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"加载最优模型 (Epoch {ckpt['epoch']}, F1={ckpt['best_f1']:.4f})")

    val_loss, val_f1, val_iou, val_oa = trainer.validate()
    print(f"\n最终评估：F1={val_f1:.4f} | IoU={val_iou:.4f} | OA={val_oa:.4f}")

    # ── 可视化 ──
    print("\n" + "=" * 60)
    print("步骤 5/5：推理可视化")
    print("=" * 60)

    visualize_change_results(val_dataset, model, device, config.output_dir, num_samples=4)
    plot_training_history(history, config.output_dir)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  训练完成！                                                   ║
║  • 最优模型：{config.model_dir}/best_siamunet.pth                        ║
║  • 结果可视化：{config.output_dir}/change_detection_results.png           ║
║  • 训练曲线：  {config.output_dir}/training_curves.png                    ║
╚══════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
