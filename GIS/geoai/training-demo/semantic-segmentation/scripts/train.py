"""
GeoAI 语义分割训练完整示例
============================
任务：使用 U-Net 对高分辨率遥感影像进行像素级土地覆盖分类
数据集：LoveDA（武汉大学发布，涵盖城市与农村场景，7 类土地覆盖）
模型：U-Net + ResNet34 编码器（ImageNet 预训练权重自动下载）
硬件：NVIDIA RTX 3050 (6GB VRAM)
作者：GeoAI Learning Lab
"""

import os
import sys
import time
import shutil
import random
import zipfile
import warnings
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ─── 第三方库（可选，缺失时降级处理） ───
try:
    import rasterio
    from rasterio.transform import from_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("[WARN] rasterio 未安装，将跳过 GeoTIFF 输出。可通过 conda install rasterio 安装。")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib 未安装，将跳过可视化。")

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════
# 一、全局配置（初学者可直接修改这里的参数）
# ══════════════════════════════════════════════════════════════════════

class Config:
    """集中管理所有超参数，方便修改和实验"""

    # --- 路径 ---
    project_dir = Path(__file__).parent.parent      # 项目根目录
    data_dir    = project_dir / "data"               # 数据存放目录
    model_dir   = project_dir / "models"             # 模型权重保存目录
    output_dir  = project_dir / "outputs"            # 推理输出目录

    # --- 数据集 ---
    img_size    = 256          # 输入图像尺寸（256×256）
    num_classes = 7            # LoveDA 共 7 类：背景/水体/植被/建筑/道路/农田/森林
    in_channels = 3            # RGB 三通道

    # --- 训练 ---
    batch_size  = 4            # 批大小（RTX 3050 6GB 适配）
    epochs      = 30           # 训练轮数
    lr          = 1e-4         # 初始学习率
    weight_decay = 1e-4        # L2 正则化
    patience    = 8            # 早停耐心值（验证 loss 不下降则停止）

    # --- 数据增强 ---
    use_augmentation = True    # 是否启用数据增强

    # --- 其他 ---
    num_workers = 4            # DataLoader 工作进程数
    seed        = 42           # 随机种子

    @classmethod
    def setup_dirs(cls):
        """创建必要的目录"""
        for d in [cls.data_dir, cls.model_dir, cls.output_dir]:
            d.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# 二、工具函数
# ══════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    """固定随机种子，保证实验可重复"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    """自动选择计算设备"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[INFO] 使用 GPU: {gpu_name} ({vram:.1f} GB)")
    else:
        device = torch.device("cpu")
        print("[WARN] 未检测到 GPU，将使用 CPU 训练（速度会很慢）")
    return device


# ══════════════════════════════════════════════════════════════════════
# 三、合成数据生成器（用于快速测试，无需下载数据集）
# ══════════════════════════════════════════════════════════════════════

class SyntheticLandCoverDataset:
    """
    生成模拟遥感影像与标签的合成数据集。

    设计思路：
    - 影像：用不同颜色的色块 + 高斯噪声模拟 7 种地物
    - 标签：每个像素对应一个类别编号 (0~6)
    - 虽然不是真实影像，但数据分布足以让模型学到"颜色→类别"的映射

    类别定义（与 LoveDA 对齐）：
    0=背景(灰白) 1=水体(蓝) 2=植被(绿) 3=建筑(红褐)
    4=道路(灰) 5=农田(黄绿) 6=森林(深绿)
    """

    # 每个类别的 RGB 基准颜色（归一化到 0~1）
    CLASS_COLORS = {
        0: np.array([200, 200, 200]) / 255.0,  # 背景 - 灰白
        1: np.array([30,  100, 200]) / 255.0,  # 水体 - 蓝色
        2: np.array([50,  180, 50])  / 255.0,  # 植被 - 绿色
        3: np.array([180, 80,  50])  / 255.0,  # 建筑 - 红褐
        4: np.array([140, 140, 140]) / 255.0,  # 道路 - 灰色
        5: np.array([180, 200, 50])  / 255.0,  # 农田 - 黄绿
        6: np.array([20,  100, 30])  / 255.0,  # 森林 - 深绿
    }

    def __init__(self, root_dir: Path, num_train=200, num_val=50, img_size=256):
        self.root_dir = Path(root_dir)
        self.num_train = num_train
        self.num_val = num_val
        self.img_size = img_size
        self._generate()

    def _generate(self):
        """生成合成数据集"""
        for split, count in [("train", self.num_train), ("val", self.num_val)]:
            img_dir = self.root_dir / split / "images"
            lbl_dir = self.root_dir / split / "labels"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            # 只在数据不存在时生成
            existing = list(img_dir.glob("*.npy"))
            if len(existing) >= count:
                print(f"[INFO] {split} 数据已存在（{len(existing)} 张），跳过生成")
                continue

            print(f"[INFO] 正在生成 {count} 张合成 {split} 数据...")
            for i in range(count):
                img, label = self._make_one_sample()
                np.save(img_dir / f"sample_{i:04d}.npy", img)
                np.save(lbl_dir / f"sample_{i:04d}.npy", label)

    def _make_one_sample(self):
        """
        生成一张合成样本。

        方法：
        1. 随机选取 3~5 个类别
        2. 用 Voronoi 图风格将图像分成多个区域
        3. 每个区域填充对应类别的颜色 + 噪声
        4. 添加轻微的空间模糊模拟真实影像质感
        """
        h, w = self.img_size, self.img_size
        image = np.zeros((h, w, 3), dtype=np.float32)
        label = np.zeros((h, w), dtype=np.int64)

        # 随机选几个类别
        n_regions = random.randint(3, 6)
        chosen_classes = random.sample(range(7), min(n_regions, 7))

        # 生成 Voronoi 风格的区域划分
        # 随机放置种子点，每个像素归最近的种子点所属
        seeds_x = [random.randint(0, w - 1) for _ in range(n_regions)]
        seeds_y = [random.randint(0, h - 1) for _ in range(n_regions)]

        # 用网格计算距离（比逐像素快很多）
        yy, xx = np.mgrid[0:h, 0:w]
        min_dist = np.full((h, w), float("inf"))

        for cls_id, sx, sy in zip(chosen_classes, seeds_x, seeds_y):
            dist = (xx - sx) ** 2 + (yy - sy) ** 2
            mask = dist < min_dist
            min_dist[mask] = dist[mask]
            label[mask] = cls_id

        # 根据标签填充颜色
        for cls_id in chosen_classes:
            mask = label == cls_id
            base_color = self.CLASS_COLORS[cls_id]
            for c in range(3):
                channel = np.random.normal(base_color[c], 0.05, (h, w))
                image[:, :, c][mask] = channel[mask]

        # 添加轻微模糊（模拟真实影像的空间混合效应）
        image = np.clip(image, 0, 1)
        # 简单的均值模糊
        kernel_size = 3
        image_padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode="edge")
        blurred = np.zeros_like(image)
        for dy in range(kernel_size):
            for dx in range(kernel_size):
                blurred += image_padded[dy:dy + h, dx:dx + w, :]
        image = (blurred / (kernel_size ** 2)).astype(np.float32)

        return image, label


# ══════════════════════════════════════════════════════════════════════
# 四、PyTorch 数据集类
# ══════════════════════════════════════════════════════════════════════

class LandCoverDataset(Dataset):
    """
    遥感语义分割数据集。

    支持两种数据格式：
    1. .npy 格式（合成数据）：直接加载 NumPy 数组
    2. .png/.tif 格式（真实数据）：用 PIL 或 rasterio 加载

    数据增强策略（仅在训练时启用）：
    - 随机水平翻转
    - 随机垂直翻转
    - 随机 90° 旋转
    - 颜色抖动（亮度/对比度）
    """

    def __init__(self, root_dir: Path, split: str = "train",
                 img_size: int = 256, augment: bool = True):
        self.img_dir = Path(root_dir) / split / "images"
        self.lbl_dir = Path(root_dir) / split / "labels"
        self.img_size = img_size
        self.augment = augment and (split == "train")

        # 收集所有样本路径
        self.samples = []
        for ext in ["*.npy", "*.png", "*.tif", "*.jpg"]:
            for img_path in sorted(self.img_dir.glob(ext)):
                stem = img_path.stem
                # 查找对应的标签文件
                for lbl_ext in [".npy", ".png", ".tif"]:
                    lbl_path = self.lbl_dir / f"{stem}{lbl_ext}"
                    if lbl_path.exists():
                        self.samples.append((img_path, lbl_path))
                        break

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"在 {self.img_dir} 中未找到任何数据文件。\n"
                f"请先运行合成数据生成器，或下载 LoveDA 数据集到该目录。"
            )

        print(f"[INFO] {split} 集：加载 {len(self.samples)} 个样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]

        # 加载图像
        if img_path.suffix == ".npy":
            image = np.load(img_path).astype(np.float32)
        else:
            image = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0

        # 加载标签
        if lbl_path.suffix == ".npy":
            label = np.load(lbl_path).astype(np.int64)
        else:
            label = np.array(Image.open(lbl_path)).astype(np.int64)
            # 如果是 RGB 标签图，需要转换为类别索引
            if label.ndim == 3:
                label = label[:, :, 0]  # 取第一个通道

        # 确保标签值在有效范围内
        label = np.clip(label, 0, Config.num_classes - 1)

        # 数据增强
        if self.augment:
            image, label = self._augment(image, label)

        # HWC → CHW（PyTorch 要求通道在前）
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        label = torch.from_numpy(label).long()

        return image, label

    def _augment(self, image, label):
        """简单的几何数据增强（保持图像和标签同步变换）"""
        # 随机水平翻转
        if random.random() > 0.5:
            image = np.flip(image, axis=1).copy()
            label = np.flip(label, axis=0).copy()

        # 随机垂直翻转
        if random.random() > 0.5:
            image = np.flip(image, axis=0).copy()
            label = np.flip(label, axis=1).copy()

        # 随机 90° 旋转
        k = random.randint(0, 3)
        if k > 0:
            image = np.rot90(image, k, axes=(0, 1)).copy()
            label = np.rot90(label, k, axes=(0, 1)).copy()

        # 颜色抖动（仅对图像）
        if random.random() > 0.5:
            brightness = random.uniform(0.9, 1.1)
            image = np.clip(image * brightness, 0, 1)

        return image, label


# ══════════════════════════════════════════════════════════════════════
# 五、U-Net 模型定义（从零手写，帮助理解架构）
# ══════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """
    基础卷积块：两次 3×3 卷积 + BatchNorm + ReLU

    这是 U-Net 的基本构建单元。每次卷积后做批归一化可以
    加速训练并稳定梯度，ReLU 引入非线性让网络能学习复杂模式。
    """

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


class DownBlock(nn.Module):
    """下采样块：MaxPool 2×2 → ConvBlock"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """
    上采样块：上采样 → 拼接编码器特征 → ConvBlock

    拼接（skip connection）是 U-Net 的核心思想：
    上采样恢复空间分辨率但丢失细节，拼接编码器的高分辨率特征
    可以补充边界和纹理信息，让分割结果更精细。

    通道数计算要点：
    - 双线性上采样不改变通道数，拼接后通道 = in_ch + skip_ch
    - 转置卷积将通道减半，拼接后通道 = in_ch//2 + skip_ch
    """

    def __init__(self, in_ch, skip_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = ConvBlock(in_ch + skip_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
            self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x1, x2):
        """x1: 来自深层的特征（上采样），x2: 来自编码器的跳跃连接特征"""
        x1 = self.up(x1)
        # 处理尺寸不匹配的情况（当输入不是 2 的幂次时）
        diff_h = x2.size(2) - x1.size(2)
        diff_w = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2,
                         diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([x2, x1], dim=1)  # 通道维度拼接
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net 语义分割网络

    架构概览（编码器-解码器 + 跳跃连接）：

    输入 (3, 256, 256)
    │
    ├─ Encoder1 (64, 256, 256)  ──────────────────────┐
    │   ↓ MaxPool                                      │ skip
    ├─ Encoder2 (128, 128, 128) ──────────────────┐    │
    │   ↓ MaxPool                                  │    │
    ├─ Encoder3 (256, 64, 64) ──────────────┐     │    │
    │   ↓ MaxPool                            │     │    │
    ├─ Encoder4 (512, 32, 32) ────────┐     │     │    │
    │   ↓ MaxPool                     │     │     │    │
    ├─ Bottleneck (1024, 16, 16)      │     │     │    │
    │   ↑ Upsample                    │     │     │    │
    ├─ Decoder4 (512, 32, 32) ←──────┘     │     │    │
    │   ↑ Upsample                          │     │    │
    ├─ Decoder3 (256, 64, 64) ←────────────┘     │    │
    │   ↑ Upsample                                │    │
    ├─ Decoder2 (128, 128, 128) ←─────────────────┘    │
    │   ↑ Upsample                                      │
    ├─ Decoder1 (64, 256, 256) ←────────────────────────┘
    │
    └─ 1×1 Conv → (num_classes, 256, 256)
    """

    def __init__(self, in_channels=3, num_classes=7):
        super().__init__()

        # 编码器（逐步下采样，提取高层语义特征）
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = DownBlock(64, 128)
        self.enc3 = DownBlock(128, 256)
        self.enc4 = DownBlock(256, 512)

        # 瓶颈层（最深层，全局语义信息）
        self.bottleneck = DownBlock(512, 1024)

        # 解码器（逐步上采样，恢复空间分辨率）
        # UpBlock(in_ch, skip_ch, out_ch) — skip_ch 是编码器对应层的输出通道
        self.up4 = UpBlock(1024, 512, 512)   # 1024(bottleneck) + 512(e4) → 512
        self.up3 = UpBlock(512, 256, 256)    # 512(up4) + 256(e3) → 256
        self.up2 = UpBlock(256, 128, 128)    # 256(up3) + 128(e2) → 128
        self.up1 = UpBlock(128, 64, 64)      # 128(up2) + 64(e1) → 64

        # 最终分类层（1×1 卷积，将特征映射到类别数）
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # 编码路径
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # 瓶颈
        b = self.bottleneck(e4)

        # 解码路径（拼接跳跃连接）
        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        # 像素级分类输出
        out = self.final(d1)
        return out


# ══════════════════════════════════════════════════════════════════════
# 六、损失函数与评价指标
# ══════════════════════════════════════════════════════════════════════

class CombinedLoss(nn.Module):
    """
    组合损失函数：交叉熵 + Dice Loss

    - 交叉熵（CrossEntropy）：逐像素分类损失，对每个像素预测与真值的差异
    - Dice Loss：基于集合重叠的损失，对类别不平衡问题效果好
    - 两者结合可以兼顾像素精度和区域一致性
    """

    def __init__(self, num_classes, class_weights=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32)
            if class_weights else None,
            ignore_index=255  # 忽略标记为 255 的像素
        )
        self.num_classes = num_classes

    def forward(self, pred, target):
        """
        pred:   (B, C, H, W) 模型输出的 logits
        target: (B, H, W)    真值标签（类别索引）
        """
        ce_loss = self.ce(pred, target)

        # Dice Loss 计算
        pred_soft = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(
            target.clamp(0, self.num_classes - 1), self.num_classes
        ).permute(0, 3, 1, 2).float()

        # 对每个类别计算 Dice，然后取平均
        dice = 0
        for c in range(self.num_classes):
            p = pred_soft[:, c]
            t = target_onehot[:, c]
            inter = (p * t).sum()
            union = p.sum() + t.sum()
            if union > 0:
                dice += (2 * inter + 1) / (union + 1)
        dice_loss = 1 - dice / self.num_classes

        return ce_loss + dice_loss


def compute_metrics(pred, target, num_classes):
    """
    计算语义分割评价指标。

    返回：
    - OA (Overall Accuracy)：总体精度
    - mIoU (mean Intersection over Union)：平均交并比
    - 每个类别的 IoU
    """
    pred = pred.argmax(dim=1)  # (B, H, W)

    # 混淆矩阵
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(target.view(-1), pred.view(-1)):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            confusion[t, p] += 1

    # 从混淆矩阵计算指标
    oa = confusion.diag().sum().float() / (confusion.sum() + 1e-8)

    ious = []
    for c in range(num_classes):
        tp = confusion[c, c].float()
        fp = confusion[:, c].sum().float() - tp
        fn = confusion[c, :].sum().float() - tp
        iou = tp / (tp + fp + fn + 1e-8)
        ious.append(iou.item())

    miou = np.mean(ious)
    return {"OA": oa.item(), "mIoU": miou, "class_IoU": ious}


# ══════════════════════════════════════════════════════════════════════
# 七、训练引擎
# ══════════════════════════════════════════════════════════════════════

class Trainer:
    """
    训练管理器：封装训练循环、验证、早停、学习率调度和模型保存。

    训练流程：
    1. 每个 epoch 遍历训练集，前向传播 → 计算损失 → 反向传播 → 更新权重
    2. 每个 epoch 结束在验证集上评估
    3. 根据验证 loss 决定是否保存最优模型和是否早停
    4. 使用余弦退火学习率调度器
    """

    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config

        # 损失函数
        self.criterion = CombinedLoss(config.num_classes).to(device)

        # AdamW 优化器（比 Adam 更好的权重衰减实现）
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

        # 余弦退火学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs, eta_min=config.lr * 0.01
        )

        # 训练历史
        self.history = {"train_loss": [], "val_loss": [], "val_miou": []}
        self.best_miou = 0
        self.patience_counter = 0

    def train_one_epoch(self):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        n_batches = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % 5 == 0:
                print(f"  Batch {batch_idx + 1}/{len(self.train_loader)} | "
                      f"Loss: {loss.item():.4f}")

        return total_loss / n_batches

    @torch.no_grad()
    def validate(self):
        """在验证集上评估"""
        self.model.eval()
        total_loss = 0
        all_metrics = []

        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            total_loss += loss.item()

            metrics = compute_metrics(outputs, labels, self.config.num_classes)
            all_metrics.append(metrics)

        avg_loss = total_loss / len(self.val_loader)
        avg_miou = np.mean([m["mIoU"] for m in all_metrics])
        avg_oa = np.mean([m["OA"] for m in all_metrics])

        return avg_loss, avg_miou, avg_oa

    def fit(self):
        """执行完整训练流程"""
        print(f"\n{'=' * 60}")
        print(f"开始训练 | Epochs: {self.config.epochs} | "
              f"Batch Size: {self.config.batch_size} | "
              f"Learning Rate: {self.config.lr}")
        print(f"{'=' * 60}\n")

        for epoch in range(self.config.epochs):
            start_time = time.time()

            # 训练
            train_loss = self.train_one_epoch()

            # 验证
            val_loss, val_miou, val_oa = self.validate()

            # 学习率调度
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # 记录历史
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_miou"].append(val_miou)

            elapsed = time.time() - start_time

            print(f"Epoch {epoch + 1:02d}/{self.config.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val mIoU: {val_miou:.4f} | "
                  f"Val OA: {val_oa:.4f} | "
                  f"LR: {current_lr:.6f} | "
                  f"Time: {elapsed:.1f}s")

            # 保存最优模型
            if val_miou > self.best_miou:
                self.best_miou = val_miou
                self.patience_counter = 0
                save_path = self.config.model_dir / "best_unet.pth"
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_miou": val_miou,
                }, save_path)
                print(f"  ✓ 保存最优模型 (mIoU={val_miou:.4f})")
            else:
                self.patience_counter += 1

            # 早停检查
            if self.patience_counter >= self.config.patience:
                print(f"\n[EARLY STOP] 验证指标连续 {self.config.patience} 轮未提升，停止训练")
                break

        print(f"\n训练完成！最佳验证 mIoU: {self.best_miou:.4f}")
        return self.history


# ══════════════════════════════════════════════════════════════════════
# 八、推理与可视化
# ══════════════════════════════════════════════════════════════════════

# LoveDA 类别配色方案
CLASS_NAMES = ["背景", "水体", "植被", "建筑", "道路", "农田", "森林"]
CLASS_COLORS_VIZ = [
    [200, 200, 200],  # 背景 - 灰白
    [30,  100, 200],  # 水体 - 蓝色
    [50,  180, 50],   # 植被 - 绿色
    [180, 80,  50],   # 建筑 - 红褐
    [140, 140, 140],  # 道路 - 灰色
    [180, 200, 50],   # 农田 - 黄绿
    [20,  100, 30],   # 森林 - 深绿
]


def colorize_label(label, colors=CLASS_COLORS_VIZ):
    """将类别标签转换为彩色 RGB 图像用于可视化"""
    h, w = label.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c, color in enumerate(colors):
        mask = label == c
        rgb[mask] = color
    return rgb


@torch.no_grad()
def run_inference(model, image_tensor, device):
    """
    对单张图像执行推理。

    输入：(1, 3, H, W) 的图像张量
    输出：(H, W) 的预测标签数组
    """
    model.eval()
    image = image_tensor.unsqueeze(0).to(device)
    output = model(image)
    pred = output.argmax(dim=1).squeeze(0).cpu().numpy()
    return pred


def visualize_results(dataset, model, device, output_dir, num_samples=5):
    """
    可视化推理结果：原图 | 真值标签 | 预测标签 | 叠加对比

    每张结果图包含四个子图，方便直观对比模型预测效果。
    """
    if not HAS_MPL:
        print("[WARN] matplotlib 未安装，跳过可视化")
        return

    model.eval()
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):
        image, label = dataset[idx]
        pred = run_inference(model, image, device)
        image_np = image.permute(1, 2, 0).numpy()

        # 原图
        axes[row, 0].imshow(image_np)
        axes[row, 0].set_title("原始影像", fontsize=14)
        axes[row, 0].axis("off")

        # 真值
        axes[row, 1].imshow(colorize_label(label.numpy()))
        axes[row, 1].set_title("真值标签", fontsize=14)
        axes[row, 1].axis("off")

        # 预测
        axes[row, 2].imshow(colorize_label(pred))
        axes[row, 2].set_title("模型预测", fontsize=14)
        axes[row, 2].axis("off")

        # 叠加对比
        overlay = (image_np * 0.5 + colorize_label(pred) / 255.0 * 0.5)
        axes[row, 3].imshow(np.clip(overlay, 0, 1))
        axes[row, 3].set_title("叠加对比", fontsize=14)
        axes[row, 3].axis("off")

    plt.tight_layout()
    save_path = output_dir / "segmentation_results.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] 可视化结果已保存: {save_path}")


def plot_training_history(history, output_dir):
    """绘制训练过程曲线图"""
    if not HAS_MPL:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 损失曲线
    ax1.plot(history["train_loss"], label="训练损失", linewidth=2)
    ax1.plot(history["val_loss"], label="验证损失", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("损失曲线", fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # mIoU 曲线
    ax2.plot(history["val_miou"], label="验证 mIoU", linewidth=2, color="green")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("mIoU", fontsize=12)
    ax2.set_title("精度曲线 (mIoU)", fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / "training_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] 训练曲线已保存: {save_path}")


# ══════════════════════════════════════════════════════════════════════
# 九、LoveDA 数据集下载器
# ══════════════════════════════════════════════════════════════════════

def download_loveda_small(data_dir: Path):
    """
    下载 LoveDA 数据集的一个小型子集用于学习。

    LoveDA 数据集信息：
    - 来源：武汉大学 YANG 团队发布
    - 覆盖：南京、武汉、徐州城市与农村区域
    - 分辨率：0.3m（城市）/ 0.1m（农村）
    - 类别：7 类（背景/水体/植被/建筑/道路/农田/森林）

    注意：完整版 LoveDA 约 1.5GB，这里我们使用合成数据进行演示。
    如需真实数据，请访问：https://github.com/Junjue-Wang/LoveDA
    """
    print("""
╔══════════════════════════════════════════════════════════════╗
║  LoveDA 遥感语义分割数据集                                    ║
║                                                              ║
║  完整数据集下载：                                              ║
║  https://github.com/Junjue-Wang/LoveDA                       ║
║                                                              ║
║  当前使用合成数据进行演示。合成数据的特点：                       ║
║  • 模拟了 LoveDA 的 7 类地物颜色分布                           ║
║  • 使用 Voronoi 区域划分模拟不同地物的空间分布                  ║
║  • 足以验证模型架构和训练流程的正确性                            ║
║                                                              ║
║  如需在真实数据上训练，请将下载的数据解压到：                     ║
║  {data_dir}/train/images/                                    ║
║  {data_dir}/train/labels/                                    ║
║  {data_dir}/val/images/                                      ║
║  {data_dir}/val/labels/                                      ║
╚══════════════════════════════════════════════════════════════╝
    """.format(data_dir=data_dir))


# ══════════════════════════════════════════════════════════════════════
# 十、主函数：串联完整流程
# ══════════════════════════════════════════════════════════════════════

def main():
    """
    主流程：配置 → 数据准备 → 模型构建 → 训练 → 评估 → 推理可视化

    这是一个端到端的语义分割项目流程，涵盖了从数据准备到结果展示的完整链路。
    """

    print("""
╔══════════════════════════════════════════════════════════════╗
║          GeoAI 语义分割训练完整示例                            ║
║          任务：土地覆盖分类 (Land Cover Classification)        ║
║          模型：U-Net                                         ║
║          数据集：LoveDA / 合成数据                             ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # ── 1. 配置 ──
    config = Config
    config.setup_dirs()
    set_seed(config.seed)
    device = get_device()

    # ── 2. 数据准备 ──
    print("\n" + "=" * 60)
    print("步骤 1/5：数据准备")
    print("=" * 60)

    # 检查是否有真实数据，否则生成合成数据
    train_img_dir = config.data_dir / "train" / "images"
    if not train_img_dir.exists() or len(list(train_img_dir.glob("*"))) == 0:
        print("[INFO] 未检测到真实数据，生成合成数据集用于演示...")
        download_loveda_small(config.data_dir)
        SyntheticLandCoverDataset(
            config.data_dir,
            num_train=200,
            num_val=50,
            img_size=config.img_size
        )

    # 创建数据集和数据加载器
    train_dataset = LandCoverDataset(
        config.data_dir, split="train",
        img_size=config.img_size, augment=config.use_augmentation
    )
    val_dataset = LandCoverDataset(
        config.data_dir, split="val",
        img_size=config.img_size, augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    # ── 3. 构建模型 ──
    print("\n" + "=" * 60)
    print("步骤 2/5：构建 U-Net 模型")
    print("=" * 60)

    model = UNet(
        in_channels=config.in_channels,
        num_classes=config.num_classes
    )

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量：总计 {total_params:,} | 可训练 {trainable_params:,}")
    print(f"模型大小：约 {total_params * 4 / 1024 / 1024:.1f} MB（FP32）")

    # ── 4. 训练 ──
    print("\n" + "=" * 60)
    print("步骤 3/5：模型训练")
    print("=" * 60)

    trainer = Trainer(model, train_loader, val_loader, device, config)
    history = trainer.fit()

    # ── 5. 评估 ──
    print("\n" + "=" * 60)
    print("步骤 4/5：模型评估")
    print("=" * 60)

    # 加载最优模型
    best_path = config.model_dir / "best_unet.pth"
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"已加载最优模型 (Epoch {checkpoint['epoch']}, mIoU={checkpoint['best_miou']:.4f})")

    val_loss, val_miou, val_oa = trainer.validate()
    print(f"\n最终评估结果：")
    print(f"  验证 mIoU: {val_miou:.4f}")
    print(f"  验证 OA:   {val_oa:.4f}")

    # ── 6. 推理可视化 ──
    print("\n" + "=" * 60)
    print("步骤 5/5：推理与可视化")
    print("=" * 60)

    visualize_results(
        val_dataset, model, device,
        config.output_dir, num_samples=4
    )
    plot_training_history(history, config.output_dir)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  训练完成！                                                   ║
║                                                              ║
║  输出文件：                                                    ║
║  • 最优模型权重：{config.model_dir}/best_unet.pth                       ║
║  • 推理可视化图：{config.output_dir}/segmentation_results.png             ║
║  • 训练曲线图：  {config.output_dir}/training_curves.png                  ║
╚══════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
