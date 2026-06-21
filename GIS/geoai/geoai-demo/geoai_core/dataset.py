"""GeoAI Core - 土地覆盖分类数据集与数据加载器"""
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader, random_split

logger = logging.getLogger(__name__)


def get_augmentation():
    """
    返回训练时的数据增强 pipeline (albumentations)。
    包含: HorizontalFlip, VerticalFlip, RandomRotate90,
          ShiftScaleRotate, RandomBrightnessContrast
    """
    try:
        import albumentations as A
    except ImportError:
        raise ImportError("需要安装 albumentations: pip install albumentations")

    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.5
        ),
    ])


class LandCoverDataset(Dataset):
    """
    土地覆盖语义分割数据集。

    从 image_dir 加载影像瓦片，从 label_dir 加载标签掩膜。
    影像和标签按文件名对应。
    """

    def __init__(
        self,
        image_dir: str | Path,
        label_dir: str | Path,
        transform=None,
    ):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform

        # 收集所有影像文件（按文件名排序）
        self.image_paths = sorted(
            list(self.image_dir.glob("*.tif"))
            + list(self.image_dir.glob("*.tiff"))
            + list(self.image_dir.glob("*.png"))
            + list(self.image_dir.glob("*.jpg"))
        )

        if not self.image_paths:
            raise ValueError(f"影像目录中没有找到文件: {self.image_dir}")

        logger.info("数据集加载完成: %d 个样本 (%s)", len(self), self.image_dir)

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_image(self, path: Path) -> np.ndarray:
        """读取影像并转换为 HWC, float32, [0,1] 格式。"""
        if path.suffix.lower() in (".tif", ".tiff"):
            with rasterio.open(path) as src:
                img = src.read()  # (C, H, W)
                img = img.astype(np.float32)
                # 归一化到 [0, 1]
                if img.max() > 1.0:
                    img = img / 255.0
                img = np.transpose(img, (1, 2, 0))  # -> (H, W, C)
        else:
            from PIL import Image
            img = np.array(Image.open(path).convert("RGB"), dtype=np.float32)
            img = img / 255.0

        return img

    def _load_label(self, image_path: Path) -> np.ndarray:
        """读取标签掩膜，返回 H, int64 类别索引。"""
        # 匹配标签文件名
        label_path = self.label_dir / image_path.name
        if not label_path.exists():
            # 尝试常见命名变体
            for suffix in [".tif", ".tiff", ".png"]:
                alt = self.label_dir / (image_path.stem + suffix)
                if alt.exists():
                    label_path = alt
                    break

        if not label_path.exists():
            raise FileNotFoundError(
                f"找不到对应标签: {image_path.name} -> {self.label_dir}"
            )

        if label_path.suffix.lower() in (".tif", ".tiff"):
            with rasterio.open(label_path) as src:
                label = src.read(1)  # (H, W)
        else:
            from PIL import Image
            label = np.array(Image.open(label_path), dtype=np.int64)

        # 确保类别索引为 int64
        label = label.astype(np.int64)
        return label

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]

        image = self._load_image(img_path)  # (H, W, C), float32
        label = self._load_label(img_path)   # (H, W), int64

        # 数据增强
        if self.transform is not None:
            augmented = self.transform(image=image, mask=label)
            image = augmented["image"]
            label = augmented["mask"]

        # 转换为 tensor
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(
                np.transpose(image, (2, 0, 1)).copy()
            ).float()
        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label.copy()).long()

        return image, label


def create_dataloaders(
    image_dir: str | Path,
    label_dir: str | Path,
    batch_size: int = 8,
    val_split: float = 0.15,
    num_workers: int = 4,
    augmentation: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证 DataLoader。

    Args:
        image_dir: 影像瓦片目录
        label_dir: 标签瓦片目录
        batch_size: 批大小
        val_split: 验证集比例
        num_workers: 工作进程数
        augmentation: 是否对训练集应用数据增强

    Returns:
        (train_loader, val_loader)
    """
    # 完整数据集（无增强）用于划分
    full_dataset = LandCoverDataset(image_dir, label_dir, transform=None)
    total = len(full_dataset)
    val_size = int(total * val_split)
    train_size = total - val_size

    if train_size == 0:
        raise ValueError(f"样本数不足: 总共 {total} 个, 无法划分训练集")

    train_subset, val_subset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # 训练集应用增强
    if augmentation:
        train_transform = get_augmentation()
    else:
        train_transform = None

    # 包装为带增强的数据集
    train_dataset = _AugmentedSubset(train_subset, train_transform)
    val_dataset = _AugmentedSubset(val_subset, transform=None)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    logger.info(
        "DataLoader 创建完成: 训练=%d, 验证=%d, batch=%d",
        train_size, val_size, batch_size,
    )
    return train_loader, val_loader


class _AugmentedSubset(Dataset):
    """对 Subset 应用数据增强的包装器。"""

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        # image 已经是 tensor, 需要转回 numpy 做增强
        if self.transform is not None:
            img_np = image.permute(1, 2, 0).numpy()  # (H, W, C)
            lbl_np = label.numpy()
            aug = self.transform(image=img_np, mask=lbl_np)
            image = torch.from_numpy(
                np.transpose(aug["image"], (2, 0, 1)).copy()
            ).float()
            label = torch.from_numpy(aug["mask"].copy()).long()
        return image, label
