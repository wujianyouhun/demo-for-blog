"""双时相数据集"""
import logging
from pathlib import Path
from typing import Optional, Callable, Tuple
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader, random_split

logger = logging.getLogger(__name__)


class BiTemporalDataset(Dataset):
    def __init__(self, dir_a, dir_b, label_dir, transform=None, file_suffix=".tif"):
        self.dir_a, self.dir_b, self.label_dir = Path(dir_a), Path(dir_b), Path(label_dir)
        self.transform = transform
        self.files_a = sorted(self.dir_a.glob(f"*{file_suffix}")) if self.dir_a.is_dir() else []
        self.files_b = sorted(self.dir_b.glob(f"*{file_suffix}")) if self.dir_b.is_dir() else []
        self.labels = sorted(self.label_dir.glob(f"*{file_suffix}")) if self.label_dir.is_dir() else []
        na, nb, nl = len(self.files_a), len(self.files_b), len(self.labels)
        if na == 0:
            raise ValueError(
                f"样本目录中没有找到 .tif 文件，请先准备训练样本。\n"
                f"  时相A目录: {self.dir_a}\n"
                f"  时相B目录: {self.dir_b}\n"
                f"  标签目录:   {self.label_dir}"
            )
        if na != nb or na != nl:
            raise ValueError(
                f"样本数量不匹配: 时相A={na}, 时相B={nb}, 标签={nl}。\n"
                f"请确保三个目录中的 .tif 文件数量一致。"
            )
        logger.info(f"数据集: {na} 对样本")

    def __len__(self):
        return len(self.files_a)

    def __getitem__(self, idx):
        img_a = self._read_img(self.files_a[idx])
        img_b = self._read_img(self.files_b[idx])
        label = self._read_lbl(self.labels[idx])

        a_hwc = np.transpose(img_a, (1, 2, 0))
        b_hwc = np.transpose(img_b, (1, 2, 0))

        if self.transform:
            c = a_hwc.shape[2]
            combined = np.concatenate([a_hwc, b_hwc], axis=2)
            aug = self.transform(image=combined, mask=label)
            combined, label = aug["image"], aug["mask"]
            a_hwc, b_hwc = combined[:, :, :c], combined[:, :, c:]

        img_a = torch.from_numpy(np.ascontiguousarray(np.transpose(a_hwc, (2, 0, 1)))).float()
        img_b = torch.from_numpy(np.ascontiguousarray(np.transpose(b_hwc, (2, 0, 1)))).float()
        label = torch.from_numpy(np.ascontiguousarray(label)).long()

        return {"image_a": img_a, "image_b": img_b, "label": label, "filename": self.files_a[idx].name}

    @staticmethod
    def _read_img(path):
        with rasterio.open(path) as src:
            d = src.read().astype(np.float32)
            return d / 255.0 if d.max() > 1 else d

    @staticmethod
    def _read_lbl(path):
        with rasterio.open(path) as src:
            return np.clip(src.read(1).astype(np.int64), 0, 1)


def get_augmentation():
    import albumentations as A
    return A.Compose([
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.3), A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
    ])


def create_dataloaders(dir_a, dir_b, label_dir, batch_size=8, val_split=0.15,
                       num_workers=4, augment=True):
    transform = get_augmentation() if augment else None
    full = BiTemporalDataset(dir_a, dir_b, label_dir, transform)
    total = len(full)
    if total < 2:
        raise ValueError(
            f"训练至少需要 2 对样本，当前只有 {total} 对。"
            "请先运行 scripts/generate_sample.py --mode synthetic 或制作真实/弱标签样本。"
        )
    val_size = max(1, int(total * val_split))
    if val_size >= total:
        val_size = 1
    train_size = total - val_size
    train_ds, val_ds = random_split(full, [total - val_size, val_size],
                                     generator=torch.Generator().manual_seed(42))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                   pin_memory=True, drop_last=train_size >= batch_size),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    )
