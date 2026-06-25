"""
GeoAI 图像分类 — 数据集模块
支持 EuroSAT（10类遥感图像）数据加载与增强
目录结构:
    data/processed/EuroSAT/
    ├── train/  ├── val/  └── test/
        ├── AnnualCrop/  ├── Forest/  └── ...
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Callable, Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

CLASS_NAMES: list[str] = [
    "AnnualCrop","Forest","HerbaceousVegetation","Highway",
    "Industrial","Pasture","PermanentCrop","Residential",
    "River","SeaLake",
]
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASS_NAMES)}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

get_inference_transforms = get_val_transforms

class EuroSATDataset(Dataset):
    """EuroSAT 遥感图像分类数据集"""
    def __init__(self, root: str | Path, split: str = "train",
                 transform: Optional[Callable] = None):
        self.root = Path(root) / split
        self.split = split
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        if not self.root.exists():
            raise FileNotFoundError(
                f"数据集目录不存在: {self.root}\n请先运行: python scripts/download_dataset.py")
        for cls_name in CLASS_NAMES:
            cls_dir = self.root / cls_name
            if not cls_dir.exists(): continue
            label = CLASS_TO_IDX[cls_name]
            for ext in ("*.jpg","*.jpeg","*.png","*.tif","*.tiff"):
                for img_path in sorted(cls_dir.glob(ext)):
                    self.samples.append((img_path, label))
        if len(self.samples) == 0:
            raise RuntimeError(f"在 {self.root} 下未找到任何图像")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def get_class_distribution(self) -> dict[str, int]:
        from collections import Counter
        return dict(Counter(CLASS_NAMES[label] for _, label in self.samples))

def build_dataloaders(data_root: str | Path, batch_size: int = 32,
                      image_size: int = 224, num_workers: int = 4) -> dict[str, DataLoader]:
    """返回 {"train": ..., "val": ..., "test": ...}"""
    tfm = {
        "train": get_train_transforms(image_size),
        "val":   get_val_transforms(image_size),
        "test":  get_val_transforms(image_size),
    }
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        ds = EuroSATDataset(data_root, split=split, transform=tfm[split])
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split=="train"),
            num_workers=num_workers, pin_memory=True, drop_last=(split=="train"))
        print(f"  [{split:5s}] {len(ds):6d} 样本  → {len(loaders[split])} 批次")
    return loaders

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent / "data" / "processed" / "EuroSAT"
    loaders = build_dataloaders(root, batch_size=8, num_workers=0)
    imgs, labels = next(iter(loaders["train"]))
    print(f"批次形状: {imgs.shape}  标签: {labels}")
