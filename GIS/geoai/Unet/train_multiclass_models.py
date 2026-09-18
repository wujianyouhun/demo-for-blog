"""使用已制作的多分类样本，依次训练六种 U-Net 变体并保存 checkpoint。"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from unet_models import MODELS, build_model


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "dataset" / "qilian_multiclass_5class"
DEFAULT_MODELS = ROOT.parent / "models" / "Unet" / "qilian_multiclass_5class"
CLASS_NAMES = ("road", "built_up", "cropland", "bare_land", "solar_farm")
IGNORE_VALUE = 255


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 U-Net、ResUNet、Attention U-Net、U-Net++、TransUNet、Swin-UNet")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--models", nargs="+", choices=tuple(MODELS) + ("all",), default=["all"])
    parser.add_argument("--epochs", type=int, default=20, help="每个模型训练轮数；流程试跑建议 3–10")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=16, help="CPU 流程试跑可设为 8；正式实验建议 16 或 32")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto", help="auto、cpu 或 cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖同名模型目录")
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else ("cpu" if value == "auto" else value))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalise(image: np.ndarray) -> np.ndarray:
    """按每个影像块各波段 99 分位归一化，保持 0 填充区为 0。"""
    image = np.nan_to_num(image.astype("float32"), nan=0.0)
    upper = np.percentile(image, 99, axis=(1, 2), keepdims=True)
    return np.clip(image / np.maximum(upper, 1e-6), 0.0, 1.0)


class SegmentationDataset(Dataset):
    def __init__(self, root: Path, split: str, augment: bool = False):
        self.root = root / split
        self.augment = augment
        images = sorted((self.root / "images").glob("*.tif"))
        self.items = []
        for image in images:
            mask = self.root / "masks" / f"{image.stem}_mask.tif"
            if not mask.is_file():
                raise FileNotFoundError(f"缺少对应标签：{mask}")
            self.items.append((image, mask))
        if not self.items:
            raise FileNotFoundError(f"{self.root} 中没有影像 / 标签样本。")
        with rasterio.open(self.items[0][0]) as source:
            self.in_channels = source.count

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        image_path, mask_path = self.items[index]
        with rasterio.open(image_path) as source:
            image = normalise(source.read())
        with rasterio.open(mask_path) as source:
            mask = source.read(1).astype("int64")
        if self.augment:
            if random.random() < 0.5:
                image, mask = image[:, :, ::-1].copy(), mask[:, ::-1].copy()
            if random.random() < 0.5:
                image, mask = image[:, ::-1, :].copy(), mask[::-1, :].copy()
            rotations = random.randint(0, 3)
            if rotations:
                image, mask = np.rot90(image, rotations, axes=(1, 2)).copy(), np.rot90(mask, rotations).copy()
        return torch.from_numpy(image), torch.from_numpy(mask)


def class_weights(dataset: SegmentationDataset, num_classes: int) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, mask_path in dataset.items:
        with rasterio.open(mask_path) as source:
            mask = source.read(1)
        for class_id in range(num_classes):
            counts[class_id] += np.count_nonzero(mask == class_id)
    if np.any(counts == 0):
        missing = [CLASS_NAMES[index] for index, count in enumerate(counts) if count == 0]
        raise ValueError(f"训练集缺少类别：{', '.join(missing)}")
    # sqrt 反频率避免道路等少数类的权重过大导致训练不稳定。
    weights = np.sqrt(counts.max() / counts.astype("float64"))
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> dict[str, float]:
    model.eval()
    intersections = np.zeros(num_classes, dtype=np.float64)
    unions = np.zeros(num_classes, dtype=np.float64)
    for images, masks in loader:
        prediction = model(images.to(device)).argmax(dim=1).cpu().numpy()
        target = masks.numpy()
        valid = target != IGNORE_VALUE
        for class_id in range(num_classes):
            predicted = np.logical_and(prediction == class_id, valid)
            expected = np.logical_and(target == class_id, valid)
            intersections[class_id] += np.logical_and(predicted, expected).sum()
            unions[class_id] += np.logical_or(predicted, expected).sum()
    iou = intersections / np.maximum(unions, 1)
    result = {f"iou_{CLASS_NAMES[index]}": float(iou[index]) for index in range(num_classes)}
    result["miou"] = float(iou.mean())
    return result


def train_one(name: str, train_set: SegmentationDataset, val_set: SegmentationDataset, args: argparse.Namespace,
              device: torch.device) -> dict:
    num_classes = len(CLASS_NAMES)
    target = args.output_dir / name
    if target.exists() and any(target.iterdir()) and not args.overwrite:
        raise FileExistsError(f"模型目录已存在：{target}。确认后增加 --overwrite。")
    target.mkdir(parents=True, exist_ok=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = build_model(name, in_channels=train_set.in_channels, base=args.base_channels, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_set, num_classes).to(device), ignore_index=IGNORE_VALUE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_miou, history = -1.0, []
    started = time.time()
    print(f"\n训练 {name}：{sum(parameter.numel() for parameter in model.parameters()):,} 参数，设备={device}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(device))
            loss = criterion(logits, masks.to(device))
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
        scheduler.step()
        metrics = evaluate(model, val_loader, device, num_classes)
        record = {"epoch": epoch, "loss": running_loss / len(train_loader), **metrics}
        history.append(record)
        print(f"  epoch {epoch:02d}/{args.epochs}: loss={record['loss']:.4f}, val_mIoU={record['miou']:.4f}")
        if metrics["miou"] > best_miou:
            best_miou = metrics["miou"]
            torch.save({
                "architecture": name, "in_channels": train_set.in_channels, "num_classes": num_classes,
                "class_names": CLASS_NAMES, "ignore_index": IGNORE_VALUE, "base_channels": args.base_channels,
                "normalisation": "per_tile_p99", "best_epoch": epoch, "best_miou": best_miou,
                "state_dict": model.state_dict(),
            }, target / "best.pth")
    result = {"model": name, "best_miou": best_miou, "seconds": round(time.time() - started, 2), "history": history}
    (target / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.base_channels < 4:
        raise ValueError("epochs、batch-size 至少为 1，base-channels 至少为 4。")
    requested = tuple(MODELS) if "all" in args.models else tuple(args.models)
    train_set = SegmentationDataset(args.dataset_dir, "train", augment=True)
    val_set = SegmentationDataset(args.dataset_dir, "val", augment=False)
    if train_set.in_channels != val_set.in_channels:
        raise ValueError("训练集与验证集的影像波段数不一致。")
    set_seed(args.seed)
    device = choose_device(args.device)
    print(f"数据集：train={len(train_set)}，val={len(val_set)}，波段={train_set.in_channels}，类别={len(CLASS_NAMES)}")
    results = [train_one(name, train_set, val_set, args, device) for name in requested]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n全部模型训练完成，权重目录：{args.output_dir}")


if __name__ == "__main__":
    main()
