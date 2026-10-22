"""使用 RGB GeoTIFF 切片及整数类别标签微调 DeepLabV3。"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from common import DEFAULT_CHECKPOINT, IGNORE, ROOT, choose_device, to_tensor, training_model


class TileDataset(Dataset):
    """train/val 下 images/*.tif 对应 masks/<stem>_mask.tif。"""

    def __init__(self, root, split, bands, scale, size, num_classes):
        self.items = []
        self.bands, self.scale, self.size = bands, scale, size
        self.num_classes = num_classes
        for image in sorted((root / split / "images").glob("*.tif")):
            mask = root / split / "masks" / f"{image.stem}_mask.tif"
            if not mask.is_file():
                raise FileNotFoundError(f"缺少标签：{mask}")
            self.items.append((image, mask))
        if not self.items:
            raise ValueError(f"{root / split} 没有训练切片。")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image_path, mask_path = self.items[index]
        with rasterio.open(image_path) as src, rasterio.open(mask_path) as labels:
            if labels.count != 1:
                raise ValueError(f"标签必须是单波段类别编号栅格：{mask_path}")
            if (src.shape != labels.shape or src.crs != labels.crs
                    or not src.transform.almost_equals(labels.transform)):
                raise ValueError(f"标签与影像的像元网格不一致：{mask_path}")
            image = src.read(self.bands)
            valid = (src.read_masks(self.bands) > 0).all(axis=0) & np.isfinite(image).all(axis=0)
            raw_mask = labels.read(1)
            valid &= labels.read_masks(1) > 0
        if not np.isfinite(raw_mask).all() or not np.equal(raw_mask, np.floor(raw_mask)).all():
            raise ValueError(f"标签必须为整数类别：{mask_path}")
        mask = raw_mask.astype("int64")
        mask[~valid] = IGNORE
        if np.any((mask != IGNORE) & ((mask < 0) | (mask >= self.num_classes))):
            raise ValueError(f"标签超出 0..{self.num_classes - 1}：{mask_path}")
        if mask.shape != (self.size, self.size):
            raise ValueError(f"切片须为 {self.size}×{self.size}：{image_path}")
        return to_tensor(image, self.scale), torch.from_numpy(mask.copy())


def run_epoch(model, loader, device, num_classes, optimizer=None, class_weights=None, return_details=False):
    model.train(optimizer is not None)
    # ASPP 全局池化分支产生 1×1 特征；冻结 BN 统计量支持 batch_size=1。
    if optimizer is not None:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    total_loss, pixels = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for image, target in loader:
            image, target = image.to(device), target.to(device)
            valid = target != IGNORE
            count = int(valid.sum().item())
            if count == 0:
                continue
            logits = model(image)["out"]
            loss = nn.functional.cross_entropy(logits, target, ignore_index=IGNORE, weight=class_weights)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            prediction = logits.detach().argmax(1)
            codes = target[valid] * num_classes + prediction[valid]
            confusion += torch.bincount(codes, minlength=num_classes ** 2).reshape(num_classes, num_classes)
            total_loss += loss.item() * count
            pixels += count
    if not pixels:
        raise ValueError("数据集全部为无效 / 未标注像元。")
    matrix = confusion.double()
    union = matrix.sum(0) + matrix.sum(1) - matrix.diag()
    present = union > 0
    miou = (matrix.diag()[present] / union[present]).mean().item()
    if return_details:
        per_class_iou = [(matrix[i, i] / union[i]).item() if union[i] > 0 else None for i in range(num_classes)]
        return total_loss / pixels, miou, dict(confusion_matrix=confusion.cpu().tolist(), per_class_iou=per_class_iou)
    return total_loss / pixels, miou


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data" / "demo")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--class-names", nargs="+", default=None, help="优先读取 dataset.json，否则使用合成数据的三类")
    parser.add_argument("--bands", nargs=3, type=int, default=None, help="RGB 对应的 1 起始波段号，优先读取 dataset.json")
    parser.add_argument("--scale", type=float, default=None, help="优先读取 dataset.json；无配置时默认 255")
    parser.add_argument("--tile-size", type=int, default=None, help="优先读取 dataset.json；无配置时默认 256")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-weighting", choices=["none", "inverse-sqrt"], default="none",
                        help="inverse-sqrt 根据训练集有效像元计数缓解类别不平衡")
    parser.add_argument("--from-scratch", action="store_true", help="随机初始化，仅适合流程检查或自行充分训练")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有 checkpoint")
    args = parser.parse_args()
    config_path = args.dataset_dir / "dataset.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    args.class_names = args.class_names if args.class_names is not None else config.get("class_names", ["background", "vegetation", "building"])
    args.bands = args.bands if args.bands is not None else config.get("bands", [1, 2, 3])
    args.scale = args.scale if args.scale is not None else config.get("scale", 255.)
    args.tile_size = args.tile_size if args.tile_size is not None else config.get("tile_size", 256)
    if args.checkpoint.exists() and not args.overwrite:
        raise FileExistsError(f"权重已存在：{args.checkpoint}；更换路径或使用 --overwrite。")
    if (args.epochs < 1 or args.batch_size < 1 or args.tile_size < 32 or min(args.bands) < 1
            or not np.isfinite(args.scale) or args.scale <= 0
            or not np.isfinite(args.learning_rate) or args.learning_rate <= 0):
        parser.error("epochs/batch-size/bands/scale/learning-rate 必须为正，tile-size 至少为 32。")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    num_classes = len(args.class_names)
    model = training_model(num_classes, pretrained=not args.from_scratch).to(device)
    loaders = [DataLoader(TileDataset(args.dataset_dir, split, args.bands, args.scale, args.tile_size, num_classes),
                          batch_size=args.batch_size, shuffle=split == "train", num_workers=0)
               for split in ("train", "val")]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    class_weights = None
    if args.class_weighting == "inverse-sqrt":
        pixel_counts = torch.zeros(num_classes, dtype=torch.float64)
        for _, mask in loaders[0].dataset:
            pixel_counts += torch.bincount(mask[mask != IGNORE], minlength=num_classes)
        if (pixel_counts == 0).any():
            raise ValueError("训练集中有类别无有效像元，无法进行类别加权。")
        class_weights = pixel_counts.rsqrt()
        class_weights = (class_weights / class_weights.mean()).float().to(device)
        print("训练类别权重：", class_weights.cpu().tolist(), flush=True)
    print(f"设备：{device}，类别：{args.class_names}，波段：{args.bands}，scale={args.scale}，"
          f"train={len(loaders[0].dataset)}，val={len(loaders[1].dataset)}", flush=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        train_loss, _ = run_epoch(model, loaders[0], device, num_classes, optimizer, class_weights)
        val_loss, miou, details = run_epoch(model.eval(), loaders[1], device, num_classes,
                                          class_weights=class_weights, return_details=True)
        record = dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss, val_miou=miou)
        record.update(details)
        history.append(record)
        print(json.dumps(record), flush=True)
        if miou > best:
            best = miou
            checkpoint = dict(
                architecture="deeplabv3_resnet50", model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                class_names=args.class_names, bands=args.bands, scale=args.scale,
                tile_size=args.tile_size, ignore_index=IGNORE, epoch=epoch, val_miou=miou,
                initialization="random" if args.from_scratch else "COCO_WITH_VOC_LABELS_V1",
                dataset_dir=str(args.dataset_dir.resolve()), class_labels=config.get("class_labels"),
                split_strategy=config.get("split_strategy"), preprocessing_note=config.get("preprocessing_note"),
                class_weighting=args.class_weighting,
                class_weights=class_weights.cpu().tolist() if class_weights is not None else None,
                validation=details, seed=args.seed,
            )
            temporary = args.checkpoint.with_suffix(".tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(args.checkpoint)
        args.checkpoint.with_suffix(".history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"最佳验证 mIoU={best:.4f}；权重：{args.checkpoint.resolve()}")


if __name__ == "__main__":
    main()
