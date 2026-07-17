from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.deeplabv3_plus import DeepLabV3Plus
from models.unet_plus_plus import UNetPlusPlus
from utils import SegDataset, calculate_metrics, split_indices

ROOT = Path(__file__).resolve().parent
GEOAI_ROOT = ROOT.parent / "geoai"
DEFAULT_IMAGES = GEOAI_ROOT / "makelable" / "data" / "output" / "images"
DEFAULT_MASKS = GEOAI_ROOT / "makelable" / "data" / "output" / "labels"
MODEL_ROOT = Path(os.getenv("GEOAI_MODELS_DIR", GEOAI_ROOT / "models")) / "geoai_segmentation"


def build_model(name: str, base_channels=32):
    if name == "unetpp":
        return UNetPlusPlus(out_ch=2, base_channels=base_channels)
    if name == "deeplabv3plus":
        return DeepLabV3Plus(out_ch=2, base_channels=base_channels)
    raise ValueError("model 必须是 unetpp 或 deeplabv3plus")


def train(args, progress=None, cancel_event=None):
    torch.manual_seed(args.seed)
    images, masks = Path(args.images), Path(args.masks)
    all_dataset = SegDataset(images, masks, args.size)
    train_indices, val_indices = split_indices(len(all_dataset), args.val_ratio, args.seed)
    train_dataset = SegDataset(images, masks, args.size, augment=True, indices=train_indices)
    val_dataset = SegDataset(images, masks, args.size, augment=False, indices=val_indices)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output_dir = Path(args.output or MODEL_ROOT / args.model)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_iou, history = -1., []
    for epoch in range(args.epochs):
        if cancel_event is not None and cancel_event.is_set():
            break
        model.train(); total_loss = 0.
        for batch_images, batch_masks in train_loader:
            batch_images, batch_masks = batch_images.to(device), batch_masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(batch_images), batch_masks)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            total_loss += loss.item()
        model.eval(); metrics=[]
        with torch.no_grad():
            for batch_images, batch_masks in val_loader:
                prediction = model(batch_images.to(device)).argmax(1).cpu().numpy()
                metrics.extend(calculate_metrics(p, t.numpy()) for p, t in zip(prediction, batch_masks))
        row = {"epoch": epoch+1, "loss": total_loss/max(len(train_loader),1), "iou": float(np.mean([m['iou'] for m in metrics])), "dice": float(np.mean([m['dice'] for m in metrics]))}
        history.append(row)
        if row["iou"] > best_iou:
            best_iou = row["iou"]
            torch.save({"model": args.model, "base_channels": args.base_channels, "state_dict": model.state_dict(), "metrics": row, "size": args.size}, output_dir / "best.pth")
        if progress:
            progress(int((epoch+1)/args.epochs*100), row)
    result = {"model": args.model, "checkpoint": str(output_dir/"best.pth"), "best_iou": best_iou, "history": history, "train_samples": len(train_dataset), "val_samples": len(val_dataset)}
    (output_dir/"history.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parser():
    result=argparse.ArgumentParser(description="训练 U-Net++ 或 DeepLabV3+ 建筑分割基线")
    result.add_argument("--images", default=str(DEFAULT_IMAGES)); result.add_argument("--masks", default=str(DEFAULT_MASKS))
    result.add_argument("--model", choices=["unetpp","deeplabv3plus"], default="unetpp")
    result.add_argument("--epochs",type=int,default=20); result.add_argument("--batch-size",type=int,default=2)
    result.add_argument("--size",type=int,default=256); result.add_argument("--base-channels",type=int,default=32)
    result.add_argument("--lr",type=float,default=3e-4); result.add_argument("--val-ratio",type=float,default=.25)
    result.add_argument("--seed",type=int,default=42); result.add_argument("--output")
    return result


if __name__ == "__main__":
    print(json.dumps(train(parser().parse_args(), lambda p,m: print(f"{p:3d}% {m}")), ensure_ascii=False, indent=2))
