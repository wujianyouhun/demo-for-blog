"""
GeoAI 图像分类 — 训练脚本
==========================
支持: ResNet50 / ResNet101 / ViT-Base
数据: EuroSAT (10类遥感图像)

用法:
    conda activate geoai
    # ResNet50 (默认)
    python scripts/train.py
    # ViT
    python scripts/train.py --model vit_base_patch16_224 --epochs 20 --lr 5e-5
    # 断点续训
    python scripts/train.py --resume checkpoints/last_model.pth
"""
import argparse, os, sys, time, json
from pathlib import Path

# 在导入 torch/torchvision/timm 前固定共享模型缓存目录。
ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT.parent
SHARED_MODELS_DIR = Path(os.getenv("GEOAI_MODELS_DIR", str(REPO_ROOT / "models"))).expanduser()
if not SHARED_MODELS_DIR.is_absolute():
    SHARED_MODELS_DIR = (REPO_ROOT / SHARED_MODELS_DIR).resolve()
os.environ.setdefault("TORCH_HOME", str(SHARED_MODELS_DIR))
os.environ.setdefault("HF_HOME", str(SHARED_MODELS_DIR / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(SHARED_MODELS_DIR / "huggingface" / "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(SHARED_MODELS_DIR / "huggingface" / "hub"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
import timm
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(ROOT / "scripts"))
from dataset import build_dataloaders, CLASS_NAMES

console = Console()

# ─── 参数 ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="GeoAI 遥感图像分类训练")
    p.add_argument("--model",      default="resnet50",
                   choices=["resnet50","resnet101","vit_base_patch16_224"],
                   help="骨干网络")
    p.add_argument("--data_root",  default=str(ROOT/"data"/"processed"/"EuroSAT"))
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--wd",         type=float, default=1e-4, help="weight decay")
    p.add_argument("--image_size", type=int,   default=224)
    p.add_argument("--num_workers",type=int,   default=4)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume",     default=None, help="checkpoint 路径")
    p.add_argument("--amp",        action="store_true", default=True,
                   help="混合精度训练 (AMP)")
    p.add_argument("--output_dir", default=str(SHARED_MODELS_DIR / "Classification" / "checkpoints"))
    p.add_argument("--log_dir",    default=str(ROOT/"logs"))
    return p.parse_args()

# ─── 模型构建 ───────────────────────────────────────────────────────
def build_model(name: str, num_classes: int) -> nn.Module:
    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "resnet101":
        m = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "vit_base_patch16_224":
        m = timm.create_model("vit_base_patch16_224", pretrained=True,
                               num_classes=num_classes)
    else:
        raise ValueError(f"不支持的模型: {name}")
    return m

# ─── 单 epoch 训练/验证 ─────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device,
              scaler=None, training=True):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labels in tqdm(loader, leave=False,
                                  desc="train" if training else "val "):
            imgs, labels = imgs.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            with autocast(enabled=(scaler is not None)):
                out  = model(imgs)
                loss = criterion(out, labels)
            if training:
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct    += (out.argmax(1) == labels).sum().item()
            total      += imgs.size(0)

    return total_loss / total, correct / total

# ─── 主函数 ────────────────────────────────────────────────────────
def main():
    args = parse_args()
    device = torch.device(args.device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    console.rule("[bold cyan]GeoAI 遥感图像分类训练")
    console.print(f"模型: [green]{args.model}[/]  "
                  f"设备: [yellow]{device}[/]  "
                  f"epochs: {args.epochs}  lr: {args.lr}")

    # 数据
    console.print("\n[bold]📂 加载数据集...")
    loaders = build_dataloaders(args.data_root, args.batch_size,
                                args.image_size, args.num_workers)

    # 模型
    console.print(f"\n[bold]🧠 构建模型: {args.model}")
    model = build_model(args.model, len(CLASS_NAMES)).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                             weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = GradScaler() if (args.amp and device.type == "cuda") else None

    # 断点续训
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        console.print(f"✓ 从 epoch {start_epoch} 继续训练, best_val_acc={best_val_acc:.4f}")

    writer = SummaryWriter(log_dir=args.log_dir)
    history = []

    # 训练循环
    console.print("\n[bold]🚀 开始训练...")
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss, train_acc = run_epoch(
            model, loaders["train"], criterion, optimizer, device, scaler, True)
        val_loss,   val_acc   = run_epoch(
            model, loaders["val"],   criterion, None,      device, None,   False)
        scheduler.step()
        elapsed = time.time() - t0

        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Acc",  {"train": train_acc,  "val": val_acc},  epoch)

        flag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            flag = " 🏆 [bold green]new best![/]"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "model_name": args.model,
                "class_names": CLASS_NAMES,
                "image_size": args.image_size,
            }, Path(args.output_dir) / "best_model.pth")

        torch.save({
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "model_name": args.model,
            "class_names": CLASS_NAMES,
            "image_size": args.image_size,
        }, Path(args.output_dir) / "last_model.pth")

        row = dict(epoch=epoch, train_loss=round(train_loss,4),
                   train_acc=round(train_acc,4), val_loss=round(val_loss,4),
                   val_acc=round(val_acc,4), elapsed=round(elapsed,1))
        history.append(row)
        console.print(
            f"Epoch [{epoch+1:3d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.2e}  {elapsed:.1f}s" + flag)

    # 保存训练历史
    with open(Path(args.output_dir) / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    writer.close()
    console.rule("[bold green]训练完成")
    console.print(f"最佳验证准确率: [bold green]{best_val_acc:.4f}")
    console.print(f"模型保存路径  : {args.output_dir}/best_model.pth")

if __name__ == "__main__":
    main()
