#!/usr/bin/env python3
"""
train.py - 训练 DeepLabV3+ 语义分割模型

Usage:
    python scripts/train.py --model deeplabv3p_resnet50 --epochs 10
    python scripts/train.py --model deeplabv3p_resnet101 --epochs 50 --batch-size 8
    python scripts/train.py --data data/beijing --model deeplabv3p_xception --lr 0.0005
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Model configurations
MODEL_CONFIGS = {
    "deeplabv3p_resnet50": {
        "backbone": "resnet50",
        "encoder_depth": 5,
        "decoder_channels": 256,
        "params": "~40M",
    },
    "deeplabv3p_resnet101": {
        "backbone": "resnet101",
        "encoder_depth": 5,
        "decoder_channels": 256,
        "params": "~60M",
    },
    "deeplabv3p_xception": {
        "backbone": "xception",
        "encoder_depth": 5,
        "decoder_channels": 256,
        "params": "~45M",
    },
    "deeplabv3p_mobilenetv2": {
        "backbone": "mobilenet_v2",
        "encoder_depth": 5,
        "decoder_channels": 256,
        "params": "~8M",
    },
}

# 6-class land cover
NUM_CLASSES = 6
CLASS_NAMES = ["background", "building", "road", "water", "vegetation", "barren"]


def print_banner(console):
    console.print("[bold blue]GeoAI Demo - 模型训练工具[/bold blue]")
    console.print("=" * 50)


def build_model(model_name, num_classes):
    """
    Build a DeepLabV3+ model.
    In production, this uses segmentation_models_pytorch.
    Here we simulate the model structure.
    """
    config = MODEL_CONFIGS[model_name]

    class MockModel:
        def __init__(self, config, num_classes):
            self.config = config
            self.num_classes = num_classes
            self.training = True

        def __call__(self, x):
            batch_size = x.shape[0]
            h, w = x.shape[2], x.shape[3]
            return np.random.rand(batch_size, self.num_classes, h, w).astype(np.float32)

        def train(self, mode=True):
            self.training = mode
            return self

        def eval(self):
            return self.train(False)

        def state_dict(self):
            return {"model": model_name, "num_classes": num_classes}

    return MockModel(config, num_classes)


def load_data(data_dir, console):
    """Load training data or create mock data."""
    data_path = Path(data_dir)

    image_path = None
    label_path = None

    # Search for data files
    images_dir = data_path / "images"
    labels_dir = data_path / "labels"

    if images_dir.exists() and labels_dir.exists():
        img_files = list(images_dir.glob("*.npy"))
        lbl_files = list(labels_dir.glob("*.npy"))
        if img_files and lbl_files:
            image_path = img_files[0]
            label_path = lbl_files[0]

    if image_path and label_path:
        console.print(f"[green]✓[/green] 找到数据: {image_path.name}, {label_path.name}")
        images = np.load(image_path)
        labels = np.load(label_path)
        return images, labels
    else:
        console.print("[yellow]未找到数据文件, 生成模拟数据...[/yellow]")
        np.random.seed(42)
        # Simulate: 10 samples of 256x256 with 3 channels
        images = np.random.randint(0, 4000, (10, 256, 256, 3), dtype=np.uint16)
        labels = np.random.randint(0, NUM_CLASSES, (10, 256, 256), dtype=np.uint8)
        return images, labels


def create_augmentation_pipeline():
    """Create albumentations augmentation pipeline."""
    # In production, this uses albumentations:
    # import albumentations as A
    # transform = A.Compose([
    #     A.RandomCrop(256, 256),
    #     A.HorizontalFlip(p=0.5),
    #     A.VerticalFlip(p=0.3),
    #     A.RandomRotate90(p=0.5),
    #     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    return {
        "transforms": [
            "RandomCrop(256, 256)",
            "HorizontalFlip(p=0.5)",
            "VerticalFlip(p=0.3)",
            "RandomRotate90(p=0.5)",
            "Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])",
        ]
    }


def compute_miou(pred, target, num_classes):
    """Compute mean Intersection over Union."""
    ious = []
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        intersection = (pred_cls & target_cls).sum()
        union = (pred_cls | target_cls).sum()
        if union > 0:
            ious.append(intersection / union)
    return np.mean(ious) if ious else 0.0


def train(model, images, labels, epochs, batch_size, lr, console):
    """Simulate training loop."""
    console.print(f"\n[bold]开始训练...[/bold]")
    console.print(f"  样本数: {len(images)}")
    console.print(f"  轮数: {epochs}")
    console.print(f"  批量: {batch_size}")
    console.print(f"  学习率: {lr}")
    console.print()

    n_samples = len(images)
    loss_history = []
    miou_history = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        n_batches = 0

        # Simulate batch iteration
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_loss = max(0.05, 1.5 - epoch * 0.1 + np.random.normal(0, 0.05))
            epoch_loss += batch_loss
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        miou = min(0.95, 0.1 + epoch * 0.08 + np.random.normal(0, 0.02))

        loss_history.append(avg_loss)
        miou_history.append(miou)

        # Progress bar simulation
        pct = epoch / epochs
        bar_len = 30
        filled = int(bar_len * pct)
        bar = "█" * filled + "░" * (bar_len - filled)

        console.print(
            f"  Epoch [{epoch:3d}/{epochs}] |{bar}| "
            f"loss: {avg_loss:.4f} | mIoU: {miou:.4f}"
        )

        time.sleep(0.05)  # Simulate computation

    return loss_history, miou_history


def save_checkpoint(model, loss_history, miou_history, output_dir, console):
    """Save model checkpoint."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state": model.state_dict(),
        "loss_history": loss_history,
        "miou_history": miou_history,
        "final_loss": loss_history[-1] if loss_history else 0,
        "final_miou": miou_history[-1] if miou_history else 0,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
    }

    ckpt_path = output_dir / "model_checkpoint.json"
    with open(ckpt_path, "w") as f:
        json.dump(checkpoint, f, indent=2, default=str)

    console.print(f"\n  [green]✓[/green] 模型已保存: {ckpt_path}")
    return ckpt_path


def main():
    parser = argparse.ArgumentParser(description="GeoAI Demo - 模型训练工具")
    parser.add_argument(
        "--model",
        type=str,
        default="deeplabv3p_resnet50",
        choices=list(MODEL_CONFIGS.keys()),
        help="模型类型 (default: deeplabv3p_resnet50)",
    )
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数 (default: 10)")
    parser.add_argument("--batch-size", type=int, default=4, help="批量大小 (default: 4)")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率 (default: 0.001)")
    parser.add_argument("--data", type=str, default="data/beijing", help="数据目录")
    parser.add_argument("--output", type=str, default="output/models", help="输出目录")
    parser.add_argument("--no-rich", action="store_true", help="禁用 rich 输出")

    args = parser.parse_args()

    # Setup console
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
    except ImportError:
        class SimpleConsole:
            def print(self, *a, **kw):
                import re
                text = " ".join(str(x) for x in a)
                text = re.sub(r"\[.*?\]", "", text)
                print(text)
        console = SimpleConsole()
        Table = None

    print_banner(console)

    # Show model config
    config = MODEL_CONFIGS[args.model]
    console.print(f"\n[bold]模型:[/bold] {args.model}")
    console.print(f"  Backbone: {config['backbone']}")
    console.print(f"  参数量: {config['params']}")
    console.print(f"  类别数: {NUM_CLASSES}")

    # Data augmentation
    aug = create_augmentation_pipeline()
    console.print(f"\n[bold]数据增强:[/bold]")
    for t in aug["transforms"]:
        console.print(f"  - {t}")

    # Load data
    console.print(f"\n[bold]加载数据...[/bold]")
    images, labels = load_data(args.data, console)

    # Build model
    console.print(f"\n[bold]构建模型...[/bold]")
    model = build_model(args.model, NUM_CLASSES)
    console.print(f"  [green]✓[/green] 模型构建完成")

    # Train
    loss_history, miou_history = train(
        model, images, labels, args.epochs, args.batch_size, args.lr, console
    )

    # Save checkpoint
    ckpt_path = save_checkpoint(model, loss_history, miou_history, args.output, console)

    # Summary table
    if Table:
        console.print("\n")
        table = Table(title="训练完成 - 摘要")
        table.add_column("项目", style="cyan")
        table.add_column("值", style="green")

        table.add_row("模型", args.model)
        table.add_row("Backbone", config["backbone"])
        table.add_row("训练轮数", str(args.epochs))
        table.add_row("批量大小", str(args.batch_size))
        table.add_row("学习率", str(args.lr))
        table.add_row("初始损失", f"{loss_history[0]:.4f}")
        table.add_row("最终损失", f"{loss_history[-1]:.4f}")
        table.add_row("初始 mIoU", f"{miou_history[0]:.4f}")
        table.add_row("最终 mIoU", f"{miou_history[-1]:.4f}")
        table.add_row("检查点", str(ckpt_path))

        console.print(table)
    else:
        console.print(f"\n[bold]训练完成! 最终损失: {loss_history[-1]:.4f}, 最终 mIoU: {miou_history[-1]:.4f}[/bold]")

    console.print(f"\n[bold green]✓ 训练完成![/bold green]")


if __name__ == "__main__":
    main()
