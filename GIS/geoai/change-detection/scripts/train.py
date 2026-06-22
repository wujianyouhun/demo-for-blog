#!/usr/bin/env python3
"""训练变化检测模型"""
import sys, argparse, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.logging import RichHandler
console = Console()
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler(console=console)])

def main():
    parser = argparse.ArgumentParser(description="训练变化检测模型")
    parser.add_argument("--model", default="siamese_unet", choices=["siamese_unet", "bit"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dir-a", default=None)
    parser.add_argument("--dir-b", default=None)
    parser.add_argument("--label-dir", default=None)
    args = parser.parse_args()

    from config import SAMPLES_DIR, MODELS_DIR
    from cdd.models import build_model
    from cdd.dataset import create_dataloaders
    from cdd.sample_builder import validate_samples
    from cdd.trainer import Trainer

    dir_a = Path(args.dir_a) if args.dir_a else SAMPLES_DIR / "time_a"
    dir_b = Path(args.dir_b) if args.dir_b else SAMPLES_DIR / "time_b"
    label_dir = Path(args.label_dir) if args.label_dir else SAMPLES_DIR / "labels"

    check = validate_samples(dir_a.parent)
    if not check["ok"]:
        counts = check["counts"]
        console.print("[red]训练样本不足或数量不匹配[/red]")
        console.print(f"  时相 A: {counts['time_a']} 张 ({dir_a})")
        console.print(f"  时相 B: {counts['time_b']} 张 ({dir_b})")
        console.print(f"  标签:   {counts['labels']} 张 ({label_dir})")
        console.print("先运行: python scripts/generate_sample.py --mode synthetic")
        return

    console.print(f"[blue]构建模型: {args.model}[/blue]")
    model = build_model(args.model)

    console.print("[blue]创建数据加载器[/blue]")
    train_loader, val_loader = create_dataloaders(
        dir_a=dir_a, dir_b=dir_b, label_dir=label_dir,
        batch_size=args.batch_size,
    )

    console.print(f"[blue]开始训练 ({args.epochs} epochs)[/blue]")
    trainer = Trainer(model=model, lr=args.lr)
    history = trainer.fit(train_loader, val_loader, epochs=args.epochs, save_dir=MODELS_DIR)

    console.print(f"\n[green]训练完成! 最佳 F1: {trainer.best_val_f1:.4f}[/green]")
    console.print(f"模型保存至: {MODELS_DIR}")

if __name__ == "__main__":
    main()
