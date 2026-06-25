"""
GeoAI 图像分类 — 评估脚本
==========================
用法:
    conda activate geoai
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pth
"""
import argparse, sys, json
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from dataset import build_dataloaders, CLASS_NAMES

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(ROOT/"checkpoints"/"best_model.pth"))
    p.add_argument("--data_root",  default=str(ROOT/"data"/"processed"/"EuroSAT"))
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers",type=int, default=4)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_dir", default=str(ROOT/"logs"))
    return p.parse_args()

def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_name  = ckpt.get("model_name",  "resnet50")
    image_size  = ckpt.get("image_size",  224)
    class_names = ckpt.get("class_names", CLASS_NAMES)

    # 动态导入构建函数
    sys.path.insert(0, str(ROOT/"scripts"))
    from train import build_model
    model = build_model(model_name, len(class_names))
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    return model, image_size, class_names

def plot_confusion_matrix(cm, class_names, output_path):
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(class_names)),
           yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           title="Confusion Matrix", ylabel="True Label", xlabel="Predicted Label")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  混淆矩阵已保存: {output_path}")

def main():
    args = parse_args()
    device = torch.device(args.device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"📂 加载 checkpoint: {args.checkpoint}")
    model, image_size, class_names = load_model(args.checkpoint, device)

    print(f"📦 加载测试集...")
    loaders = build_dataloaders(args.data_root, args.batch_size,
                                image_size, args.num_workers)
    test_loader = loaders["test"]

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluating"):
            imgs = imgs.to(device)
            out  = model(imgs)
            preds = out.argmax(1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc  = accuracy_score(all_labels, all_preds)
    f1   = f1_score(all_labels, all_preds, average="macro")
    report = classification_report(all_labels, all_preds,
                                   target_names=class_names, digits=4)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n{'='*60}")
    print(f"  Overall Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Macro F1 Score   : {f1:.4f}")
    print(f"{'='*60}")
    print("\nClassification Report:")
    print(report)

    # 保存结果
    results = {"accuracy": acc, "macro_f1": f1,
               "class_names": class_names,
               "per_class": {}}
    for i, cls in enumerate(class_names):
        mask = all_labels == i
        cls_acc = (all_preds[mask] == i).mean() if mask.sum() > 0 else 0.0
        results["per_class"][cls] = float(cls_acc)

    out_json = Path(args.output_dir) / "eval_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ 评估结果已保存: {out_json}")

    plot_confusion_matrix(cm, class_names,
                          Path(args.output_dir) / "confusion_matrix.png")

if __name__ == "__main__":
    main()
