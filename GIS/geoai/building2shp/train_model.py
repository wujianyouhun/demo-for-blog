#!/usr/bin/env python3
"""
快速生成合成训练数据并训练 DeepLabV3+ 建筑物分割模型。

数据包含模拟的遥感影像和对应标签:
  - 0: background (黑色)
  - 1: building   (灰色/白色矩形)
  - 2: road       (灰色线条)
  - 3: water      (蓝色区域)
  - 4: vegetation (绿色区域)
  - 5: barren     (棕色区域)
"""
import os
import sys
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# 设置 TORCH_HOME
PROJECT_ROOT = Path(__file__).parent
REPO_ROOT = PROJECT_ROOT.parent
SHARED_MODELS_DIR = Path(os.getenv("GEOAI_MODELS_DIR", str(REPO_ROOT / "models"))).expanduser()
if not SHARED_MODELS_DIR.is_absolute():
    SHARED_MODELS_DIR = (REPO_ROOT / SHARED_MODELS_DIR).resolve()
os.environ["TORCH_HOME"] = str(SHARED_MODELS_DIR)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

TILE_SIZE = 256
NUM_CLASSES = 6
NUM_SAMPLES = 200  # 合成样本数
EPOCHS = 20
BATCH_SIZE = 8
LR = 1e-4

IMG_DIR = PROJECT_ROOT / "data" / "train_images"
LBL_DIR = PROJECT_ROOT / "data" / "train_labels"
MODEL_DIR = SHARED_MODELS_DIR / "building2shp" / "trained"


def generate_sample(idx: int):
    """生成一张合成训练样本 (影像 + 标签)。"""
    rng = random.Random(idx)
    img = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
    lbl = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)

    # 背景: 随机选择植被或裸土
    if rng.random() < 0.6:
        # 植被背景
        img[:, :, 0] = rng.randint(40, 90)
        img[:, :, 1] = rng.randint(100, 160)
        img[:, :, 2] = rng.randint(30, 80)
        lbl[:, :] = 4  # vegetation
    else:
        # 裸土背景
        img[:, :, 0] = rng.randint(140, 180)
        img[:, :, 1] = rng.randint(120, 150)
        img[:, :, 2] = rng.randint(80, 110)
        lbl[:, :] = 5  # barren

    # 添加噪声纹理
    noise = rng.randint(-15, 15)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 画道路 (0-2条)
    n_roads = rng.randint(0, 2)
    for _ in range(n_roads):
        road_w = rng.randint(4, 10)
        road_color = rng.randint(130, 170)
        if rng.random() < 0.5:
            # 水平道路
            y = rng.randint(20, TILE_SIZE - 20)
            img[y:y+road_w, :, :] = road_color
            lbl[y:y+road_w, :] = 2
        else:
            # 垂直道路
            x = rng.randint(20, TILE_SIZE - 20)
            img[:, x:x+road_w, :] = road_color
            lbl[:, x:x+road_w] = 2

    # 画建筑物 (2-8 个)
    n_buildings = rng.randint(2, 8)
    for _ in range(n_buildings):
        bw = rng.randint(15, 60)
        bh = rng.randint(15, 60)
        bx = rng.randint(0, TILE_SIZE - bw)
        by = rng.randint(0, TILE_SIZE - bh)

        # 建筑物颜色: 灰白色调
        br = rng.randint(170, 230)
        bg = rng.randint(165, 225)
        bb = rng.randint(160, 220)

        img[by:by+bh, bx:bx+bw, 0] = br
        img[by:by+bh, bx:bx+bw, 1] = bg
        img[by:by+bh, bx:bx+bw, 2] = bb
        lbl[by:by+bh, bx:bx+bw] = 1

        # 有时加阴影边缘
        if rng.random() < 0.3:
            shadow_w = rng.randint(2, 5)
            img[by:by+bh, bx+bw:bx+bw+shadow_w, :] = np.clip(
                img[by:by+bh, bx+bw:bx+bw+shadow_w, :].astype(np.int16) - 60, 0, 255
            ).astype(np.uint8)

    # 有时画水体
    if rng.random() < 0.2:
        wr = rng.randint(20, 50)
        wx = rng.randint(0, TILE_SIZE - wr)
        wy = rng.randint(0, TILE_SIZE - wr)
        img[wy:wy+wr, wx:wx+wr, 0] = rng.randint(20, 50)
        img[wy:wy+wr, wx:wx+wr, 1] = rng.randint(40, 80)
        img[wy:wy+wr, wx:wx+wr, 2] = rng.randint(150, 210)
        lbl[wy:wy+wr, wx:wx+wr] = 3

    return img, lbl


def generate_dataset():
    """生成全部合成训练样本。"""
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    LBL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"生成 {NUM_SAMPLES} 张合成训练样本 ...")
    for i in range(NUM_SAMPLES):
        img, lbl = generate_sample(i)
        Image.fromarray(img).save(IMG_DIR / f"tile_{i:04d}.png")
        Image.fromarray(lbl).save(LBL_DIR / f"tile_{i:04d}.png")
        if (i + 1) % 50 == 0:
            print(f"  已生成 {i+1}/{NUM_SAMPLES}")
    print(f"完成: {IMG_DIR} / {LBL_DIR}")


class BuildingDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, augment=False):
        self.img_files = sorted(Path(img_dir).glob("*.png"))
        self.lbl_files = sorted(Path(lbl_dir).glob("*.png"))
        self.augment = augment

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.img_files[idx])).astype(np.float32) / 255.0
        lbl = np.array(Image.open(self.lbl_files[idx])).astype(np.int64)

        if self.augment:
            # 随机水平翻转
            if random.random() < 0.5:
                img = np.fliplr(img).copy()
                lbl = np.fliplr(lbl).copy()
            # 随机垂直翻转
            if random.random() < 0.5:
                img = np.flipud(img).copy()
                lbl = np.flipud(lbl).copy()
            # 随机旋转 90/180/270
            k = random.randint(0, 3)
            if k > 0:
                img = np.rot90(img, k).copy()
                lbl = np.rot90(lbl, k).copy()

        # HWC -> CHW
        img = torch.from_numpy(img.transpose(2, 0, 1))
        lbl = torch.from_numpy(lbl)
        return img, lbl


def train():
    """训练 DeepLabV3+ 模型。"""
    import segmentation_models_pytorch as smp

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练设备: {device}")

    # 构建模型
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
    ).to(device)

    # 数据集
    train_ds = BuildingDataset(IMG_DIR, LBL_DIR, augment=True)
    val_ds = BuildingDataset(IMG_DIR, LBL_DIR, augment=False)

    # 85/15 拆分
    n_total = len(train_ds)
    n_val = max(1, int(n_total * 0.15))
    n_train = n_total - n_val
    train_subset, val_subset = torch.utils.data.random_split(
        train_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"训练集: {n_train}, 验证集: {n_val}")

    # 损失函数 (类别权重: 给 building 更高权重)
    class_weights = torch.tensor([1.0, 3.0, 1.0, 1.0, 0.5, 0.5]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_f1 = 0
    best_epoch = 0

    for epoch in range(EPOCHS):
        # ── 训练 ──
        model.train()
        train_loss = 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(imgs)
                loss = criterion(outputs, lbls)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # ── 验证 ──
        model.eval()
        val_correct = 0
        val_total = 0
        val_tp = 0
        val_fp = 0
        val_fn = 0

        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)

                val_correct += (preds == lbls).sum().item()
                val_total += lbls.numel()

                # 建筑物 F1
                pred_bld = (preds == 1)
                true_bld = (lbls == 1)
                val_tp += (pred_bld & true_bld).sum().item()
                val_fp += (pred_bld & ~true_bld).sum().item()
                val_fn += (~pred_bld & true_bld).sum().item()

        val_acc = val_correct / val_total
        precision = val_tp / (val_tp + val_fp + 1e-8)
        recall = val_tp / (val_tp + val_fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        print(
            f"Epoch {epoch+1:3d}/{EPOCHS} | "
            f"loss={avg_train_loss:.4f} | "
            f"acc={val_acc:.3f} | "
            f"bld_F1={f1:.3f} (P={precision:.3f} R={recall:.3f})"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch + 1

    # ── 保存模型 ──
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODEL_DIR / "best_model.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "best_f1": best_f1,
        "best_epoch": best_epoch,
        "num_classes": NUM_CLASSES,
    }, save_path)

    print(f"\n训练完成! 最佳 F1={best_f1:.3f} (epoch {best_epoch})")
    print(f"模型已保存: {save_path} ({save_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    generate_dataset()
    train()
