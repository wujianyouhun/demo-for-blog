#!/usr/bin/env python3
"""生成模拟双时相样本数据（用于快速体验，无需下载真实数据）"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rich.console import Console

console = Console()

def main():
    console.print("[bold blue]生成模拟双时相变化检测样本[/bold blue]\n")

    from config import SAMPLES_DIR
    dir_a = SAMPLES_DIR / "time_a"
    dir_b = SAMPLES_DIR / "time_b"
    dir_l = SAMPLES_DIR / "labels"
    for d in [dir_a, dir_b, dir_l]:
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(42)
    num_samples = 100
    tile_size = 256

    # 模拟地理范围
    transform = from_bounds(116.3, 39.8, 116.4, 39.9, tile_size, tile_size)
    crs = "EPSG:4326"

    for i in range(num_samples):
        # 时相 A: 随机 RGB 地物
        base = rng.randint(50, 200, size=(3, tile_size, tile_size), dtype=np.uint8)

        # 添加一些结构（模拟建筑物/道路）
        for _ in range(rng.randint(3, 8)):
            cx, cy = rng.randint(20, tile_size-20, 2)
            w, h = rng.randint(10, 40, 2)
            color = rng.randint(100, 255, 3)
            base[:, cy:cy+h, cx:cx+w] = color.reshape(3, 1, 1)

        # 时相 B: 在 A 的基础上引入变化
        img_b = base.copy()

        # 添加变化区域（模拟新建建筑 / 拆除建筑）
        label = np.zeros((tile_size, tile_size), dtype=np.uint8)
        num_changes = rng.randint(1, 5)

        for _ in range(num_changes):
            cx, cy = rng.randint(30, tile_size-30, 2)
            w, h = rng.randint(8, 30, 2)

            # 变化：用新颜色替换
            new_color = rng.randint(50, 255, 3)
            img_b[:, cy:cy+h, cx:cx+w] = new_color.reshape(3, 1, 1)
            label[cy:cy+h, cx:cx+w] = 1

            # 有时也在 A 中添加变化（模拟双向变化）
            if rng.random() > 0.5:
                old_color = rng.randint(50, 255, 3)
                base[:, cy:cy+h, cx:cx+w] = old_color.reshape(3, 1, 1)

        # 添加噪声
        noise_a = rng.normal(0, 5, base.shape).astype(np.int16)
        noise_b = rng.normal(0, 5, img_b.shape).astype(np.int16)
        base = np.clip(base.astype(np.int16) + noise_a, 0, 255).astype(np.uint8)
        img_b = np.clip(img_b.astype(np.int16) + noise_b, 0, 255).astype(np.uint8)

        name = f"sample_{i:04d}"
        profile = {
            "driver": "GTiff", "dtype": "uint8",
            "width": tile_size, "height": tile_size,
            "count": 3, "crs": crs, "transform": transform,
        }

        with rasterio.open(dir_a / f"{name}.tif", "w", **profile) as dst:
            dst.write(base)
        with rasterio.open(dir_b / f"{name}.tif", "w", **profile) as dst:
            dst.write(img_b)

        lbl_profile = {**profile, "count": 1, "dtype": "uint8"}
        with rasterio.open(dir_l / f"{name}.tif", "w", **lbl_profile) as dst:
            dst.write(label, 1)

    console.print(f"[green]生成 {num_samples} 对样本[/green]")
    console.print(f"  时相 A: {dir_a}")
    console.print(f"  时相 B: {dir_b}")
    console.print(f"  标签:   {dir_l}")

if __name__ == "__main__":
    main()
