"""生成带地理参考的合成三分类数据，仅用于检查训练 / 推理流程。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parent


def scene(rng, height, width):
    rows, cols = np.mgrid[:height, :width]
    label = np.zeros((height, width), dtype="uint8")
    cy, cx = rng.integers(height // 3, 2 * height // 3), rng.integers(width // 3, 2 * width // 3)
    label[((rows - cy) / (height / 3)) ** 2 + ((cols - cx) / (width / 3)) ** 2 < 1] = 1
    label[height // 5:height // 2, width // 5:width // 2] = 2
    colors = np.array([[130, 110, 85], [35, 150, 50], [190, 185, 180]])
    rgb = np.clip(colors[label] + rng.normal(0, 10, (height, width, 3)), 1, 255).astype("uint8")
    # 加入无效角落，以便检查 NoData 是否正确传递。
    rgb[:4, :4] = 0
    label[:4, :4] = 255
    return rgb.transpose(2, 0, 1), label


def write_pair(image_path, mask_path, rng, height, width, x_origin):
    image, mask = scene(rng, height, width)
    profile = dict(driver="GTiff", height=height, width=width, crs="EPSG:32647",
                   transform=from_origin(x_origin, 4200000, 10, 10), compress="deflate")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(image_path, "w", count=3, dtype="uint8", nodata=0, **profile) as dst:
        dst.write(image)
    with rasterio.open(mask_path, "w", count=1, dtype="uint8", nodata=255, **profile) as dst:
        dst.write(mask, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "demo")
    parser.add_argument("--tile-size", type=int, default=256)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"目录已存在，为避免覆盖请使用新的 --output-dir：{args.output_dir}")
    if args.tile_size < 32:
        parser.error("tile-size 至少为 32。")
    rng = np.random.default_rng(42)
    for split, count in (("train", 6), ("val", 2)):
        for i in range(count):
            base = args.output_dir / split
            write_pair(base / "images" / f"tile_{i:03d}.tif", base / "masks" / f"tile_{i:03d}_mask.tif",
                       rng, args.tile_size, args.tile_size, 500000 + i * args.tile_size * 10)
    # 非整块尺寸，覆盖推理窗口的右边缘和下边缘。
    write_pair(args.output_dir / "scene.tif", args.output_dir / "scene_mask.tif",
               rng, args.tile_size + 37, args.tile_size * 2 + 19, 500000)
    print(f"合成数据：{args.output_dir.resolve()}；类别 0=背景、1=植被、2=建筑。")
    print("这是流程测试数据，训练指标不代表真实遥感精度。")


if __name__ == "__main__":
    main()
