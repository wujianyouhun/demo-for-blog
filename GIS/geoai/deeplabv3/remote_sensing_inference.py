"""按窗口分割遥感 GeoTIFF，保留 CRS、像元大小、地理变换及 NoData。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
import torch

from common import DEFAULT_CHECKPOINT, IGNORE, ROOT, choose_device, load_checkpoint, palette, to_tensor


def predict(image_path, output_path, checkpoint_path, device_name="auto", tile_size=512, halo=32):
    if tile_size < 32 or halo < 0 or halo * 2 >= tile_size:
        raise ValueError("tile-size 至少为 32，halo 必须满足 0 <= 2*halo < tile-size。")
    if image_path.resolve() == output_path.resolve():
        raise ValueError("输出路径不能与输入影像相同。")
    device = choose_device(device_name)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    bands, scale, names = checkpoint["bands"], checkpoint["scale"], checkpoint["class_names"]
    counts = np.zeros(len(names), dtype="int64")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 每次读取 tile_size×tile_size（含 halo）；只写入中心区域，边缘像元获得额外上下文。
    stride = tile_size - 2 * halo
    with rasterio.open(image_path) as src:
        if max(bands) > src.count:
            raise ValueError(f"影像只有 {src.count} 个波段，模型需要 {bands}。")
        profile = dict(driver="GTiff", height=src.height, width=src.width, count=1,
                       dtype="uint8", crs=src.crs, transform=src.transform,
                       nodata=IGNORE, compress="deflate", BIGTIFF="IF_SAFER")
        with rasterio.open(output_path, "w", **profile) as dst, torch.inference_mode():
            for y in range(0, src.height, stride):
                for x in range(0, src.width, stride):
                    height, width = min(stride, src.height - y), min(stride, src.width - x)
                    window = Window(x - halo, y - halo, tile_size, tile_size)
                    image = src.read(bands, window=window, boundless=True, fill_value=0)
                    valid = (src.read_masks(bands, window=window, boundless=True) > 0).all(axis=0)
                    valid &= np.isfinite(image).all(axis=0)
                    center = (slice(halo, halo + height), slice(halo, halo + width))
                    if valid[center].any():
                        tensor = to_tensor(image, scale).unsqueeze(0).to(device)
                        logits = model(tensor)["out"]
                        mask = logits.argmax(1)[0].cpu().numpy()[center].astype("uint8")
                        mask[~valid[center]] = IGNORE
                    else:
                        mask = np.full((height, width), IGNORE, dtype="uint8")
                    dst.write(mask, 1, window=Window(x, y, width, height))
                    ids, sizes = np.unique(mask[mask != IGNORE], return_counts=True)
                    counts[ids] += sizes
                print(f"已处理 {min(y + stride, src.height)}/{src.height} 行", flush=True)
            colors = palette(256)
            dst.write_colormap(1, {i: tuple(colors[3*i:3*i+3]) + (255,) for i in range(len(names))}
                               | {IGNORE: (0, 0, 0, 0)})
            dst.update_tags(class_names=json.dumps(names, ensure_ascii=False),
                            class_labels=json.dumps(checkpoint.get("class_labels") or names, ensure_ascii=False),
                            checkpoint=str(checkpoint_path.resolve()), bands=str(bands), scale=str(scale),
                            checkpoint_epoch=str(checkpoint.get("epoch", "")),
                            checkpoint_val_miou=str(checkpoint.get("val_miou", "")))
    total = int(counts.sum())
    summary = [dict(class_id=i, name=name, pixels=int(counts[i]),
                    fraction=float(counts[i] / total) if total else 0.0) for i, name in enumerate(names)]
    output_path.with_suffix(".classes.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"输出：{output_path.resolve()}；有效像元：{total}")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=ROOT / "out" / "remote_sensing" / "classification.tif")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--halo", type=int, default=32, help="块边缘额外上下文像元数")
    args = parser.parse_args()
    predict(args.image, args.output, args.checkpoint, args.device, args.tile_size, args.halo)


if __name__ == "__main__":
    main()
