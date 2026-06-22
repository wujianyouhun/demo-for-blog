#!/usr/bin/env python3
"""制作变化检测训练样本。

支持三种模式：
- synthetic: 生成教学用模拟样本。
- weak-label: 用 GeoAI ChangeStar 对双时相影像生成弱标签后切片。
- vector-label: 用真实变化矢量标签栅格化后切片。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

console = Console()


def main():
    parser = argparse.ArgumentParser(description="制作 GeoAI 变化检测训练样本")
    parser.add_argument("--mode", choices=["synthetic", "weak-label", "vector-label"], default="synthetic")
    parser.add_argument("--image-a", help="时相 A GeoTIFF，用于 weak-label/vector-label")
    parser.add_argument("--image-b", help="时相 B GeoTIFF，用于 weak-label/vector-label")
    parser.add_argument("--vector-label", help="真实变化矢量标签，支持 GeoJSON/GPKG/SHP")
    parser.add_argument("--num-samples", type=int, default=100, help="synthetic 模式样本数")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--min-change-pixels", type=int, default=20)
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--geoai-model", default="s1_s1c1_vitb")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--keep-existing", action="store_true", help="追加样本，不清空既有 .tif")
    args = parser.parse_args()

    from config import SAMPLES_DIR
    from cdd.sample_builder import (
        generate_synthetic_samples,
        generate_vector_label_samples,
        generate_weak_label_samples,
    )

    overwrite = not args.keep_existing
    if args.mode == "synthetic":
        console.print("[bold blue]生成模拟双时相变化检测样本[/bold blue]")
        result = generate_synthetic_samples(
            SAMPLES_DIR,
            num_samples=args.num_samples,
            tile_size=args.tile_size,
            overwrite=overwrite,
        )
    elif args.mode == "weak-label":
        _require(args.image_a, "--image-a")
        _require(args.image_b, "--image-b")
        console.print("[bold blue]使用 GeoAI ChangeStar 生成弱标签样本[/bold blue]")
        result = generate_weak_label_samples(
            args.image_a,
            args.image_b,
            SAMPLES_DIR,
            tile_size=args.tile_size,
            stride=args.stride,
            min_change_pixels=args.min_change_pixels,
            max_tiles=args.max_tiles,
            model_name=args.geoai_model,
            threshold=args.threshold,
            overwrite=overwrite,
        )
    else:
        _require(args.image_a, "--image-a")
        _require(args.image_b, "--image-b")
        _require(args.vector_label, "--vector-label")
        console.print("[bold blue]使用真实变化矢量标签制作监督样本[/bold blue]")
        result = generate_vector_label_samples(
            args.image_a,
            args.image_b,
            args.vector_label,
            SAMPLES_DIR,
            tile_size=args.tile_size,
            stride=args.stride,
            min_change_pixels=args.min_change_pixels,
            max_tiles=args.max_tiles,
            overwrite=overwrite,
        )

    counts = result["counts"]
    console.print("[green]样本制作完成[/green]")
    console.print(f"  时相 A: {counts['time_a']} 张")
    console.print(f"  时相 B: {counts['time_b']} 张")
    console.print(f"  标签:   {counts['labels']} 张")
    console.print(f"  目录:   {result['samples_dir']}")


def _require(value, name):
    if not value:
        raise SystemExit(f"{name} 是当前模式的必填参数")


if __name__ == "__main__":
    main()
