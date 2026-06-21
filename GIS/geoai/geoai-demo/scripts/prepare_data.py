#!/usr/bin/env python3
"""
prepare_data.py - 下载示例 Sentinel-2 数据并生成模拟标签

Usage:
    python scripts/prepare_data.py --region beijing --date 2023-06-01
    python scripts/prepare_data.py --region shanghai --date 2023-07-01 --output data/shanghai
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Region presets with bounding boxes [min_lon, min_lat, max_lon, max_lat]
REGION_PRESETS = {
    "beijing": {"bbox": [116.2, 39.7, 116.6, 40.1], "name": "北京"},
    "shanghai": {"bbox": [121.3, 31.0, 121.7, 31.4], "name": "上海"},
    "shenzhen": {"bbox": [113.8, 22.4, 114.2, 22.7], "name": "深圳"},
    "chengdu": {"bbox": [103.9, 30.5, 104.3, 30.8], "name": "成都"},
    "wuhan": {"bbox": [114.1, 30.4, 114.5, 30.7], "name": "武汉"},
}

# 6-class land cover schema
CLASSES = {
    0: "background",
    1: "building",
    2: "road",
    3: "water",
    4: "vegetation",
    5: "barren",
}


def print_banner(console):
    """Print startup banner."""
    console.print("[bold blue]GeoAI Demo - 数据准备工具[/bold blue]")
    console.print("=" * 50)


def download_sentinel2(region_info, date_str, output_dir, console):
    """
    Simulate downloading Sentinel-2 data.
    In production, this would use planetary_computer / pystac_client.
    """
    console.print(f"\n[bold]区域:[/bold] {region_info['name']}")
    console.print(f"[bold]范围:[/bold] {region_info['bbox']}")
    console.print(f"[bold]日期:[/bold] {date_str}")
    console.print(f"[bold]输出:[/bold] {output_dir}")
    console.print()

    bbox = region_info["bbox"]
    width, height = 512, 512

    console.print("[yellow]模拟下载 Sentinel-2 影像...[/yellow]")

    # Generate mock multispectral image (13 bands for Sentinel-2)
    bands = ["B01", "B02", "B03", "B04", "B05", "B06",
             "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]

    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    # Create a synthetic image
    np.random.seed(42)
    image_data = np.random.randint(0, 4000, (height, width, len(bands)), dtype=np.uint16)

    # Add some spatial patterns to make it look more realistic
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    # Simulate water body (lower-left corner)
    water_mask = (x_coords < width * 0.3) & (y_coords > height * 0.7)
    image_data[water_mask, 1] = 200   # Low blue in water
    image_data[water_mask, 7] = 100   # Low NIR in water

    # Simulate vegetation (upper-right area)
    veg_mask = (x_coords > width * 0.5) & (y_coords < height * 0.5)
    image_data[veg_mask, 7] = 3500    # High NIR for vegetation
    image_data[veg_mask, 3] = 300     # Low red for vegetation

    image_path = image_dir / f"sentinel2_{region_info['name']}_{date_str}.npy"
    np.save(image_path, image_data)

    console.print(f"  [green]✓[/green] 已保存影像: {image_path.name} ({width}x{height}, {len(bands)} bands)")

    return image_path, width, height


def generate_labels(width, height, output_dir, console):
    """Generate mock classification labels."""
    console.print("\n[yellow]生成模拟标签...[/yellow]")

    label_dir = output_dir / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(123)
    labels = np.zeros((height, width), dtype=np.uint8)

    # Create spatial regions for each class
    y_coords, x_coords = np.mgrid[0:height, 0:width]

    # Water: lower-left
    water_mask = (x_coords < width * 0.25) & (y_coords > height * 0.75)
    labels[water_mask] = 3

    # Vegetation: upper-right quadrant
    veg_mask = (x_coords > width * 0.55) & (y_coords < height * 0.45)
    labels[veg_mask] = 4

    # Building: center block
    build_mask = (
        (x_coords > width * 0.3) & (x_coords < width * 0.5) &
        (y_coords > height * 0.3) & (y_coords < height * 0.5)
    )
    labels[build_mask] = 1

    # Road: horizontal stripe
    road_mask = (
        (y_coords > height * 0.48) & (y_coords < height * 0.52)
    )
    labels[road_mask] = 2

    # Barren: lower-right
    barren_mask = (x_coords > width * 0.7) & (y_coords > height * 0.6)
    labels[barren_mask] = 5

    # Add some noise
    noise_mask = np.random.random((height, width)) < 0.02
    labels[noise_mask] = np.random.randint(0, 6, noise_mask.sum())

    label_path = label_dir / "labels.npy"
    np.save(label_path, labels)

    console.print(f"  [green]✓[/green] 已保存标签: {label_path.name}")

    # Class distribution
    console.print("\n[bold]类别分布:[/bold]")
    total_pixels = height * width
    for class_id, class_name in CLASSES.items():
        count = (labels == class_id).sum()
        pct = count / total_pixels * 100
        bar = "█" * int(pct / 2)
        console.print(f"  {class_name:12s} ({class_id}): {count:7d} px ({pct:5.1f}%) {bar}")

    return label_path


def generate_metadata(region_info, date_str, image_path, label_path, output_dir):
    """Save metadata JSON."""
    metadata = {
        "region": region_info["name"],
        "bbox": region_info["bbox"],
        "date": date_str,
        "source": "Sentinel-2 (simulated)",
        "classes": CLASSES,
        "image_file": str(image_path.name),
        "label_file": str(label_path.name),
        "created_at": datetime.now().isoformat(),
    }

    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return meta_path


def main():
    parser = argparse.ArgumentParser(description="GeoAI Demo - 数据准备工具")
    parser.add_argument(
        "--region",
        type=str,
        default="beijing",
        choices=list(REGION_PRESETS.keys()),
        help="选择预设区域 (default: beijing)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default="2023-06-01",
        help="影像日期 (default: 2023-06-01)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出目录 (default: data/<region>)",
    )
    parser.add_argument(
        "--no-rich",
        action="store_true",
        help="禁用 rich 输出",
    )

    args = parser.parse_args()

    # Setup console
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
    except ImportError:
        # Fallback if rich is not installed
        class SimpleConsole:
            def print(self, *a, **kw):
                import re
                text = " ".join(str(x) for x in a)
                text = re.sub(r"\[.*?\]", "", text)
                print(text)
        console = SimpleConsole()
        Table = None

    print_banner(console)

    region_info = REGION_PRESETS[args.region]
    output_dir = Path(args.output) if args.output else Path("data") / args.region

    console.print(f"\n[bold]输出目录:[/bold] {output_dir.resolve()}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download image
    image_path, width, height = download_sentinel2(region_info, args.date, output_dir, console)

    # Step 2: Generate labels
    label_path = generate_labels(width, height, output_dir, console)

    # Step 3: Save metadata
    meta_path = generate_metadata(region_info, args.date, image_path, label_path, output_dir)
    console.print(f"\n  [green]✓[/green] 已保存元数据: {meta_path.name}")

    # Summary table
    if Table:
        console.print("\n")
        table = Table(title="数据准备完成 - 摘要")
        table.add_column("项目", style="cyan")
        table.add_column("值", style="green")

        table.add_row("区域", region_info["name"])
        table.add_row("日期", args.date)
        table.add_row("影像尺寸", f"{width} x {height}")
        table.add_row("波段数", "13")
        table.add_row("类别数", "6")
        table.add_row("输出目录", str(output_dir.resolve()))
        table.add_row("影像文件", str(image_path.name))
        table.add_row("标签文件", str(label_path.name))

        console.print(table)
    else:
        console.print("\n[bold]数据准备完成![/bold]")

    console.print(f"\n[bold green]✓ 数据准备完成![/bold green]")
    console.print(f"  影像: {image_path}")
    console.print(f"  标签: {label_path}")
    console.print(f"  元数据: {meta_path}")


if __name__ == "__main__":
    main()
