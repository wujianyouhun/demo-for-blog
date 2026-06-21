#!/usr/bin/env python3
"""
predict.py - 运行推理并可选正则化

Usage:
    python scripts/predict.py --input image.tif --model model.pth
    python scripts/predict.py --input image.tif --model model.pth --regularize
    python scripts/predict.py --input data/beijing/images/sentinel2.npy --model output/models/model_checkpoint.json --regularize --output output/results
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# 6-class land cover
NUM_CLASSES = 6
CLASS_NAMES = ["background", "building", "road", "water", "vegetation", "barren"]
CLASS_NAMES_ZH = ["背景", "建筑", "道路", "水体", "植被", "裸地"]
CLASS_COLORS = {
    0: "#808080",
    1: "#e6194b",
    2: "#ffe119",
    3: "#3cb44b",
    4: "#42d4f4",
    5: "#f58231",
}


def print_banner(console):
    console.print("[bold blue]GeoAI Demo - 推理与正则化工具[/bold blue]")
    console.print("=" * 50)


def load_model(model_path, console):
    """Load trained model."""
    model_path = Path(model_path)
    if model_path.exists():
        with open(model_path) as f:
            checkpoint = json.load(f)
        console.print(f"[green]✓[/green] 已加载模型: {model_path.name}")
        console.print(f"  最终损失: {checkpoint.get('final_loss', 'N/A')}")
        console.print(f"  最终 mIoU: {checkpoint.get('final_miou', 'N/A')}")
        return checkpoint
    else:
        console.print(f"[yellow]模型文件未找到: {model_path}, 使用模拟模型[/yellow]")
        return {"num_classes": NUM_CLASSES, "class_names": CLASS_NAMES}


def load_image(image_path, console):
    """Load input image."""
    image_path = Path(image_path)
    if image_path.exists() and image_path.suffix == ".npy":
        data = np.load(image_path)
        console.print(f"[green]✓[/green] 已加载影像: {image_path.name}")
        console.print(f"  形状: {data.shape}")
        return data
    else:
        console.print(f"[yellow]影像文件未找到: {image_path}, 生成模拟影像[/yellow]")
        np.random.seed(42)
        return np.random.randint(0, 4000, (512, 512, 3), dtype=np.uint16)


def run_inference(image, model_info, tile_size=256, overlap=32, threshold=0.5, console=None):
    """
    Run tiled inference on the image.
    In production, this uses PyTorch + the actual model.
    Here we simulate with random predictions + spatial patterns.
    """
    h, w = image.shape[:2]
    console.print(f"\n[bold]开始推理...[/bold]")
    console.print(f"  影像尺寸: {w}x{h}")
    console.print(f"  切片尺寸: {tile_size}")
    console.print(f"  重叠: {overlap}")
    console.print(f"  阈值: {threshold}")

    start_time = time.time()

    # Simulate prediction with spatial patterns
    np.random.seed(42)
    prediction = np.zeros((h, w), dtype=np.uint8)

    y_coords, x_coords = np.mgrid[0:h, 0:w]

    # Create realistic-looking segmentation
    # Water: bottom-left region
    water_mask = (x_coords < w * 0.25) & (y_coords > h * 0.7)
    prediction[water_mask] = 3

    # Vegetation: upper-right
    veg_mask = (x_coords > w * 0.5) & (y_coords < h * 0.4)
    prediction[veg_mask] = 4

    # Building: center blocks
    for i in range(5):
        cx = np.random.randint(w * 0.2, w * 0.7)
        cy = np.random.randint(h * 0.2, h * 0.7)
        bw = np.random.randint(20, 60)
        bh = np.random.randint(20, 60)
        build_mask = (
            (x_coords > cx) & (x_coords < cx + bw) &
            (y_coords > cy) & (y_coords < cy + bh)
        )
        prediction[build_mask] = 1

    # Road: horizontal and vertical lines
    road_h = (y_coords > h * 0.48) & (y_coords < h * 0.52)
    road_v = (x_coords > w * 0.48) & (x_coords < w * 0.52)
    prediction[road_h] = 2
    prediction[road_v] = 2

    # Barren: scattered patches
    barren_mask = (x_coords > w * 0.7) & (y_coords > h * 0.6)
    prediction[barren_mask] = 5

    # Add noise
    noise = np.random.random((h, w)) < 0.03
    prediction[noise] = np.random.randint(0, NUM_CLASSES, noise.sum())

    elapsed = time.time() - start_time
    console.print(f"  [green]✓[/green] 推理完成, 耗时: {elapsed:.2f}s")

    return prediction, elapsed


def vectorize_mask(prediction, console):
    """Convert raster prediction to vector features (GeoJSON)."""
    console.print(f"\n[bold]矢量化...[/bold]")

    h, w = prediction.shape
    features = []

    # Simple vectorization: create rectangular polygons per class region
    # In production, use rasterio.features.shapes() or shapely
    for class_id in range(NUM_CLASSES):
        mask = prediction == class_id
        pixel_count = mask.sum()
        if pixel_count == 0:
            continue

        # Find bounding regions using simple connected component approximation
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue

        # Create a simplified polygon from the extent
        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()

        # Convert pixel coords to mock geographic coords
        # (in production, use the actual geotransform)
        base_lon, base_lat = 116.3, 39.9
        scale = 0.00001

        coords = [
            [base_lon + min_x * scale, base_lat + min_y * scale],
            [base_lon + max_x * scale, base_lat + min_y * scale],
            [base_lon + max_x * scale, base_lat + max_y * scale],
            [base_lon + min_x * scale, base_lat + max_y * scale],
            [base_lon + min_x * scale, base_lat + min_y * scale],
        ]

        area = pixel_count * (scale * 111000) ** 2  # rough m² conversion

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords],
            },
            "properties": {
                "class_id": int(class_id),
                "class_name": CLASS_NAMES[class_id],
                "class_name_zh": CLASS_NAMES_ZH[class_id],
                "pixel_count": int(pixel_count),
                "area": float(area),
            },
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    console.print(f"  [green]✓[/green] 生成 {len(features)} 个要素")
    return geojson


def regularize(geojson, tolerance=1.0, smooth_iter=2, min_area=50, orthogonalize=True, console=None):
    """
    Apply regularization to vectorized features.
    In production, uses shapely for geometry operations.
    """
    console.print(f"\n[bold]要素正则化...[/bold]")
    console.print(f"  简化容差: {tolerance}")
    console.print(f"  平滑迭代: {smooth_iter}")
    console.print(f"  最小面积: {min_area} m²")
    console.print(f"  正交化: {orthogonalize}")

    stats_before = {
        "count": len(geojson["features"]),
        "total_area": sum(f["properties"]["area"] for f in geojson["features"]),
        "vertices": sum(
            len(f["geometry"]["coordinates"][0]) for f in geojson["features"]
        ),
    }

    # Filter by min area
    filtered_features = [
        f for f in geojson["features"]
        if f["properties"]["area"] >= min_area
    ]

    # Simplify coordinates (mock simplification)
    for feature in filtered_features:
        coords = feature["geometry"]["coordinates"][0]
        if len(coords) > 5:
            # Keep only key vertices (simplified)
            step = max(1, len(coords) // 8)
            simplified = coords[::step]
            if simplified[0] != simplified[-1]:
                simplified.append(simplified[0])
            feature["geometry"]["coordinates"][0] = simplified

    result_geojson = {
        "type": "FeatureCollection",
        "features": filtered_features,
    }

    stats_after = {
        "count": len(filtered_features),
        "total_area": sum(f["properties"]["area"] for f in filtered_features),
        "vertices": sum(
            len(f["geometry"]["coordinates"][0]) for f in filtered_features
        ),
    }

    console.print(f"  [green]✓[/green] 正则化完成")
    console.print(f"  要素: {stats_before['count']} -> {stats_after['count']}")
    console.print(f"  顶点: {stats_before['vertices']} -> {stats_after['vertices']}")

    return result_geojson, stats_before, stats_after


def save_results(prediction, geojson, output_dir, console):
    """Save output mask and vector files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save prediction mask
    mask_path = output_dir / "prediction_mask.npy"
    np.save(mask_path, prediction)
    console.print(f"  [green]✓[/green] 预测结果: {mask_path}")

    # Save GeoJSON
    geojson_path = output_dir / "prediction_vectors.geojson"
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)
    console.print(f"  [green]✓[/green] 矢量结果: {geojson_path}")

    return mask_path, geojson_path


def main():
    parser = argparse.ArgumentParser(description="GeoAI Demo - 推理与正则化工具")
    parser.add_argument("--input", type=str, required=True, help="输入影像路径")
    parser.add_argument("--model", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--tile-size", type=int, default=256, help="切片尺寸 (default: 256)")
    parser.add_argument("--overlap", type=int, default=32, help="重叠区域 (default: 32)")
    parser.add_argument("--threshold", type=float, default=0.5, help="置信阈值 (default: 0.5)")
    parser.add_argument("--regularize", action="store_true", help="启用正则化")
    parser.add_argument("--tolerance", type=float, default=1.0, help="简化容差 (default: 1.0)")
    parser.add_argument("--min-area", type=float, default=50, help="最小面积 m² (default: 50)")
    parser.add_argument("--output", type=str, default="output/results", help="输出目录")
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

    # Load model
    model_info = load_model(args.model, console)

    # Load image
    image = load_image(args.input, console)

    # Run inference
    prediction, elapsed = run_inference(
        image, model_info,
        tile_size=args.tile_size,
        overlap=args.overlap,
        threshold=args.threshold,
        console=console,
    )

    # Vectorize
    geojson = vectorize_mask(prediction, console)

    # Regularize if requested
    stats_before = stats_after = None
    if args.regularize:
        geojson, stats_before, stats_after = regularize(
            geojson,
            tolerance=args.tolerance,
            min_area=args.min_area,
            console=console,
        )

    # Save results
    console.print(f"\n[bold]保存结果...[/bold]")
    mask_path, geojson_path = save_results(prediction, geojson, args.output, console)

    # Class statistics
    if Table:
        console.print("\n")
        table = Table(title="推理结果 - 类别统计")
        table.add_column("类别", style="cyan")
        table.add_column("ID", justify="center")
        table.add_column("像素数", justify="right", style="green")
        table.add_column("占比", justify="right")

        total_pixels = prediction.size
        for class_id in range(NUM_CLASSES):
            count = (prediction == class_id).sum()
            pct = count / total_pixels * 100
            bar = "█" * int(pct / 3)
            table.add_row(
                f"{CLASS_NAMES_ZH[class_id]} ({CLASS_NAMES[class_id]})",
                str(class_id),
                f"{count:,}",
                f"{pct:.1f}% {bar}",
            )
        console.print(table)

    # Summary
    console.print(f"\n[bold green]✓ 推理完成![/bold green]")
    console.print(f"  耗时: {elapsed:.2f}s")
    console.print(f"  预测: {mask_path}")
    console.print(f"  矢量: {geojson_path}")

    if stats_before and stats_after:
        console.print(f"  正则化: {stats_before['count']} -> {stats_after['count']} 要素")


if __name__ == "__main__":
    main()
