#!/usr/bin/env python3
"""执行双时相变化检测"""
import sys, argparse, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.logging import RichHandler
console = Console()
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler(console=console)])

def main():
    parser = argparse.ArgumentParser(description="双时相变化检测推理")
    parser.add_argument("--image-a", required=True, help="时相 A 影像")
    parser.add_argument("--image-b", required=True, help="时相 B 影像")
    parser.add_argument("--model", required=True, help="模型权重")
    parser.add_argument("--model-name", default="siamese_unet")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--min-area", type=int, default=30)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--visualize", action="store_true", help="生成可视化对比图")
    args = parser.parse_args()

    from config import OUTPUT_DIR
    from cdd.models import load_model
    from cdd.inference import ChangeDetector

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR

    console.print(f"[blue]加载模型: {args.model}[/blue]")
    model = load_model(args.model, model_name=args.model_name)

    console.print("[blue]执行变化检测[/blue]")
    detector = ChangeDetector(model=model, tile_size=256, overlap=32)

    result = detector.detect_and_vectorize(
        args.image_a, args.image_b, output_dir,
        threshold=args.threshold,
        smoothing_sigma=args.smoothing,
        min_area_pixels=args.min_area,
    )

    console.print(f"[green]变化图: {result['mask']}[/green]")
    console.print(f"[green]矢量: {result['vectors']}[/green]")

    # 可视化
    if args.visualize:
        console.print("[blue]生成可视化对比图[/blue]")
        from cdd.visualize import ChangeVisualizer

        ChangeVisualizer.create_comparison_image(
            args.image_a, args.image_b,
            change_map_path=str(result['mask']),
            output_path=str(output_dir / "compare_side.png"),
            mode="side_by_side",
        )
        ChangeVisualizer.create_difference_heatmap(
            args.image_a, args.image_b,
            output_path=str(output_dir / "diff_heatmap.png"),
        )
        ChangeVisualizer.create_change_overlay(
            args.image_a, str(result['mask']),
            output_path=str(output_dir / "change_overlay.png"),
        )
        console.print(f"[green]可视化图已生成[/green]")

if __name__ == "__main__":
    main()
