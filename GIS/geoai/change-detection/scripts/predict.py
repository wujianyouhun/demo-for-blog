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
    parser.add_argument("--engine", choices=["geoai", "cdd"], default="geoai",
                        help="geoai 使用预训练 ChangeStar；cdd 使用本项目训练权重")
    parser.add_argument("--image-a", required=True, help="时相 A 影像")
    parser.add_argument("--image-b", required=True, help="时相 B 影像")
    parser.add_argument("--model", default=None, help="cdd 引擎模型权重")
    parser.add_argument("--model-name", default=None,
                        help="geoai 模式为 ChangeStar 模型名，cdd 模式为网络结构")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--min-area", type=int, default=30)
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--visualize", action="store_true", help="生成可视化对比图")
    args = parser.parse_args()

    from config import OUTPUT_DIR

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR

    if args.engine == "geoai":
        from cdd.geoai_change import run_geoai_change_detection

        model_name = args.model_name or "s1_s1c1_vitb"
        console.print(f"[blue]使用 GeoAI ChangeStar: {model_name}[/blue]")
        result = run_geoai_change_detection(
            image_a=args.image_a,
            image_b=args.image_b,
            output_dir=output_dir,
            model_name=model_name,
            tile_size=args.tile_size or 1024,
            overlap=args.overlap or 64,
            threshold=args.threshold,
            device=args.device,
            visualize=args.visualize,
        )
    else:
        if not args.model:
            raise SystemExit("cdd 引擎需要 --model 指定训练好的 .pth 权重")
        from cdd.models import load_model
        from cdd.inference import ChangeDetector

        model_name = args.model_name or "siamese_unet"
        console.print(f"[blue]加载自训练模型: {args.model}[/blue]")
        model = load_model(args.model, model_name=model_name)

        console.print("[blue]执行变化检测[/blue]")
        detector = ChangeDetector(
            model=model,
            tile_size=args.tile_size or 256,
            overlap=args.overlap or 32,
            device=args.device or "auto",
        )
        result = detector.detect_and_vectorize(
            args.image_a, args.image_b, output_dir,
            threshold=args.threshold,
            smoothing_sigma=args.smoothing,
            min_area_pixels=args.min_area,
        )

    console.print(f"[green]变化图: {result['mask']}[/green]")
    console.print(f"[green]矢量: {result['vectors']}[/green]")

    # 可视化
    if args.visualize and args.engine == "cdd":
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
