"""GeoAI 统一入口：调用六种 U-Net 变体进行遥感分类与矢量提取。

示例：
    python Unet/geoai_unet.py --model resunet --task both --device auto
    python Unet/geoai_unet.py --model all --task both --min-vector-area 100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from infer_multiclass import parse_args, run_inference
from unet_models import MODELS


ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT.parent / "models" / "Unet" / "qilian_multiclass_5class_refined"
# 与历史栅格输出分开，首次运行统一入口不会覆盖已有的模型对比结果。
OUTPUT_ROOT = ROOT / "out" / "geoai_results"


def defaults_for(model_name: str) -> dict:
    """所有架构共享同一批已制作样本、波段顺序和空间参考配置。"""
    return {
        "input": ROOT / "DATA" / "Qilian.tif",
        "weights": MODEL_ROOT / model_name / "best.pth",
        "output": OUTPUT_ROOT / model_name,
        "tile_size": 256,
        "stride": 192,
        "target_crs": "EPSG:3857",
        "target_resolution": 10.0,
        "device": "auto",
    }


def arguments_for_model(arguments: list[str], model_name: str, split_output: bool) -> list[str]:
    """在 ``--model all`` 时将用户指定的输出根目录按模型拆分。"""
    if not split_output:
        return arguments
    result = list(arguments)
    for index, value in enumerate(result):
        if value == "--output-dir" and index + 1 < len(result):
            result[index + 1] = str(Path(result[index + 1]) / model_name)
        elif value.startswith("--output-dir="):
            root = Path(value.split("=", 1)[1])
            result[index] = f"--output-dir={root / model_name}"
    return result


def main() -> None:
    controller = argparse.ArgumentParser(add_help=False)
    controller.add_argument(
        "--model",
        choices=tuple(MODELS) + ("all",),
        default="resunet",
        help="模型名称；all 会按顺序运行全部六种模型。",
    )
    selected, remaining = controller.parse_known_args()
    if "-h" in remaining or "--help" in remaining:
        print("GeoAI U-Net 统一入口；--model 可选：" + ", ".join((*MODELS, "all")))
        print("使用 --model all 时，若指定 --output-dir，会自动为每种模型创建同名子目录。\n")
        original_argv = sys.argv
        try:
            sys.argv = [original_argv[0], "--help"]
            parse_args(selected.model if selected.model != "all" else "resunet", defaults_for("resunet"))
        finally:
            sys.argv = original_argv
        return
    # 将其余参数原样交给共用的 GeoTIFF 推理参数解析器，保证单模型入口和统一入口行为一致。
    names = tuple(MODELS) if selected.model == "all" else (selected.model,)
    for model_name in names:
        original_argv = sys.argv
        try:
            sys.argv = [original_argv[0], *arguments_for_model(remaining, model_name, selected.model == "all")]
            run_inference(parse_args(model_name, defaults_for(model_name)))
        finally:
            sys.argv = original_argv


if __name__ == "__main__":
    main()
