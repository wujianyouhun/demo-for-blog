from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import DATA_DIR, PROFILES
from .data import generate_dataset, prepare_real_building_dataset
from .engine import compare_models, evaluate, train_one_model
from .inference import load_checkpoint, predict_geotiff, vectorize_mask
from .data import build_loaders
from .models import MODEL_NAMES


def _dataset_root(profile: str, value: str | None) -> Path:
    return Path(value).resolve() if value else DATA_DIR / "synthetic" / profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="U-Net GeoAI 多类地物分割")
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate-data", help="生成离线多类数据或整理真实建筑数据")
    generate.add_argument("--profile", choices=PROFILES, default="quick")
    generate.add_argument("--real-buildings", action="store_true")
    generate.add_argument("--output")

    train = subparsers.add_parser("train", help="训练一个模型")
    train.add_argument("--model", choices=MODEL_NAMES, default="unet")
    train.add_argument("--profile", choices=PROFILES, default="quick")
    train.add_argument("--dataset")
    train.add_argument("--epochs", type=int)
    train.add_argument("--binary", action="store_true")
    train.add_argument("--resume")

    compare = subparsers.add_parser("compare", help="顺序训练并比较四种模型")
    compare.add_argument("--profile", choices=PROFILES, default="quick")
    compare.add_argument("--dataset")
    compare.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))

    evaluate_parser = subparsers.add_parser("evaluate", help="评估检查点")
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--profile", choices=PROFILES, default="quick")
    evaluate_parser.add_argument("--dataset")

    predict = subparsers.add_parser("predict", help="对 GeoTIFF 执行重叠瓦片推理")
    predict.add_argument("--input", required=True)
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--output")
    predict.add_argument("--tile-size", type=int, default=256)
    predict.add_argument("--overlap", type=int, default=64)

    vectorize = subparsers.add_parser("vectorize", help="将类别掩膜矢量化")
    vectorize.add_argument("--mask", required=True)
    vectorize.add_argument("--output")

    serve = subparsers.add_parser("serve", help="启动 FastAPI 后端")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8028)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate-data":
        result = prepare_real_building_dataset(Path(args.output) if args.output else None) if args.real_buildings else generate_dataset(args.profile, Path(args.output) if args.output else None)
    elif args.command == "train":
        root = _dataset_root(args.profile, args.dataset)
        if not root.exists() and not args.binary:
            generate_dataset(args.profile, root)
        result = train_one_model(args.model, root, args.profile, args.epochs, args.binary, Path(args.resume) if args.resume else None,
                                 status_callback=lambda **state: print(state.get("message", "")))
    elif args.command == "compare":
        root = _dataset_root(args.profile, args.dataset)
        if not root.exists():
            generate_dataset(args.profile, root)
        result = compare_models(root, args.profile, args.models, status_callback=lambda **state: print(state.get("message", "")))
    elif args.command == "evaluate":
        model, state, device = load_checkpoint(Path(args.checkpoint))
        root = _dataset_root(args.profile, args.dataset)
        loaders = build_loaders(root, args.profile, binary=bool(state.get("binary", False)))
        result = evaluate(model, loaders["test"], device, int(state["num_classes"]))
    elif args.command == "predict":
        result = predict_geotiff(Path(args.input), Path(args.checkpoint), Path(args.output) if args.output else None,
                                 args.tile_size, args.overlap, progress=lambda value, message: print(f"{value:3d}% {message}"))
    elif args.command == "vectorize":
        result = vectorize_mask(Path(args.mask), Path(args.output) if args.output else None)
    else:
        import uvicorn
        uvicorn.run("backend.main:app", host=args.host, port=args.port)
        return 0
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
