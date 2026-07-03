"""
GeoAI 图像分类 — 命令行单张/批量推理
=====================================
用法:
    # 单张
    python scripts/infer.py --image path/to/image.jpg
    # 批量目录
    python scripts/infer.py --dir path/to/images/ --output results.json
"""
import argparse, os, sys, json
from pathlib import Path

import torch
ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT.parent
SHARED_MODELS_DIR = Path(os.getenv("GEOAI_MODELS_DIR", str(REPO_ROOT / "models"))).expanduser()
if not SHARED_MODELS_DIR.is_absolute():
    SHARED_MODELS_DIR = (REPO_ROOT / SHARED_MODELS_DIR).resolve()
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "backend"))

def parse_args():
    p = argparse.ArgumentParser(description="GeoAI 命令行推理")
    p.add_argument("--image",      default=None, help="单张图像路径")
    p.add_argument("--dir",        default=None, help="批量推理目录")
    p.add_argument("--checkpoint", default=str(SHARED_MODELS_DIR / "Classification" / "checkpoints" / "best_model.pth"))
    p.add_argument("--device",     default="auto")
    p.add_argument("--output",     default=None, help="结果输出 JSON 路径")
    p.add_argument("--topk",       type=int, default=5)
    return p.parse_args()

def main():
    args = parse_args()
    from predictor import get_predictor
    predictor = get_predictor(args.checkpoint, args.device)
    predictor.load()

    results = []

    def infer_one(img_path):
        with open(img_path, "rb") as f:
            content = f.read()
        r = predictor.predict(content)
        r["file"] = str(img_path)
        return r

    if args.image:
        r = infer_one(args.image)
        results.append(r)
        print(f"\n📍 文件  : {r['file']}")
        print(f"   类别  : {r['class_name']} ({r['class_name_zh']})")
        print(f"   置信度: {r['confidence']*100:.2f}%")
        print(f"   推理  : {r['infer_time_ms']} ms")
        print(f"   描述  : {r['description']}")
        print(f"\n   Top-{args.topk}:")
        for i, item in enumerate(r["top5"][:args.topk]):
            bar = "█" * int(item["prob"] * 30)
            print(f"   {i+1}. {item['class']:<28} {item['prob']*100:6.2f}%  {bar}")

    elif args.dir:
        from tqdm import tqdm
        img_dir = Path(args.dir)
        exts = {".jpg",".jpeg",".png",".tif",".tiff"}
        images = [p for p in img_dir.rglob("*") if p.suffix.lower() in exts]
        print(f"🔍 发现 {len(images)} 张图像，开始推理...")
        for img_path in tqdm(images, desc="推理"):
            try:
                r = infer_one(img_path)
                results.append({"success": True, **r})
            except Exception as e:
                results.append({"success": False, "file": str(img_path), "error": str(e)})

        ok  = sum(1 for r in results if r.get("success"))
        print(f"\n✅ 推理完成: {ok}/{len(results)} 成功")

    else:
        print("请指定 --image 或 --dir"); return

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"💾 结果已保存: {args.output}")

if __name__ == "__main__":
    main()
