"""普通 RGB 图片：预训练模型调用、类别掩膜、着色叠加与类别统计。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from common import ROOT, WEIGHTS, choose_device, palette, pretrained_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "out" / "basic")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = choose_device(args.device)
    model = pretrained_model().to(device).eval()
    image = Image.open(args.image).convert("RGB")
    # 官方变换：短边缩放到 520、转张量、ImageNet 归一化。
    inputs = WEIGHTS.transforms()(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(inputs)["out"]
        # 先插值 logits 再 argmax，恢复到原图大小。
        logits = F.interpolate(logits, size=(image.height, image.width), mode="bilinear", align_corners=False)
        mask = logits.argmax(1)[0].cpu().numpy().astype("uint8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(args.output_dir / "mask.png")
    colored = Image.fromarray(mask).convert("P")
    colored.putpalette(palette(256))
    colored.save(args.output_dir / "colored_mask.png")
    Image.blend(image, colored.convert("RGB"), 0.45).save(args.output_dir / "overlay.png")
    categories = WEIGHTS.meta["categories"]
    ids, counts = np.unique(mask, return_counts=True)
    summary = [dict(class_id=int(i), name=categories[i], pixels=int(n), fraction=float(n / mask.size))
               for i, n in zip(ids, counts)]
    (args.output_dir / "classes.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"输出：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
