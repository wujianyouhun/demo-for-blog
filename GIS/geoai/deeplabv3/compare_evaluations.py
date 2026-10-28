"""对齐数据集、split 与类别后比较两个评价报告，输出可审阅的 Markdown。"""
import argparse
import json
from pathlib import Path


def compare(baseline, candidate):
    for key in ["dataset", "split", "tiles"]:
        if baseline[key] != candidate[key]:
            raise ValueError(f"评价的 {key} 不一致，不能直接比较。")
    if not baseline.get("dataset_sha256") or baseline["dataset_sha256"] != candidate.get("dataset_sha256"):
        raise ValueError("样本文件指纹缺失或不同，请重新对相同数据集执行统一评价。")
    old, new = baseline["metrics"], candidate["metrics"]
    a, b = old["per_class"], new["per_class"]
    if [x["name"] for x in a] != [x["name"] for x in b] or [x["pixels"] for x in a] != [x["pixels"] for x in b]:
        raise ValueError("类别顺序或真值像元数量不同。")
    delta = new["miou"] - old["miou"]
    lines = ["# DeepLabV3 优化候选与原模型的同集评价", "",
             f"原模型 mIoU：{old['miou']:.6f}；候选：{new['miou']:.6f}；差值：{delta:+.6f}。", "",
             f"评价范围：{baseline['split']}，{baseline['tiles']} 个切片，{old['valid_pixels']:,} 个有效像元。", "",
             "| 类别 | 原 IoU | 候选 IoU | 原预测像元 | 候选预测像元 |", "|---|---:|---:|---:|---:|"]
    number = lambda value: f"{value:.6f}" if value is not None else "无定义"
    for x, y in zip(a, b):
        lines.append(f"| {x['name']} | {number(x['iou'])} | {number(y['iou'])} | {x['predicted_pixels']} | {y['predicted_pixels']} |")
    lines += ["", "候选无预测像元的类别：" + ("、".join(new["zero_prediction_classes"]) or "无"), "",
              "这只比较当前固定切片的直接推理，不代表独立区域精度，也不能证明单个优化组件的贡献。",
              "查看每类 IoU 与召回率后再决定是否使用候选参数。少量 mIoU 上升不能说明所有类别可用。", "",
              f"原权重：`{baseline['checkpoint']}`", f"候选：`{candidate['checkpoint']}`", ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    text = compare(json.loads(a.baseline.read_text(encoding="utf-8")), json.loads(a.candidate.read_text(encoding="utf-8")))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
