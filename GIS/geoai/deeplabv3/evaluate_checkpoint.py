"""在固定标签切片上评价 checkpoint；全局混淆矩阵，不修改验证数据。"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common import choose_device, load_checkpoint
from train_remote_sensing import TileDataset
from optimization import epoch_pass


def dataset_signature(dataset):
    digest = hashlib.sha256()
    for image, mask in dataset.items:
        for path in (image, mask):
            digest.update(path.name.encode("utf-8"))
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def evaluate(dataset_dir, checkpoint, split="val", device="auto", batch_size=2):
    device = choose_device(device)
    model, meta = load_checkpoint(checkpoint, device)
    config_path = dataset_dir / "dataset.json"
    if config_path.exists():
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        if cfg.get("class_names") != list(meta["class_names"]):
            raise ValueError("数据集类别顺序与权重不一致。")
    dataset = TileDataset(dataset_dir, split, meta["bands"], meta["scale"], meta["tile_size"],
                          len(meta["class_names"]), meta.get("normalization"))
    metrics = epoch_pass(model, DataLoader(dataset, batch_size=batch_size), device, meta["class_names"])
    return dict(checkpoint=str(checkpoint.resolve()), dataset=str(dataset_dir.resolve()), split=split,
                tiles=len(dataset), dataset_sha256=dataset_signature(dataset),
                epoch=meta.get("epoch"), metrics=metrics,
                note="固定切片直接推理；概率融合/TTA 必须另用真值栅格评价，不能套用此分数。")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        p.error("batch-size 与 threads 必须为正。")
    torch.set_num_threads(args.threads)
    result = evaluate(args.dataset_dir, args.checkpoint, args.split, args.device, args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
