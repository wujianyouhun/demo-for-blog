"""优化版 DeepLabV3：空间增强、有界采样、CE+Dice、分组学习率和完整断点。"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from common import ROOT, MODEL_DIR, choose_device, normalization_for_channels, training_model
from train_remote_sensing import TileDataset
from optimization import AugmentedDataset, dataset_counts, sampling_weights, epoch_pass


def atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def rng_state(generator):
    state = np.random.get_state()
    return dict(python=random.getstate(), torch=torch.get_rng_state(),
                numpy=[state[0], torch.from_numpy(state[1].astype("int64")), *state[2:]],
                loader=generator.get_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state, generator):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    n = state["numpy"]
    np.random.set_state((n[0], n[1].numpy().astype("uint32"), *n[2:]))
    generator.set_state(state["loader"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", type=Path, default=ROOT / "data/landcover_6class")
    p.add_argument("--output-dir", type=Path, default=MODEL_DIR / "landcover_6class_optimized")
    p.add_argument("--class-names", nargs="+")
    p.add_argument("--bands", type=int, nargs="+", help="明确指定 3 或 4 个通道；第四通道含义须自行核实")
    p.add_argument("--scale", type=float)
    p.add_argument("--tile-size", type=int)
    p.add_argument("--aspp-rates", type=int, nargs=3, default=[12, 24, 36],
                   help="保持步长 8；小切片可单独实验 6 12 18，必须重新训练和验证")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=3e-4)
    p.add_argument("--head-only-epochs", type=int, default=2)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dice-weight", type=float, default=.3)
    p.add_argument("--class-weighting", choices=["none", "inverse-sqrt"], default="inverse-sqrt")
    p.add_argument("--sampling", choices=["shuffle", "balanced"], default="balanced")
    p.add_argument("--max-sampling-ratio", type=float, default=3.)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--patience", type=int, default=15, help="0 关闭早停")
    p.add_argument("--min-delta", type=float, default=0.)
    p.add_argument("--grad-clip", type=float, default=1.)
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--from-scratch", action="store_true", help="仅用于流程验证或有充分数据时训练")
    p.add_argument("--init-checkpoint", type=Path, help="同类别、波段、缩放的旧权重，用作新实验初始化")
    p.add_argument("--resume", type=Path, help="本入口的 last.pth；参数须与原实验一致")
    args = p.parse_args(argv)
    path = args.dataset_dir / "dataset.json"
    cfg = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    names = args.class_names or cfg.get("class_names")
    if not names or len(names) < 2 or len(set(names)) != len(names):
        p.error("需要 dataset.json 的 class_names 或显式 --class-names，类别名不可重复。")
    bands = args.bands or cfg.get("bands", [1, 2, 3])
    scale = args.scale if args.scale is not None else cfg.get("scale", 10000.)
    size = args.tile_size if args.tile_size is not None else cfg.get("tile_size", 256)
    if (len(bands) not in (3, 4) or len(set(bands)) != len(bands) or min(bands) < 1
            or not np.isfinite(scale) or scale <= 0 or size < 32 or args.epochs < 1
            or args.batch_size < 1 or args.threads < 1 or args.patience < 0
            or not 0 <= args.head_only_epochs <= args.epochs
            or min(args.aspp_rates) < 1
            or not 0 <= args.dice_weight <= 1 or not np.isfinite(args.min_delta) or args.min_delta < 0):
        p.error("请检查波段、scale、tile-size、epochs、batch-size、热身轮数和损失权重。")
    for value in [args.backbone_lr, args.head_lr, args.grad_clip]:
        if not np.isfinite(value) or value <= 0:
            p.error("学习率与梯度裁剪阈值必须为正有限数。")
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        p.error("weight-decay 必须是非负有限数。")
    if sum([bool(args.resume), bool(args.init_checkpoint), args.from_scratch]) > 1:
        p.error("resume、init-checkpoint 与 from-scratch 互斥。")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError("实验目录非空，请更换 --output-dir；完整恢复使用 --resume。")
    if args.resume and args.resume.resolve() != (args.output_dir / "last.pth").resolve():
        raise ValueError("恢复时 output-dir 必须是 last.pth 所在目录，以保留同次实验的 best.pth。")
    torch.set_num_threads(args.threads)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    device = choose_device(args.device)
    norm = normalization_for_channels(len(bands))
    train = TileDataset(args.dataset_dir, "train", bands, scale, size, len(names), norm)
    val = TileDataset(args.dataset_dir, "val", bands, scale, size, len(names), norm)
    counts = dataset_counts(train)
    # 验证数据也提前读取，防止第一轮训练后才发现无效标签或错位。
    val_counts = dataset_counts(val).sum(0)
    if (counts.sum(0) <= 0).any() or (val_counts <= 0).any():
        raise ValueError("训练或验证缺少类别，请完善样本或另行制定评估类别，不自动伪造标签。")
    weights = None
    if args.class_weighting == "inverse-sqrt":
        weights = counts.sum(0).double().rsqrt()
        weights = (weights / weights.mean()).float().to(device)
    sampler = None
    if args.sampling == "balanced":
        sampler = WeightedRandomSampler(sampling_weights(counts, args.max_sampling_ratio),
                                        len(train), replacement=True, generator=generator)
    loaders = [DataLoader(AugmentedDataset(train, not args.no_augment), batch_size=args.batch_size,
                          sampler=sampler, shuffle=sampler is None, num_workers=0, generator=generator),
               DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=0)]
    model = training_model(len(names), pretrained=not (args.from_scratch or args.init_checkpoint or args.resume),
                           in_channels=len(bands), aspp_rates=args.aspp_rates).to(device)
    source = None
    if args.init_checkpoint or args.resume:
        source = torch.load(args.resume or args.init_checkpoint, map_location="cpu", weights_only=True)
        if (source.get("architecture") != "deeplabv3_resnet50" or list(source["class_names"]) != list(names)
                or source["bands"] != bands or source["scale"] != scale
                or source.get("normalization", normalization_for_channels(3)) != norm):
            raise ValueError("初始/恢复权重与类别、波段或预处理不同；请另建匹配实验。")
        model.load_state_dict(source["model"], strict=True)
    optimizer = torch.optim.AdamW([
        dict(params=model.backbone.parameters(), lr=args.backbone_lr),
        dict(params=model.classifier.parameters(), lr=args.head_lr),
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # 不将设备/输出目录作为模型语义，但为精确续训校验训练超参数。
    recipe = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k not in {"resume", "init_checkpoint", "output_dir", "device"}}
    history, best, best_epoch, bad_epochs, start = [], -1., 0, 0, 1
    if args.resume:
        if "optimizer" not in source or "rng" not in source:
            raise ValueError("resume 需要优化入口的 last.pth，旧 best 仅能使用 --init-checkpoint。")
        old_recipe = source["recipe"].copy()
        old_recipe.setdefault("aspp_rates", source.get("aspp_rates", [12, 24, 36]))
        # 初始化来源无需重复给出；其余设置保持完全一致。
        for key in ["from_scratch"]:
            old_recipe.pop(key, None)
        new_recipe = {k:v for k,v in recipe.items() if k != "from_scratch"}
        if old_recipe != new_recipe:
            raise ValueError("恢复配置不同于原实验；请保留原参数（包括计划 epochs）。")
        optimizer.load_state_dict(source["optimizer"])
        scheduler.load_state_dict(source["scheduler"])
        history, best = source["history"], source["best_miou"]
        best_epoch, bad_epochs, start = source["best_epoch"], source["bad_epochs"], source["epoch"] + 1
        restore_rng(source["rng"], generator)
        if start > args.epochs or (args.patience and bad_epochs >= args.patience):
            print("该实验已完成计划轮次或满足早停条件；如需继续改进，请以 best 初始化新实验。")
            return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(format_version=2, architecture="deeplabv3_resnet50", class_names=list(names),
                class_labels=cfg.get("class_labels") if list(names) == cfg.get("class_names") else None,
                bands=bands, in_channels=len(bands), scale=scale, normalization=norm,
                tile_size=size, aspp_rates=args.aspp_rates, ignore_index=255, dataset_dir=str(args.dataset_dir.resolve()),
                split_strategy=cfg.get("split_strategy", "existing train/val folders; audit independently"),
                preprocessing_note=cfg.get("preprocessing_note"), recipe=recipe,
                initialization=str(args.init_checkpoint or args.resume or ("random" if args.from_scratch else "COCO_WITH_VOC_LABELS_V1")))
    manifest = dict(**meta, train_pixels=counts.sum(0).tolist(), val_pixels=val_counts.tolist(),
                    train_tiles=len(train), val_tiles=len(val),
                    class_weights=weights.cpu().tolist() if weights is not None else None)
    (args.output_dir / "experiment.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"设备={device}；train={len(train)}, val={len(val)}；类别={names}；波段={bands}", flush=True)
    for epoch in range(start, args.epochs + 1):
        started = time.monotonic()
        frozen = epoch <= args.head_only_epochs
        for param in model.backbone.parameters():
            param.requires_grad_(not frozen)
        train_metrics = epoch_pass(model, loaders[0], device, names, optimizer, weights,
                                   args.dice_weight, args.grad_clip, frozen)
        val_metrics = epoch_pass(model, loaders[1], device, names)
        lr = [group["lr"] for group in optimizer.param_groups]
        scheduler.step()
        record = dict(epoch=epoch, learning_rates=lr, train_objective=train_metrics["objective"],
                      train_ce=train_metrics["ce_loss"], validation=val_metrics,
                      seconds=round(time.monotonic() - started, 2))
        history.append(record)
        miou = val_metrics["miou"]
        improved = miou > best
        # 保存真正最高分；min_delta 只控制早停计数，不丢弃微小提升。
        if miou > best + args.min_delta:
            bad_epochs = 0
        else:
            bad_epochs += 1
        if improved:
            best, best_epoch = miou, epoch
        current = dict(**meta, model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                       epoch=epoch, val_miou=miou, validation=val_metrics)
        if improved:
            atomic_save(current, args.output_dir / "best.pth")
        last = dict(**current, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                    history=history, best_miou=best, best_epoch=best_epoch, bad_epochs=bad_epochs,
                    rng=rng_state(generator))
        atomic_save(last, args.output_dir / "last.pth")
        (args.output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if args.patience and bad_epochs >= args.patience:
            print(f"早停：连续 {bad_epochs} 轮未有足够改善。", flush=True)
            break
    print(f"最佳 epoch={best_epoch}, mIoU={best:.6f}；{args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
