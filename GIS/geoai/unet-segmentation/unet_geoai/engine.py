from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import (
    CHECKPOINT_DIR, CLASS_NAMES, DEFAULT_LR, DEFAULT_SEED, DEFAULT_WEIGHT_DECAY,
    NUM_CLASSES, PROFILES, REPORT_DIR,
)
from .data import build_loaders
from .metrics import CrossEntropyDiceLoss, boundary_f1, confusion_matrix, metrics_from_confusion
from .models import MODEL_NAMES, build_model, count_parameters

StatusCallback = Callable[..., None] | None


def set_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _emit(callback: StatusCallback, **values: Any) -> None:
    if callback:
        callback(**values)


def evaluate(model: torch.nn.Module, loader, device: torch.device, num_classes: int) -> dict:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    boundary_scores: list[float] = []
    elapsed = 0.0
    images = 0
    model.eval()
    with torch.no_grad():
        for batch_images, batch_masks in loader:
            batch_images = batch_images.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            logits = model(batch_images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - start
            predictions = logits.argmax(1).cpu().numpy()
            truths = batch_masks.numpy()
            images += len(predictions)
            for prediction, truth in zip(predictions, truths):
                matrix += confusion_matrix(prediction, truth, num_classes)
                boundary_scores.append(boundary_f1(prediction, truth, num_classes))
    metrics = metrics_from_confusion(matrix)
    metrics["boundary_f1"] = float(np.mean(boundary_scores)) if boundary_scores else 0.0
    metrics["images_per_second"] = float(images / max(elapsed, 1e-8))
    return metrics


def train_one_model(
    model_name: str,
    dataset_root: Path,
    profile: str = "quick",
    epochs: int | None = None,
    binary: bool = False,
    resume: Path | None = None,
    status_callback: StatusCallback = None,
    cancel_event=None,
) -> dict:
    if profile not in PROFILES:
        raise ValueError(f"未知训练配置: {profile}")
    if model_name not in MODEL_NAMES:
        raise ValueError(f"未知模型: {model_name}")
    set_seed()
    config = dict(PROFILES[profile])
    max_epochs = int(epochs or config["epochs"])
    num_classes = 2 if binary else NUM_CLASSES
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, num_classes=num_classes, base_channels=config["base_channels"]).to(device)
    criterion = CrossEntropyDiceLoss(num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=DEFAULT_LR, weight_decay=DEFAULT_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loaders = build_loaders(Path(dataset_root), profile, binary=binary)
    start_epoch = 0
    best_miou = -1.0
    history: list[dict] = []
    checkpoint_dir = CHECKPOINT_DIR / ("real_buildings" if binary else profile) / model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pth"

    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_epoch = int(state["epoch"]) + 1
        best_miou = float(state.get("best_miou", -1))
        history = list(state.get("history", []))

    patience_counter = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    training_started = time.perf_counter()
    for epoch in range(start_epoch, max_epochs):
        if cancel_event is not None and cancel_event.is_set():
            break
        model.train()
        running_loss = 0.0
        for step, (images, masks) in enumerate(loaders["train"]):
            if cancel_event is not None and cancel_event.is_set():
                break
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())
            overall = ((epoch + (step + 1) / len(loaders["train"])) / max_epochs) * 100
            _emit(status_callback, progress=int(overall), stage=f"train:{model_name}", message=f"{model_name} epoch {epoch + 1}/{max_epochs} batch {step + 1}/{len(loaders['train'])}")
        scheduler.step()
        val_metrics = evaluate(model, loaders["val"], device, num_classes)
        row = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(len(loaders["train"]), 1),
            "val_miou": val_metrics["miou"],
            "val_mdice": val_metrics["mdice"],
            "val_boundary_f1": val_metrics["boundary_f1"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        state = {
            "model_name": model_name,
            "profile": profile,
            "binary": binary,
            "num_classes": num_classes,
            "base_channels": config["base_channels"],
            "epoch": epoch,
            "best_miou": best_miou,
            "history": history,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(state, checkpoint_dir / "last.pth")
        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            state["best_miou"] = best_miou
            torch.save(state, best_path)
            patience_counter = 0
        else:
            patience_counter += 1
        _emit(status_callback, metrics={"history": history, "latest": row, "best_miou": best_miou})
        if patience_counter >= config["patience"]:
            break

    if not best_path.exists():
        torch.save(state, best_path)
    best_state = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_state["model_state_dict"])
    test_metrics = evaluate(model, loaders["test"], device, num_classes)
    test_metrics["parameters"] = count_parameters(model)
    test_metrics["peak_vram_mb"] = float(torch.cuda.max_memory_allocated() / 1024**2) if device.type == "cuda" else 0.0
    test_metrics["training_seconds"] = float(time.perf_counter() - training_started)
    result = {
        "model": model_name,
        "profile": profile,
        "binary": binary,
        "checkpoint": str(best_path),
        "epochs_trained": len(history),
        "history": history,
        "metrics": test_metrics,
    }
    (checkpoint_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _write_comparison_report(results: list[dict], profile: str) -> dict:
    report_dir = REPORT_DIR / f"comparison_{profile}"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "comparison.json"
    csv_path = report_dir / "comparison.csv"
    png_path = report_dir / "comparison.png"
    html_path = report_dir / "comparison.html"
    payload = {"profile": profile, "classes": CLASS_NAMES, "results": results}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    columns = ["model", "miou", "mdice", "boundary_f1", "pixel_accuracy", "parameters", "images_per_second", "peak_vram_mb", "training_seconds"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for result in results:
            writer.writerow({"model": result["model"], **{key: result["metrics"].get(key, 0) for key in columns[1:]}})
    labels = [result["model"] for result in results]
    x = np.arange(len(labels))
    width = 0.25
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(x - width, [r["metrics"]["miou"] for r in results], width, label="mIoU")
    axis.bar(x, [r["metrics"]["mdice"] for r in results], width, label="mDice")
    axis.bar(x + width, [r["metrics"]["boundary_f1"] for r in results], width, label="Boundary F1")
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1)
    axis.set_title(f"U-Net 结构对比 - {profile}")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(png_path, dpi=160)
    plt.close(figure)
    rows = "".join(
        f"<tr><td>{r['model']}</td><td>{r['metrics']['miou']:.4f}</td><td>{r['metrics']['mdice']:.4f}</td><td>{r['metrics']['boundary_f1']:.4f}</td><td>{r['metrics']['parameters']:,}</td><td>{r['metrics']['images_per_second']:.2f}</td></tr>"
        for r in results
    )
    html_path.write_text(
        "<!doctype html><meta charset='utf-8'><title>U-Net 对比报告</title>"
        "<style>body{font-family:system-ui;margin:36px;color:#172133}table{border-collapse:collapse;width:100%}th,td{padding:10px;border:1px solid #dbe2ea;text-align:right}th:first-child,td:first-child{text-align:left}img{max-width:100%}</style>"
        f"<h1>U-Net 结构对比报告</h1><p>训练配置：{profile}</p><img src='comparison.png'>"
        f"<table><thead><tr><th>模型</th><th>mIoU</th><th>mDice</th><th>边界 F1</th><th>参数量</th><th>图像/秒</th></tr></thead><tbody>{rows}</tbody></table>",
        encoding="utf-8",
    )
    return {"json": str(json_path), "csv": str(csv_path), "png": str(png_path), "html": str(html_path)}


def compare_models(dataset_root: Path, profile: str = "quick", models: list[str] | None = None,
                   status_callback: StatusCallback = None, cancel_event=None) -> dict:
    selected = models or list(MODEL_NAMES)
    results = []
    for index, model_name in enumerate(selected):
        if cancel_event is not None and cancel_event.is_set():
            break
        _emit(status_callback, stage=f"compare:{model_name}", message=f"训练对比模型 {index + 1}/{len(selected)}: {model_name}")

        def scaled_callback(**values):
            local_progress = values.pop("progress", 0)
            values["progress"] = int((index + local_progress / 100) / len(selected) * 100)
            _emit(status_callback, **values)

        results.append(train_one_model(model_name, dataset_root, profile, status_callback=scaled_callback, cancel_event=cancel_event))
    report = _write_comparison_report(results, profile)
    unet = next((item for item in results if item["model"] == "unet"), None)
    no_skip = next((item for item in results if item["model"] == "no_skip"), None)
    skip_effect = None
    if unet and no_skip:
        skip_effect = {
            "miou_gain": unet["metrics"]["miou"] - no_skip["metrics"]["miou"],
            "boundary_f1_gain": unet["metrics"]["boundary_f1"] - no_skip["metrics"]["boundary_f1"],
        }
    return {"profile": profile, "results": results, "report": report, "skip_effect": skip_effect}
