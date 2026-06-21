"""模型训练路由"""
import uuid
import traceback
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from config import (
    MODEL_CONFIG, TRAIN_CONFIG, MODELS_DIR,
    SAMPLES_DIR, NUM_CLASSES,
)

router = APIRouter()

# 后台任务状态跟踪
_tasks = {}


class TrainRequest(BaseModel):
    model_name: str = "deeplabv3p_resnet50"
    epochs: int = TRAIN_CONFIG["epochs"]
    batch_size: int = TRAIN_CONFIG["batch_size"]
    lr: float = TRAIN_CONFIG["learning_rate"]


class PrepareRequest(BaseModel):
    tile_size: int = TRAIN_CONFIG["tile_size"]
    stride: int = TRAIN_CONFIG["stride"]
    augmentation: bool = TRAIN_CONFIG["augmentation"]


def _run_prepare(task_id: str, req: PrepareRequest):
    """后台执行样本制备"""
    try:
        _tasks[task_id]["status"] = "running"
        _tasks[task_id]["progress"] = 0

        from geoai_core import SampleGenerator

        generator = SampleGenerator(
            tile_size=req.tile_size,
            stride=req.stride,
            augmentation=req.augmentation,
        )

        result = generator.generate(
            image_dir=str(SAMPLES_DIR / "images"),
            label_dir=str(SAMPLES_DIR / "labels"),
            output_dir=str(SAMPLES_DIR),
            progress_callback=lambda p: _tasks[task_id].update({"progress": p}),
        )

        _tasks[task_id]["status"] = "completed"
        _tasks[task_id]["progress"] = 100
        _tasks[task_id]["result"] = {
            "num_samples": result.get("num_samples", 0),
            "message": f"样本制备完成，共 {result.get('num_samples', 0)} 个样本",
        }
    except Exception as e:
        _tasks[task_id]["status"] = "failed"
        _tasks[task_id]["error"] = str(e)
        _tasks[task_id]["traceback"] = traceback.format_exc()


def _run_train(task_id: str, req: TrainRequest):
    """后台执行模型训练"""
    try:
        _tasks[task_id]["status"] = "running"
        _tasks[task_id]["progress"] = 0

        from geoai_core import build_model, Trainer, create_dataloaders

        # 构建模型
        model = build_model(req.model_name, num_classes=NUM_CLASSES)

        # 创建数据加载器
        train_loader, val_loader = create_dataloaders(
            samples_dir=str(SAMPLES_DIR),
            batch_size=req.batch_size,
            num_workers=TRAIN_CONFIG["num_workers"],
            val_split=TRAIN_CONFIG["val_split"],
        )

        # 训练
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=req.epochs,
            lr=req.lr,
            weight_decay=TRAIN_CONFIG["weight_decay"],
            lr_scheduler=TRAIN_CONFIG["lr_scheduler"],
            early_stopping_patience=TRAIN_CONFIG["early_stopping_patience"],
            mixed_precision=TRAIN_CONFIG["mixed_precision"],
            save_dir=str(MODELS_DIR),
        )

        result = trainer.train(
            progress_callback=lambda epoch, total, metrics: _tasks[task_id].update({
                "progress": int(epoch / total * 100),
                "current_epoch": epoch,
                "metrics": metrics,
            }),
        )

        _tasks[task_id]["status"] = "completed"
        _tasks[task_id]["progress"] = 100
        _tasks[task_id]["result"] = {
            "model_path": result.get("model_path", ""),
            "best_val_iou": result.get("best_val_iou", 0),
            "epochs_trained": result.get("epochs_trained", 0),
            "message": f"训练完成，最佳验证 IoU: {result.get('best_val_iou', 0):.4f}",
        }
    except Exception as e:
        _tasks[task_id]["status"] = "failed"
        _tasks[task_id]["error"] = str(e)
        _tasks[task_id]["traceback"] = traceback.format_exc()


@router.post("/prepare-samples")
def prepare_samples(req: PrepareRequest, background_tasks: BackgroundTasks):
    """提交样本制备任务"""
    task_id = str(uuid.uuid4())[:8]
    _tasks[task_id] = {"status": "pending", "progress": 0, "type": "prepare"}
    background_tasks.add_task(_run_prepare, task_id, req)
    return {"task_id": task_id, "status": "pending"}


@router.post("/start")
def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    """提交模型训练任务"""
    if req.model_name not in MODEL_CONFIG:
        return {"error": f"不支持的模型: {req.model_name}", "available": list(MODEL_CONFIG.keys())}
    task_id = str(uuid.uuid4())[:8]
    _tasks[task_id] = {
        "status": "pending", "progress": 0, "type": "train",
        "model_name": req.model_name, "epochs": req.epochs,
    }
    background_tasks.add_task(_run_train, task_id, req)
    return {"task_id": task_id, "status": "pending"}


@router.get("/status/{task_id}")
def training_status(task_id: str):
    """查询训练任务状态"""
    if task_id not in _tasks:
        return {"error": "任务不存在", "task_id": task_id}
    return _tasks[task_id]


@router.get("/models")
def list_models():
    """列出可用的模型配置"""
    saved = []
    if MODELS_DIR.exists():
        saved = [
            {"name": f.name, "size": f.stat().st_size, "path": str(f.relative_to(MODELS_DIR))}
            for f in sorted(MODELS_DIR.glob("*.pth"))
        ]
    return {
        "models": MODEL_CONFIG,
        "saved_models": saved,
        "train_config": TRAIN_CONFIG,
    }
