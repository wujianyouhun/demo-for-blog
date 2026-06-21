"""变化检测 API"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from config import MODELS_DIR, OUTPUT_DIR, SAMPLES_DIR

router = APIRouter()
_tasks = {}


class DetectRequest(BaseModel):
    image_a: str
    image_b: str
    model_path: str
    model_name: str = "siamese_unet"
    tile_size: int = 256
    overlap: int = 32
    threshold: float = 0.5
    smoothing_sigma: float = 1.0
    min_area: int = 30


class TrainRequest(BaseModel):
    model_name: str = "siamese_unet"
    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 1e-4


@router.post("/run")
async def run_detection(req: DetectRequest, background_tasks: BackgroundTasks):
    import uuid
    task_id = str(uuid.uuid4())[:8]
    _tasks[task_id] = {"status": "running", "message": "检测中...", "result": None}

    def _run():
        try:
            from cdd.models import load_model
            from cdd.inference import ChangeDetector
            model = load_model(req.model_path, model_name=req.model_name)
            det = ChangeDetector(model=model, tile_size=req.tile_size, overlap=req.overlap)
            r = det.detect_and_vectorize(req.image_a, req.image_b, OUTPUT_DIR, req.threshold, req.smoothing_sigma, req.min_area)
            _tasks[task_id] = {"status": "completed", "message": "检测完成", "result": {"mask": str(r["mask"]), "vectors": str(r["vectors"])}}
        except Exception as e:
            _tasks[task_id] = {"status": "failed", "message": str(e), "result": None}

    background_tasks.add_task(_run)
    return {"task_id": task_id, "status": "running", "message": "检测任务已启动"}


@router.get("/status/{task_id}")
async def detect_status(task_id: str):
    if task_id not in _tasks: raise HTTPException(404)
    return {"task_id": task_id, **_tasks[task_id]}


@router.post("/train")
async def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    import uuid

    # 训练前检查样本目录
    dir_a = SAMPLES_DIR / "time_a"
    dir_b = SAMPLES_DIR / "time_b"
    dir_lbl = SAMPLES_DIR / "labels"
    n_a = len(list(dir_a.glob("*.tif"))) if dir_a.is_dir() else 0
    n_b = len(list(dir_b.glob("*.tif"))) if dir_b.is_dir() else 0
    n_l = len(list(dir_lbl.glob("*.tif"))) if dir_lbl.is_dir() else 0
    if n_a == 0 or n_b == 0 or n_l == 0:
        raise HTTPException(
            status_code=400,
            detail=f"训练样本不足。时相A: {n_a} 张, 时相B: {n_b} 张, 标签: {n_l} 张。"
                   f"请先将训练样本放入 data/samples/time_a、time_b、labels 目录。",
        )
    if n_a != n_b or n_a != n_l:
        raise HTTPException(
            status_code=400,
            detail=f"样本数量不匹配: 时相A={n_a}, 时相B={n_b}, 标签={n_l}。三个目录文件数量需一致。",
        )

    task_id = str(uuid.uuid4())[:8]
    _tasks[task_id] = {"status": "running", "message": "训练中...", "result": None}

    def _run():
        try:
            import logging; logging.basicConfig(level=logging.INFO)
            from cdd.models import build_model
            from cdd.dataset import create_dataloaders
            from cdd.trainer import Trainer
            model = build_model(req.model_name)
            tl, vl = create_dataloaders(SAMPLES_DIR / "time_a", SAMPLES_DIR / "time_b", SAMPLES_DIR / "labels", req.batch_size)
            trainer = Trainer(model=model, lr=req.learning_rate)
            h = trainer.fit(tl, vl, epochs=req.epochs, save_dir=MODELS_DIR)
            _tasks[task_id] = {"status": "completed", "message": f"训练完成 F1={trainer.best_val_f1:.4f}",
                                "result": {"best_f1": trainer.best_val_f1,
                                           "history": {k: [float(v) for v in vals] for k, vals in h.items()}}}
        except Exception as e:
            _tasks[task_id] = {"status": "failed", "message": str(e), "result": None}

    background_tasks.add_task(_run)
    return {"task_id": task_id, "status": "running", "message": "训练任务已启动"}


@router.get("/models")
async def list_models():
    return {"models": [{"name": f.name, "size": f.stat().st_size, "path": str(f)} for f in sorted(MODELS_DIR.glob("*.pth"))]} if MODELS_DIR.exists() else {"models": []}


@router.get("/results")
async def list_results():
    m = [f for f in sorted(OUTPUT_DIR.glob("*.tif"))] if OUTPUT_DIR.exists() else []
    v = [f for f in sorted(OUTPUT_DIR.glob("*.gpkg"))] if OUTPUT_DIR.exists() else []
    return {"masks": [{"name": f.name, "path": str(f), "size": f.stat().st_size} for f in m],
            "vectors": [{"name": f.name, "path": str(f), "size": f.stat().st_size} for f in v]}
