"""变化检测 API"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Literal, Optional
from config import MODELS_DIR, OUTPUT_DIR, SAMPLES_DIR

router = APIRouter()
_tasks = {}


class DetectRequest(BaseModel):
    engine: Literal["geoai", "cdd"] = "geoai"
    image_a: str
    image_b: str
    model_path: Optional[str] = None
    model_name: Optional[str] = None
    tile_size: int = 1024
    overlap: int = 64
    threshold: float = 0.5
    smoothing_sigma: float = 1.0
    min_area: int = 30
    device: Optional[str] = None


class TrainRequest(BaseModel):
    model_name: str = "siamese_unet"
    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 1e-4


class SampleRequest(BaseModel):
    mode: Literal["synthetic", "weak-label", "vector-label"] = "synthetic"
    image_a: Optional[str] = None
    image_b: Optional[str] = None
    vector_label: Optional[str] = None
    num_samples: int = 100
    tile_size: int = 256
    stride: int = 256
    min_change_pixels: int = 20
    max_tiles: Optional[int] = None
    geoai_model: str = "s1_s1c1_vitb"
    threshold: float = 0.5
    overwrite: bool = True


@router.post("/run")
async def run_detection(req: DetectRequest, background_tasks: BackgroundTasks):
    import uuid
    task_id = str(uuid.uuid4())[:8]
    _tasks[task_id] = {"status": "running", "message": "检测中...", "result": None}

    if req.engine == "cdd" and not req.model_path:
        raise HTTPException(400, "cdd 自训练引擎需要 model_path；GeoAI 引擎无需模型权重。")

    def _run():
        try:
            if req.engine == "geoai":
                from cdd.geoai_change import DEFAULT_GEOAI_MODEL, run_geoai_change_detection

                r = run_geoai_change_detection(
                    image_a=req.image_a,
                    image_b=req.image_b,
                    output_dir=OUTPUT_DIR,
                    model_name=req.model_name or DEFAULT_GEOAI_MODEL,
                    tile_size=req.tile_size,
                    overlap=req.overlap,
                    threshold=req.threshold,
                    device=req.device,
                    visualize=True,
                )
            else:
                from cdd.models import load_model
                from cdd.inference import ChangeDetector

                model = load_model(req.model_path, model_name=req.model_name or "siamese_unet")
                det = ChangeDetector(
                    model=model,
                    tile_size=req.tile_size or 256,
                    overlap=req.overlap or 32,
                    device=req.device or "auto",
                )
                r = det.detect_and_vectorize(
                    req.image_a, req.image_b, OUTPUT_DIR,
                    req.threshold, req.smoothing_sigma, req.min_area,
                )
                r = {"engine": "cdd", "mask": str(r["mask"]), "vectors": str(r["vectors"])}

            _tasks[task_id] = {"status": "completed", "message": "检测完成", "result": r}
        except Exception as e:
            _tasks[task_id] = {"status": "failed", "message": str(e), "result": None}

    background_tasks.add_task(_run)
    return {"task_id": task_id, "status": "running", "message": "检测任务已启动"}


@router.get("/status/{task_id}")
async def detect_status(task_id: str):
    if task_id not in _tasks: raise HTTPException(404)
    return {"task_id": task_id, **_tasks[task_id]}


@router.get("/samples")
async def get_samples():
    from cdd.sample_builder import validate_samples

    return validate_samples(SAMPLES_DIR)


@router.post("/samples")
async def create_samples(req: SampleRequest, background_tasks: BackgroundTasks):
    import uuid

    if req.mode in {"weak-label", "vector-label"} and (not req.image_a or not req.image_b):
        raise HTTPException(400, "weak-label/vector-label 模式需要 image_a 和 image_b")
    if req.mode == "vector-label" and not req.vector_label:
        raise HTTPException(400, "vector-label 模式需要 vector_label")

    task_id = str(uuid.uuid4())[:8]
    _tasks[task_id] = {"status": "running", "message": "样本制作中...", "result": None}

    def _run():
        try:
            from cdd.sample_builder import (
                generate_synthetic_samples,
                generate_vector_label_samples,
                generate_weak_label_samples,
            )

            if req.mode == "synthetic":
                r = generate_synthetic_samples(
                    SAMPLES_DIR,
                    num_samples=req.num_samples,
                    tile_size=req.tile_size,
                    overwrite=req.overwrite,
                )
            elif req.mode == "weak-label":
                r = generate_weak_label_samples(
                    req.image_a,
                    req.image_b,
                    SAMPLES_DIR,
                    tile_size=req.tile_size,
                    stride=req.stride,
                    min_change_pixels=req.min_change_pixels,
                    max_tiles=req.max_tiles,
                    model_name=req.geoai_model,
                    threshold=req.threshold,
                    overwrite=req.overwrite,
                )
            else:
                r = generate_vector_label_samples(
                    req.image_a,
                    req.image_b,
                    req.vector_label,
                    SAMPLES_DIR,
                    tile_size=req.tile_size,
                    stride=req.stride,
                    min_change_pixels=req.min_change_pixels,
                    max_tiles=req.max_tiles,
                    overwrite=req.overwrite,
                )
            _tasks[task_id] = {"status": "completed", "message": "样本制作完成", "result": r}
        except Exception as e:
            _tasks[task_id] = {"status": "failed", "message": str(e), "result": None}

    background_tasks.add_task(_run)
    return {"task_id": task_id, "status": "running", "message": "样本制作任务已启动"}


@router.post("/train")
async def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    import uuid

    # 训练前检查样本目录
    from cdd.sample_builder import validate_samples

    check = validate_samples(SAMPLES_DIR)
    counts = check["counts"]
    if not check["ok"]:
        raise HTTPException(
            status_code=400,
            detail=f"{check['message']} 当前数量: 时相A={counts['time_a']}, "
                   f"时相B={counts['time_b']}, 标签={counts['labels']}。",
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
    from cdd.geoai_change import list_geoai_changestar_models

    cdd_models = [{"name": f.name, "size": f.stat().st_size, "path": str(f)} for f in sorted(MODELS_DIR.glob("*.pth"))] if MODELS_DIR.exists() else []
    geoai_models = [{"name": k, "description": v} for k, v in list_geoai_changestar_models().items()]
    return {"models": cdd_models, "geoai_models": geoai_models}


@router.get("/results")
async def list_results():
    m = [f for f in sorted(OUTPUT_DIR.glob("*.tif"))] if OUTPUT_DIR.exists() else []
    v = [f for f in sorted(OUTPUT_DIR.glob("*.gpkg"))] if OUTPUT_DIR.exists() else []
    p = [f for f in sorted(OUTPUT_DIR.glob("*.png"))] if OUTPUT_DIR.exists() else []
    return {"masks": [{"name": f.name, "path": str(f), "size": f.stat().st_size} for f in m],
            "vectors": [{"name": f.name, "path": str(f), "size": f.stat().st_size} for f in v],
            "previews": [{"name": f.name, "path": str(f), "size": f.stat().st_size} for f in p]}
