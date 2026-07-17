from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GEOAI_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(GEOAI_ROOT))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from web_common import TaskRegistry, list_files
from unet_geoai.config import (
    CHECKPOINT_DIR, CLASS_COLORS, CLASS_NAMES, CLASS_NAMES_ZH, DATA_DIR,
    NUM_CLASSES, OUTPUT_DIR, PROFILES, REAL_BUILDING_DIR,
)
from unet_geoai.data import generate_dataset, prepare_real_building_dataset
from unet_geoai.engine import compare_models, train_one_model
from unet_geoai.inference import predict_geotiff
from unet_geoai.models import ARCHITECTURE, MODEL_NAMES, build_model, count_parameters

app = FastAPI(title="U-Net GeoAI 多类分割 API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
tasks = TaskRegistry(max_workers=1)
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class GenerateRequest(BaseModel):
    profile: str = "quick"
    real_buildings: bool = False


class TrainRequest(BaseModel):
    model: str = "unet"
    profile: str = "quick"
    dataset: str | None = None
    epochs: int | None = Field(default=None, ge=1, le=200)
    binary: bool = False
    resume: str | None = None


class CompareRequest(BaseModel):
    profile: str = "quick"
    dataset: str | None = None
    models: list[str] = list(MODEL_NAMES)


class PredictRequest(BaseModel):
    input_path: str
    checkpoint: str
    tile_size: int = Field(default=256, ge=64, le=2048)
    overlap: int = Field(default=64, ge=0, le=1024)


def _generate_job(request: GenerateRequest, *, update, cancel_event, **_):
    if request.real_buildings:
        update(stage="prepare-real", message="整理真实建筑样本")
        return prepare_real_building_dataset()
    return generate_dataset(request.profile, progress=lambda value, message: update(progress=value, stage="generate", message=message))


def _train_job(request: TrainRequest, *, update, cancel_event, **_):
    root = Path(request.dataset).expanduser().resolve() if request.dataset else DATA_DIR / "synthetic" / request.profile
    if request.binary and request.dataset is None:
        payload = prepare_real_building_dataset()
        root = Path(payload["root"])
    elif not root.exists():
        generate_dataset(request.profile, root, progress=lambda value, message: update(progress=min(value // 5, 20), stage="generate", message=message))
    return train_one_model(request.model, root, request.profile, request.epochs, request.binary,
                           Path(request.resume) if request.resume else None, update, cancel_event)


def _compare_job(request: CompareRequest, *, update, cancel_event, **_):
    root = Path(request.dataset).expanduser().resolve() if request.dataset else DATA_DIR / "synthetic" / request.profile
    if not root.exists():
        generate_dataset(request.profile, root, progress=lambda value, message: update(progress=min(value // 5, 20), stage="generate", message=message))
    return compare_models(root, request.profile, request.models, update, cancel_event)


def _predict_job(request: PredictRequest, *, update, cancel_event, task_id, **_):
    input_path = Path(request.input_path).expanduser().resolve()
    checkpoint = Path(request.checkpoint).expanduser().resolve()
    if not input_path.exists() or input_path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("输入必须是存在的 GeoTIFF")
    if not checkpoint.exists() or checkpoint.suffix.lower() not in {".pth", ".pt"}:
        raise ValueError("检查点不存在或格式不支持")
    return predict_geotiff(input_path, checkpoint, OUTPUT_DIR / f"task_{task_id}", request.tile_size, request.overlap,
                           lambda value, message: update(progress=value, stage="predict", message=message))


@app.get("/api/health")
def health():
    import torch
    return {"status": "ok", "cuda": torch.cuda.is_available(), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}


@app.get("/api/config")
def config():
    return {
        "classes": [{"id": i, "name": name, "name_zh": CLASS_NAMES_ZH[i], "color": CLASS_COLORS[i]} for i, name in enumerate(CLASS_NAMES)],
        "profiles": PROFILES, "real_building_dir": str(REAL_BUILDING_DIR),
        "ports": {"backend": 8028, "frontend": 5188},
    }


@app.get("/api/models")
def models():
    details = []
    for name in MODEL_NAMES:
        model = build_model(name, base_channels=16)
        details.append({"name": name, "parameters_quick": count_parameters(model)})
    return {"models": details, "checkpoints": list_files(CHECKPOINT_DIR, {".pth", ".pt"}), "architecture": [stage.__dict__ for stage in ARCHITECTURE]}


@app.get("/api/datasets")
def datasets():
    result = []
    for profile in PROFILES:
        root = DATA_DIR / "synthetic" / profile
        result.append({"name": profile, "path": str(root), "ready": (root / "manifest.json").exists()})
    return {"datasets": result, "real_buildings": (DATA_DIR / "real_buildings" / "manifest.json").exists(), "uploads": list_files(UPLOAD_DIR)}


@app.post("/api/uploads")
async def upload(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".tif", ".tiff"}:
        raise HTTPException(400, "仅支持 GeoTIFF")
    target = UPLOAD_DIR / Path(file.filename or "upload.tif").name
    target.write_bytes(await file.read())
    return {"name": target.name, "path": str(target), "size": target.stat().st_size}


@app.post("/api/datasets/generate")
def generate(request: GenerateRequest):
    if request.profile not in PROFILES:
        raise HTTPException(400, "未知训练配置")
    return tasks.submit("generate", _generate_job, request)


@app.post("/api/train")
def train(request: TrainRequest):
    if request.model not in MODEL_NAMES or request.profile not in PROFILES:
        raise HTTPException(400, "未知模型或训练配置")
    return tasks.submit("train", _train_job, request)


@app.post("/api/compare")
def compare(request: CompareRequest):
    if request.profile not in PROFILES or any(name not in MODEL_NAMES for name in request.models):
        raise HTTPException(400, "未知模型或训练配置")
    return tasks.submit("compare", _compare_job, request)


@app.post("/api/evaluate")
def evaluate_endpoint(request: TrainRequest):
    if not request.resume:
        raise HTTPException(400, "evaluate 需要 resume 指向检查点")
    request.epochs = 1
    return tasks.submit("evaluate", _train_job, request)


@app.post("/api/predict")
def predict(request: PredictRequest):
    if request.overlap >= request.tile_size:
        raise HTTPException(400, "overlap 必须小于 tile_size")
    return tasks.submit("predict", _predict_job, request)


@app.get("/api/tasks")
def task_list():
    return {"tasks": tasks.list()}


@app.get("/api/tasks/{task_id}")
def task_status(task_id: str):
    try:
        return tasks.public(task_id)
    except KeyError:
        raise HTTPException(404, "任务不存在")


@app.post("/api/tasks/{task_id}/cancel")
def cancel(task_id: str):
    try:
        return tasks.cancel(task_id)
    except KeyError:
        raise HTTPException(404, "任务不存在")


@app.get("/api/results/{task_id}")
def result(task_id: str):
    try:
        task = tasks.public(task_id)
    except KeyError:
        raise HTTPException(404, "任务不存在")
    if task["status"] != "completed":
        raise HTTPException(409, "任务尚未完成")
    return task["result"]


@app.get("/api/download")
def download(path: str):
    candidate = Path(path).expanduser().resolve()
    roots = [OUTPUT_DIR.resolve(), CHECKPOINT_DIR.resolve(), DATA_DIR.resolve()]
    if not candidate.is_file() or not any(candidate == root or root in candidate.parents for root in roots):
        raise HTTPException(404, "成果不存在或不允许下载")
    return FileResponse(candidate, filename=candidate.name)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8028)
