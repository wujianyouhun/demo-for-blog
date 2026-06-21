"""要素提取路由"""
import uuid
import traceback
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from config import (
    MODELS_DIR, OUTPUT_DIR, INFERENCE_CONFIG,
    MODEL_CONFIG, NUM_CLASSES,
)

router = APIRouter()

# 后台任务状态跟踪
_tasks = {}


class PredictRequest(BaseModel):
    image_path: str
    model_path: Optional[str] = None
    model_name: str = "deeplabv3p_resnet50"
    tile_size: int = INFERENCE_CONFIG["tile_size"]
    overlap: int = INFERENCE_CONFIG["overlap"]
    batch_size: int = INFERENCE_CONFIG["batch_size"]
    threshold: float = INFERENCE_CONFIG["threshold"]
    smoothing_sigma: float = INFERENCE_CONFIG["smoothing_sigma"]


def _run_predict(task_id: str, req: PredictRequest):
    """后台执行推理"""
    try:
        _tasks[task_id]["status"] = "running"
        _tasks[task_id]["progress"] = 0

        from geoai_core import build_model, load_model, InferenceEngine
        from pathlib import Path
        import time

        # 加载模型
        model = build_model(req.model_name, num_classes=NUM_CLASSES)
        if req.model_path:
            model = load_model(model, req.model_path)
        else:
            # 尝试加载最新的模型权重
            saved = sorted(MODELS_DIR.glob("*.pth"), key=lambda f: f.stat().st_mtime, reverse=True)
            if saved:
                model = load_model(model, str(saved[0]))

        model.eval()

        # 创建推理引擎
        engine = InferenceEngine(
            model=model,
            device="cuda",
            tile_size=req.tile_size,
            overlap=req.overlap,
            batch_size=req.batch_size,
        )

        # 生成输出路径
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_path = str(OUTPUT_DIR / f"prediction_{ts}.tif")

        # 执行推理
        result = engine.predict(
            input_path=req.image_path,
            output_path=output_path,
            threshold=req.threshold,
            smoothing_sigma=req.smoothing_sigma,
            progress_callback=lambda p: _tasks[task_id].update({"progress": p}),
        )

        _tasks[task_id]["status"] = "completed"
        _tasks[task_id]["progress"] = 100
        _tasks[task_id]["result"] = {
            "output_path": output_path,
            "filename": Path(output_path).name,
            "message": f"推理完成，结果保存至 {Path(output_path).name}",
        }
    except Exception as e:
        _tasks[task_id]["status"] = "failed"
        _tasks[task_id]["error"] = str(e)
        _tasks[task_id]["traceback"] = traceback.format_exc()


@router.post("/predict")
def predict(req: PredictRequest, background_tasks: BackgroundTasks):
    """提交推理任务"""
    task_id = str(uuid.uuid4())[:8]
    _tasks[task_id] = {
        "status": "pending", "progress": 0, "type": "predict",
        "image_path": req.image_path,
    }
    background_tasks.add_task(_run_predict, task_id, req)
    return {"task_id": task_id, "status": "pending"}


@router.get("/status/{task_id}")
def prediction_status(task_id: str):
    """查询推理任务状态"""
    if task_id not in _tasks:
        return {"error": "任务不存在", "task_id": task_id}
    return _tasks[task_id]


@router.get("/result/{task_id}")
def get_result(task_id: str):
    """获取推理结果文件信息"""
    if task_id not in _tasks:
        return {"error": "任务不存在", "task_id": task_id}

    task = _tasks[task_id]
    if task["status"] != "completed":
        return {"error": "任务尚未完成", "status": task["status"]}

    result = task.get("result", {})
    output_path = result.get("output_path", "")

    from pathlib import Path
    if output_path and Path(output_path).exists():
        f = Path(output_path)
        return {
            "filename": f.name,
            "path": str(f),
            "size": f.stat().st_size,
            "suffix": f.suffix,
        }
    return {"error": "结果文件不存在"}
