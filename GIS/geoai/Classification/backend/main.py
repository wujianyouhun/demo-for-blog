"""
GeoAI 图像分类 — FastAPI 后端服务
===================================
用法:
    conda activate geoai
    cd F:/geoai
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

接口文档: http://localhost:8000/docs
"""
from __future__ import annotations
import os, sys, io, base64, time
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# ── 环境配置 ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
SHARED_MODELS_DIR = Path(os.getenv("GEOAI_MODELS_DIR", str(ROOT.parent / "models"))).expanduser()
if not SHARED_MODELS_DIR.is_absolute():
    SHARED_MODELS_DIR = (ROOT.parent / SHARED_MODELS_DIR).resolve()
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "backend"))

from predictor import get_predictor, CLASS_NAMES, CLASS_NAMES_ZH

MODEL_PATH  = os.getenv("MODEL_PATH",  str(SHARED_MODELS_DIR / "Classification" / "checkpoints" / "best_model.pth"))
DEVICE      = os.getenv("DEVICE", "cpu")
MAX_FILE_MB = 10


# ── 生命周期 ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 GeoAI 服务启动中...")
    predictor = get_predictor(MODEL_PATH, DEVICE)
    try:
        predictor.load()
        print("✅ 模型加载成功")
    except FileNotFoundError as e:
        print(f"⚠  模型未找到，请先训练: {e}")
    yield
    print("🛑 GeoAI 服务关闭")


# ── FastAPI 应用 ─────────────────────────────────────────────────────────
app = FastAPI(
    title="GeoAI 遥感图像分类 API",
    description="基于深度学习的遥感卫星图像地物类别分类服务",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载前端静态文件
frontend_dir = ROOT / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


# ── Pydantic 模型 ────────────────────────────────────────────────────────
class PredictionResult(BaseModel):
    class_name:    str
    class_name_zh: str
    class_id:      int
    confidence:    float
    description:   str
    top5:          list[dict]
    infer_time_ms: float

class HealthResponse(BaseModel):
    status:      str
    loaded:      bool
    model_name:  str
    device:      str
    num_classes: int
    image_size:  int
    load_time_s: float


# ── 工具函数 ─────────────────────────────────────────────────────────────
def validate_image(file: UploadFile):
    allowed = {"image/jpeg","image/png","image/tiff","image/tif","image/webp"}
    if file.content_type and file.content_type not in allowed:
        raise HTTPException(400, f"不支持的文件类型: {file.content_type}")


# ── 路由 ─────────────────────────────────────────────────────────────────
@app.get("/", response_class=FileResponse, include_in_schema=False)
async def serve_index():
    index = ROOT / "frontend" / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"message": "GeoAI API 运行中，请访问 /docs"})


@app.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """服务健康检查"""
    predictor = get_predictor(MODEL_PATH, DEVICE)
    h = predictor.health()
    return {**h, "status": "ok" if h["loaded"] else "model_not_loaded"}


@app.get("/classes", tags=["信息"])
async def get_classes():
    """获取所有支持的地物类别"""
    return {
        "num_classes": len(CLASS_NAMES),
        "classes": [
            {"id": i, "name": CLASS_NAMES[i],
             "name_zh": CLASS_NAMES_ZH[i] if i < len(CLASS_NAMES_ZH) else ""}
            for i in range(len(CLASS_NAMES))
        ]
    }


@app.post("/predict", response_model=PredictionResult, tags=["推理"])
async def predict_image(file: UploadFile = File(..., description="遥感图像文件")):
    """
    上传遥感图像，返回地物分类结果
    - 支持格式: JPG / PNG / TIFF / WebP
    - 最大文件: 10 MB
    """
    validate_image(file)
    content = await file.read()
    if len(content) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(413, f"文件过大，最大支持 {MAX_FILE_MB} MB")

    predictor = get_predictor(MODEL_PATH, DEVICE)
    if not predictor._loaded:
        raise HTTPException(503, "模型未加载，请先训练: python scripts/train.py")

    try:
        result = predictor.predict(content)
    except Exception as e:
        raise HTTPException(500, f"推理失败: {str(e)}")

    return result


@app.post("/predict/base64", response_model=PredictionResult, tags=["推理"])
async def predict_base64(payload: dict):
    """
    通过 Base64 编码图像进行分类推理
    请求体: {"image": "<base64_string>"}
    """
    b64 = payload.get("image", "")
    if not b64:
        raise HTTPException(400, "缺少 image 字段")
    try:
        # 去掉 data URL 前缀
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        content = base64.b64decode(b64)
    except Exception:
        raise HTTPException(400, "Base64 解码失败")

    predictor = get_predictor(MODEL_PATH, DEVICE)
    if not predictor._loaded:
        raise HTTPException(503, "模型未加载，请先训练: python scripts/train.py")

    try:
        result = predictor.predict(content)
    except Exception as e:
        raise HTTPException(500, f"推理失败: {str(e)}")
    return result


@app.post("/predict/batch", tags=["推理"])
async def predict_batch(files: list[UploadFile] = File(...)):
    """批量图像分类（最多 16 张）"""
    if len(files) > 16:
        raise HTTPException(400, "批量最多支持 16 张图像")
    predictor = get_predictor(MODEL_PATH, DEVICE)
    if not predictor._loaded:
        raise HTTPException(503, "模型未加载")
    results = []
    for f in files:
        content = await f.read()
        try:
            r = predictor.predict(content)
            r["filename"] = f.filename
            results.append({"success": True, "filename": f.filename, "result": r})
        except Exception as e:
            results.append({"success": False, "filename": f.filename, "error": str(e)})
    return {"count": len(results), "results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app",
                host=os.getenv("API_HOST", "0.0.0.0"),
                port=int(os.getenv("API_PORT", 8000)),
                reload=True,
                app_dir=str(ROOT / "backend"))
