"""GeoAI Demo 后端"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from config import API_HOST, API_PORT, CORS_ORIGINS, OUTPUT_DIR, DATA_DIR
from backend.routers import data, train, extract, regularize

app = FastAPI(title="GeoAI Demo API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])
app.include_router(data.router, prefix="/api/data", tags=["数据管理"])
app.include_router(train.router, prefix="/api/train", tags=["模型训练"])
app.include_router(extract.router, prefix="/api/extract", tags=["要素提取"])
app.include_router(regularize.router, prefix="/api/regularize", tags=["要素正则化"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")


@app.get("/")
def root():
    return {"name": "GeoAI Demo API", "version": "1.0.0", "docs": "/docs"}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/config")
def get_config():
    from config import PRESET_REGIONS, MODEL_CONFIG, TRAIN_CONFIG, INFERENCE_CONFIG, REGULARIZE_CONFIG, CLASS_NAMES
    return {
        "regions": PRESET_REGIONS, "models": list(MODEL_CONFIG.keys()),
        "classes": CLASS_NAMES,
        "train": TRAIN_CONFIG, "inference": INFERENCE_CONFIG,
        "regularize": REGULARIZE_CONFIG,
    }


@app.get("/api/files")
def list_files():
    result = {}
    for sub in ["raw", "samples/images", "samples/labels", "models", "output"]:
        d = DATA_DIR / sub
        result[sub] = [
            {"name": f.name, "path": str(f.relative_to(DATA_DIR)), "size": f.stat().st_size, "suffix": f.suffix}
            for f in sorted(d.rglob("*")) if f.is_file() and not f.name.startswith(".")
        ] if d.exists() else []
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=API_HOST, port=API_PORT, reload=True)
