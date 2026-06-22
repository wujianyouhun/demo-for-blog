"""ChangeDetection 后端"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from config import API_HOST, API_PORT, CORS_ORIGINS, OUTPUT_DIR, DATA_DIR
from backend.routers import data, detection, compare

app = FastAPI(title="ChangeDetection API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])
app.include_router(data.router, prefix="/api/data", tags=["数据管理"])
app.include_router(detection.router, prefix="/api/detect", tags=["变化检测"])
app.include_router(compare.router, prefix="/api/compare", tags=["对比分析"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")


@app.get("/")
def root():
    return {"name": "ChangeDetection API", "version": "1.0.0", "docs": "/docs"}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/config")
def get_config():
    from config import PRESET_REGIONS, MODEL_CONFIG, TRAIN_CONFIG, INFERENCE_CONFIG, COMPARE_CONFIG
    from cdd.geoai_change import list_geoai_changestar_models
    return {"regions": PRESET_REGIONS, "models": list(MODEL_CONFIG.keys()),
            "geoai_models": list(list_geoai_changestar_models().keys()),
            "train": TRAIN_CONFIG, "inference": INFERENCE_CONFIG, "compare": COMPARE_CONFIG}


@app.get("/api/files")
def list_files():
    result = {}
    for sub in ["raw/time_a", "raw/time_b", "samples", "models", "output"]:
        d = DATA_DIR / sub
        result[sub] = [{"name": f.name, "path": str(f.relative_to(DATA_DIR)),
                         "size": f.stat().st_size, "suffix": f.suffix}
                        for f in sorted(d.rglob("*")) if f.is_file() and not f.name.startswith(".")] if d.exists() else []
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=API_HOST, port=API_PORT, reload=True)
