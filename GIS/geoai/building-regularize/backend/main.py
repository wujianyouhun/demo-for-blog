"""
建筑正则化 FastAPI 后端服务。

提供：
  - 加载演示数据
  - 上传 GeoJSON
  - 逐步执行正则化流水线并返回中间结果
  - 导出 GeoJSON / Shapefile / GeoPackage
"""

import io
import json
import os
import tempfile
from typing import Any, Dict, List, Optional

import geopandas as gpd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from shapely.geometry import Polygon, mapping, shape

from .regularize import RegularizeConfig, RegularizePipeline

# ──────────────────────────────────────────────
# App
# ──────────────────────────────────────────────
app = FastAPI(title="建筑物轮廓正则化", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# 存放当前会话的数据（单用户简化版）
_session: Dict[str, Any] = {
    "raw": [],           # List[Polygon]
    "name": "",
    "steps": {},
}


# ──────────────────────────────────────────────
# 请求/响应模型
# ──────────────────────────────────────────────
class ConfigModel(BaseModel):
    min_area: float = 20.0
    dp_tolerance: float = 0.5
    use_pca: bool = False
    angle_threshold: float = 10.0
    snap_angles: List[float] = [0, 45, 90, 135]
    enable_symmetry: bool = False
    symmetry_tolerance: float = 2.0
    smooth_iterations: int = 0
    smooth_ratio: float = 0.25


class RunRequest(BaseModel):
    config: ConfigModel = Field(default_factory=ConfigModel)


class StepInfo(BaseModel):
    key: str
    label: str
    count: int
    geojson: dict


# ──────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────
STEP_LABELS = {
    "0_topology_fix": "拓扑修复",
    "1_area_filter": "面积过滤",
    "2_dp_simplify": "Douglas-Peucker 简化",
    "3_direction_detect": "主方向检测",
    "4_orthogonalize": "直角化",
    "5_symmetry": "对称化",
    "6_smooth": "Chaikin 平滑",
    "7_final": "最终结果",
}


def _polys_to_geojson(polys: List[Polygon]) -> dict:
    features = []
    for i, p in enumerate(polys):
        features.append({
            "type": "Feature",
            "id": i,
            "geometry": mapping(p),
            "properties": {
                "area": round(p.area, 2),
                "vertex_count": len(p.exterior.coords) - 1,
            },
        })
    return {"type": "FeatureCollection", "features": features}


def _load_geojson(geojson: dict) -> List[Polygon]:
    polys = []
    features = geojson.get("features", [])
    for f in features:
        geom = shape(f["geometry"])
        if isinstance(geom, Polygon):
            polys.append(geom)
        elif hasattr(geom, "geoms"):
            for g in geom.geoms:
                if isinstance(g, Polygon):
                    polys.append(g)
    return polys


# ──────────────────────────────────────────────
# API 路由
# ──────────────────────────────────────────────
@app.get("/api/demo-list")
async def list_demos():
    """列出所有演示数据集。"""
    demos = []
    if os.path.isdir(DATA_DIR):
        for fn in sorted(os.listdir(DATA_DIR)):
            if fn.endswith(".geojson"):
                demos.append(fn.replace(".geojson", ""))
    return {"demos": demos}


@app.get("/api/demo/{name}")
async def load_demo(name: str):
    """加载指定演示数据。"""
    path = os.path.join(DATA_DIR, f"{name}.geojson")
    if not os.path.isfile(path):
        raise HTTPException(404, f"演示数据 '{name}' 不存在")

    with open(path, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    polys = _load_geojson(geojson)
    _session["raw"] = polys
    _session["name"] = name
    _session["steps"] = {}

    return {
        "name": name,
        "count": len(polys),
        "geojson": _polys_to_geojson(polys),
    }


@app.post("/api/upload")
async def upload_geojson(file: UploadFile = File(...)):
    """上传 GeoJSON 文件。"""
    content = await file.read()
    try:
        geojson = json.loads(content)
    except json.JSONDecodeError:
        raise HTTPException(400, "无效的 JSON 文件")

    polys = _load_geojson(geojson)
    if not polys:
        raise HTTPException(400, "未找到有效的 Polygon 要素")

    name = os.path.splitext(file.filename)[0]
    _session["raw"] = polys
    _session["name"] = name
    _session["steps"] = {}

    return {
        "name": name,
        "count": len(polys),
        "geojson": _polys_to_geojson(polys),
    }


@app.post("/api/run")
async def run_pipeline(req: RunRequest):
    """执行正则化流水线，返回所有步骤的中间结果。"""
    if not _session["raw"]:
        raise HTTPException(400, "请先加载或上传数据")

    cfg = RegularizeConfig(
        min_area=req.config.min_area,
        dp_tolerance=req.config.dp_tolerance,
        use_pca=req.config.use_pca,
        angle_threshold=req.config.angle_threshold,
        snap_angles=req.config.snap_angles,
        enable_symmetry=req.config.enable_symmetry,
        symmetry_tolerance=req.config.symmetry_tolerance,
        smooth_iterations=req.config.smooth_iterations,
        smooth_ratio=req.config.smooth_ratio,
    )

    pipeline = RegularizePipeline(cfg)
    pipeline.run(list(_session["raw"]))

    _session["steps"] = pipeline.steps

    # 构建步骤响应
    steps_resp = []
    for key, polys in pipeline.steps.items():
        steps_resp.append(StepInfo(
            key=key,
            label=STEP_LABELS.get(key, key),
            count=len(polys),
            geojson=_polys_to_geojson(polys),
        ))

    # 计算主方向
    directions = []
    for p in pipeline.steps.get("3_direction_detect", _session["raw"]):
        from regularize import get_main_direction
        d = get_main_direction(p, cfg.use_pca)
        directions.append(round(d, 1))

    return {
        "steps": [s.dict() for s in steps_resp],
        "directions": directions,
        "raw_count": len(_session["raw"]),
    }


@app.get("/api/step/{key}")
async def get_step(key: str):
    """获取指定步骤的结果 GeoJSON。"""
    polys = _session["steps"].get(key)
    if polys is None:
        raise HTTPException(404, f"步骤 '{key}' 不存在，请先运行流水线")
    return _polys_to_geojson(polys)


@app.get("/api/export/{fmt}")
async def export_result(fmt: str = "geojson"):
    """导出最终结果为 GeoJSON / Shapefile / GeoPackage。"""
    final = _session["steps"].get("7_final")
    if final is None:
        raise HTTPException(400, "请先运行流水线")

    gdf = gpd.GeoDataFrame(
        {"area": [round(p.area, 2) for p in final],
         "vertices": [len(p.exterior.coords) - 1 for p in final]},
        geometry=list(final),
        crs="EPSG:4326",
    )

    if fmt == "geojson":
        content = gdf.to_json()
        return StreamingResponse(
            io.BytesIO(content.encode()),
            media_type="application/geo+json",
            headers={"Content-Disposition": "attachment; filename=regularized.geojson"},
        )
    elif fmt == "gpkg":
        buf = io.BytesIO()
        gdf.to_file(buf, driver="GPKG")
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/geopackage+sqlite3",
            headers={"Content-Disposition": "attachment; filename=regularized.gpkg"},
        )
    elif fmt == "shp":
        with tempfile.TemporaryDirectory() as tmpdir:
            shp_path = os.path.join(tmpdir, "regularized.shp")
            gdf.to_file(shp_path, driver="ESRI Shapefile")
            import zipfile
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w") as zf:
                for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
                    fp = shp_path.replace(".shp", ext)
                    if os.path.isfile(fp):
                        zf.write(fp, f"regularized{ext}")
            zip_buf.seek(0)
            return StreamingResponse(
                zip_buf,
                media_type="application/zip",
                headers={"Content-Disposition": "attachment; filename=regularized.zip"},
            )
    else:
        raise HTTPException(400, f"不支持的格式: {fmt}")


@app.get("/api/compare")
async def compare_stats():
    """返回正则化前后的统计对比。"""
    raw = _session["raw"]
    final = _session["steps"].get("7_final", [])

    def _stats(polys):
        if not polys:
            return {"count": 0, "total_area": 0, "avg_vertices": 0}
        total_area = sum(p.area for p in polys)
        total_v = sum(len(p.exterior.coords) - 1 for p in polys)
        return {
            "count": len(polys),
            "total_area": round(total_area, 2),
            "avg_vertices": round(total_v / len(polys), 1),
        }

    return {
        "before": _stats(raw),
        "after": _stats(final),
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "project": "building-regularize"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
