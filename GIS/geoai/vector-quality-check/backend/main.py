"""
矢量数据质量自动检查 —— FastAPI 后端服务
==========================================
8 个 API 端点：
  1. GET  /api/demo-list          获取演示数据列表
  2. GET  /api/demo/{name}        加载演示数据
  3. POST /api/upload             上传 GeoJSON
  4. POST /api/check              执行质量检查
  5. POST /api/repair             执行一键修复
  6. GET  /api/repair-step/{key}  获取修复步骤结果
  7. GET  /api/report             获取检查报告
  8. GET  /api/export/{fmt}       导出修复后数据
"""

import os
import json
import uuid
import tempfile
import shutil
from typing import Optional

import geopandas as gpd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from checker import VectorQualityChecker, VectorAutoRepair, DEFAULT_REPAIR_CONFIG

app = FastAPI(title="矢量数据质量自动检查", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── 会话状态 ──
_session: dict = {}


def _reset_session():
    _session.clear()
    _session.update({
        "raw_gdf": None,
        "raw_geojson": None,
        "checker": None,
        "issues": [],
        "report": {},
        "issues_geojson": None,
        "repair": None,
        "repaired_gdf": None,
        "repair_steps": [],
    })


_reset_session()


# ── 辅助函数 ──
def _gdf_to_geojson(gdf: gpd.GeoDataFrame) -> dict:
    """将 GeoDataFrame 转为 WGS84 GeoJSON (前端显示用)"""
    display_gdf = gdf.to_crs("EPSG:4326") if gdf.crs and not gdf.crs.is_geographic else gdf
    return json.loads(display_gdf.to_json())


def _load_geojson_file(path: str) -> gpd.GeoDataFrame:
    """加载矢量文件并投影到 UTM (方便面积/距离计算)"""
    gdf = gpd.read_file(path)
    if gdf.crs and gdf.crs.is_geographic:
        gdf = gdf.to_crs(gdf.estimate_utm_crs())
    return gdf


def _issues_geojson_to_wgs84(issues_geojson: dict, crs) -> dict:
    """将问题标记的 UTM 几何转为 WGS84 (供前端显示)"""
    if not crs or crs.is_geographic or not issues_geojson.get("features"):
        return issues_geojson
    from pyproj import Transformer
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    def _transform_geom(geom_dict):
        """递归转换 GeoJSON 几何的坐标"""
        geom_type = geom_dict.get("type")
        coords = geom_dict.get("coordinates")
        if not coords:
            return geom_dict

        def _transform_coords(coord_list):
            if isinstance(coord_list[0], (int, float)):
                x, y = transformer.transform(coord_list[0], coord_list[1])
                return [round(x, 8), round(y, 8)]
            return [_transform_coords(c) for c in coord_list]

        return {"type": geom_type, "coordinates": _transform_coords(coords)}

    result = dict(issues_geojson)
    result["features"] = []
    for f in issues_geojson["features"]:
        nf = dict(f)
        if "geometry" in nf and nf["geometry"]:
            nf["geometry"] = _transform_geom(nf["geometry"])
        result["features"].append(nf)
    return result


# ════════════════════════════════════════════════
#  API Endpoints
# ════════════════════════════════════════════════

# 1. 获取演示数据列表
@app.get("/api/demo-list")
def demo_list():
    demos = []
    if os.path.isdir(DATA_DIR):
        for f in sorted(os.listdir(DATA_DIR)):
            if f.endswith(".geojson"):
                demos.append(f.replace(".geojson", ""))
    return {"demos": demos}


# 2. 加载演示数据
@app.get("/api/demo/{name}")
def load_demo(name: str):
    path = os.path.join(DATA_DIR, f"{name}.geojson")
    if not os.path.exists(path):
        raise HTTPException(404, f"演示数据 {name} 不存在")
    _reset_session()
    gdf = _load_geojson_file(path)
    _session["raw_gdf"] = gdf
    _session["raw_geojson"] = _gdf_to_geojson(gdf)
    return {"geojson": _session["raw_geojson"], "count": len(gdf)}


# 3. 上传 GeoJSON
@app.post("/api/upload")
async def upload_geojson(file: UploadFile = File(...)):
    _reset_session()
    suffix = os.path.splitext(file.filename)[1] or ".geojson"
    save_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{suffix}")
    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)
    try:
        gdf = _load_geojson_file(save_path)
    except Exception as e:
        os.remove(save_path)
        raise HTTPException(400, f"无法解析文件: {e}")
    os.remove(save_path)
    _session["raw_gdf"] = gdf
    _session["raw_geojson"] = _gdf_to_geojson(gdf)
    return {"geojson": _session["raw_geojson"], "count": len(gdf)}


# 4. 执行质量检查
class CheckRequest(BaseModel):
    sliver_min_area: float = 5.0
    overlap_threshold: float = 0.05


@app.post("/api/check")
def run_check(req: CheckRequest):
    if _session.get("raw_gdf") is None:
        raise HTTPException(400, "请先加载数据")

    gdf = _session["raw_gdf"]
    checker = VectorQualityChecker(gdf)
    issues = checker.run_all_checks(
        sliver_min_area=req.sliver_min_area,
        overlap_threshold=req.overlap_threshold,
    )
    report = checker.report
    issues_geojson_raw = checker.get_issues_geojson()
    issues_geojson = _issues_geojson_to_wgs84(issues_geojson_raw, gdf.crs)

    _session["checker"] = checker
    _session["issues"] = issues
    _session["report"] = report
    _session["issues_geojson"] = issues_geojson

    return {
        "issues": issues,
        "report": report,
        "issues_geojson": issues_geojson,
    }


# 5. 执行一键修复
class RepairRequest(BaseModel):
    repair_invalid: bool = True
    fill_holes: bool = True
    max_hole_area: Optional[float] = None
    remove_overlaps: bool = True
    overlap_method: str = "area"
    overlap_threshold: float = 0.4
    remove_slivers: bool = True
    min_area: float = 10.0
    explode_multipart: bool = True
    remove_duplicates: bool = True
    simplify: bool = False
    simplify_tolerance: float = 0.5
    preserve_topology: bool = True


@app.post("/api/repair")
def run_repair(req: RepairRequest):
    if _session.get("raw_gdf") is None:
        raise HTTPException(400, "请先加载数据")

    config = req.model_dump()
    gdf = _session["raw_gdf"]

    repairer = VectorAutoRepair(gdf, config=config)
    repaired = repairer.repair_all()

    _session["repair"] = repairer
    _session["repaired_gdf"] = repaired
    _session["repair_steps"] = [
        {"key": k, "label": _step_label(k), "geojson": _gdf_to_geojson(v)}
        for k, v in repairer.step_results.items()
    ]

    return {
        "steps": _session["repair_steps"],
        "repair_log": repairer.get_repair_log(),
        "repaired_geojson": _gdf_to_geojson(repaired),
        "before_count": len(gdf),
        "after_count": len(repaired),
    }


def _step_label(key: str) -> str:
    labels = {
        "repair_invalid": "修复无效几何",
        "fill_holes": "填充孔洞",
        "remove_overlaps": "去除重叠",
        "remove_slivers": "去除碎片",
        "explode_multipart": "拆分多部件",
        "remove_duplicates": "去除重复",
        "simplify": "几何简化",
        "final": "最终验证",
    }
    return labels.get(key, key)


# 6. 获取修复步骤结果
@app.get("/api/repair-step/{key}")
def get_repair_step(key: str):
    repairer = _session.get("repair")
    if repairer is None:
        raise HTTPException(400, "请先执行修复")
    geojson = repairer.get_step_geojson(key)
    if geojson is None:
        raise HTTPException(404, f"步骤 {key} 不存在")
    return {"key": key, "label": _step_label(key), "geojson": geojson}


# 7. 获取检查报告
@app.get("/api/report")
def get_report():
    return {
        "issues": _session.get("issues", []),
        "report": _session.get("report", {}),
        "issues_geojson": _session.get("issues_geojson"),
    }


# 8. 导出修复后数据
@app.get("/api/export/{fmt}")
def export_result(fmt: str):
    gdf = _session.get("repaired_gdf")
    if gdf is None:
        raise HTTPException(400, "请先执行修复")

    tmp_dir = tempfile.mkdtemp()

    try:
        if fmt == "geojson":
            path = os.path.join(tmp_dir, "repaired.geojson")
            gdf.to_file(path, driver="GeoJSON")
            return FileResponse(path, filename="repaired.geojson", media_type="application/geo+json")
        elif fmt == "gpkg":
            path = os.path.join(tmp_dir, "repaired.gpkg")
            gdf.to_file(path, driver="GPKG")
            return FileResponse(path, filename="repaired.gpkg", media_type="application/octet-stream")
        elif fmt == "shp":
            shp_dir = os.path.join(tmp_dir, "shp")
            os.makedirs(shp_dir)
            gdf.to_file(os.path.join(shp_dir, "repaired.shp"), driver="ESRI Shapefile")
            zip_path = os.path.join(tmp_dir, "repaired.zip")
            shutil.make_archive(os.path.join(tmp_dir, "repaired"), "zip", shp_dir)
            return FileResponse(zip_path, filename="repaired.zip", media_type="application/zip")
        else:
            raise HTTPException(400, f"不支持的格式: {fmt}")
    except Exception as e:
        raise HTTPException(500, f"导出失败: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
