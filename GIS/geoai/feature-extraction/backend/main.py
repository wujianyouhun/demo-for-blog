"""
GeoAI 要素提取 —— FastAPI 后端
================================
基于 geoai-py 库 + 传统 CV 的遥感影像要素提取服务

7 个 API 端点:
  1. GET  /api/tif-info             TIF 元数据
  2. GET  /api/tiles/{z}/{x}/{y}    瓦片服务
  3. POST /api/generate-tiles       预生成瓦片
  4. POST /api/extract              提取要素 (ROI/全图)  ← 支持 geoai/cv/dl/hybrid 四种方法
  5. GET  /api/result               获取提取结果 GeoJSON
  6. GET  /api/stats                获取统计信息
  7. GET  /api/export/{fmt}         导出结果

核心依赖: geoai-py (BuildingFootprintExtractor, GroundedSAM)
"""

import os
import json
import asyncio
from typing import Optional, List
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
from pydantic import BaseModel
import uvicorn

from tiles import TIFTileServer
from extractor import GeoAIExtractor, ExtractionConfig
from geoai_extractor import GeoAILibraryExtractor, GeoAIConfig

app = FastAPI(title="GeoAI 要素提取", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 路径配置 ──
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
TIF_PATH = os.environ.get(
    "TIF_PATH", os.path.join(DATA_DIR, "sample.tif")
)
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
TILES_DIR = os.path.join(DATA_DIR, "tiles")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TILES_DIR, exist_ok=True)

# ── 全局状态 ──
_tile_server: Optional[TIFTileServer] = None
_extractor: Optional[GeoAIExtractor] = None
_geoai_extractor: Optional[GeoAILibraryExtractor] = None
_extraction_result = {}  # 存储最新一次提取结果


def _get_tile_server() -> TIFTileServer:
    global _tile_server
    if _tile_server is None:
        if not os.path.exists(TIF_PATH):
            raise HTTPException(404, f"TIF 文件不存在: {TIF_PATH}")
        _tile_server = TIFTileServer(TIF_PATH, cache_dir=TILES_DIR)
    return _tile_server


def _get_extractor() -> GeoAIExtractor:
    global _extractor
    if _extractor is None:
        if not os.path.exists(TIF_PATH):
            raise HTTPException(404, f"TIF 文件不存在: {TIF_PATH}")
        _extractor = GeoAIExtractor(TIF_PATH)
    return _extractor


def _get_geoai_extractor() -> GeoAILibraryExtractor:
    global _geoai_extractor
    if _geoai_extractor is None:
        if not os.path.exists(TIF_PATH):
            raise HTTPException(404, f"TIF 文件不存在: {TIF_PATH}")
        _geoai_extractor = GeoAILibraryExtractor(TIF_PATH)
    return _geoai_extractor


# ════════════════════════════════════════════════
#  API 端点
# ════════════════════════════════════════════════

# 1. TIF 元数据
@app.get("/api/tif-info")
def tif_info():
    ts = _get_tile_server()
    info = ts.get_info()
    info["file_name"] = os.path.basename(TIF_PATH)
    info["file_size_mb"] = round(os.path.getsize(TIF_PATH) / 1024 / 1024, 1)
    return info


# 2. 瓦片服务
@app.get("/api/tiles/{z}/{x}/{y}")
def get_tile(z: int, x: int, y: int):
    ts = _get_tile_server()
    data = ts.get_tile(z, x, y)
    if data is None:
        raise HTTPException(404, "Tile not found")
    return Response(content=data, media_type="image/png")


# 3. 预生成瓦片
@app.post("/api/generate-tiles")
def generate_tiles(zoom_levels: Optional[List[int]] = None):
    ts = _get_tile_server()
    if zoom_levels is None:
        max_z = ts.max_zoom
        zoom_levels = list(range(max(0, max_z - 3), max_z + 1))

    total = ts.generate_tiles(zoom_levels)
    return {"generated": total, "zoom_levels": zoom_levels}


# 4. 提取要素
class ExtractRequest(BaseModel):
    # 提取类别
    targets: List[str] = ["building"]
    # 提取方法: geoai / cv / dl / hybrid
    method: str = "geoai"
    # ROI 范围 (经纬度), None 表示全图
    roi: Optional[dict] = None  # {left, bottom, right, top}
    # geoai 库相关参数
    geoai_device: str = "cpu"
    geoai_building_regularize: bool = True
    geoai_building_batch_size: int = 4
    geoai_groundedsam_threshold: float = 0.3
    # CV 参数
    building_canny_low: int = 50
    building_canny_high: int = 150
    building_min_area_px: int = 100
    building_kernel_size: int = 5
    forest_hsv_h_range: list = [35, 85]
    forest_hsv_s_min: int = 40
    forest_hsv_v_min: int = 40
    forest_min_area_px: int = 200
    grassland_hsv_h_range: list = [25, 75]
    grassland_hsv_s_range: list = [20, 120]
    grassland_hsv_v_range: list = [60, 200]
    grassland_min_area_px: int = 300
    # 通用
    simplify_tolerance: float = 1.0
    min_polygon_area: float = 10.0


@app.post("/api/extract")
def extract_features(req: ExtractRequest):
    global _extraction_result

    # 根据方法选择提取器
    if req.method == "geoai":
        return _extract_with_geoai(req)
    else:
        return _extract_with_cv_dl(req)


def _extract_with_geoai(req: ExtractRequest):
    """使用 geoai-py 库进行提取"""
    global _extraction_result

    extractor = _get_geoai_extractor()

    # 构建配置
    config = GeoAIConfig(
        targets=req.targets,
        device=req.geoai_device,
        building_regularize=req.geoai_building_regularize,
        building_batch_size=req.geoai_building_batch_size,
        grounded_sam_threshold=req.geoai_groundedsam_threshold,
        min_polygon_area=req.min_polygon_area,
        simplify_tolerance=req.simplify_tolerance,
    )
    extractor.config = config

    # 执行提取
    results = extractor.extract(bounds_latlon=req.roi)

    # 存储结果
    geojson_by_target = {}
    for target, gdf in results.items():
        if len(gdf) > 0:
            geojson_by_target[target] = json.loads(
                gdf.to_crs("EPSG:4326").to_json()
            )
        else:
            geojson_by_target[target] = {"type": "FeatureCollection", "features": []}

    _extraction_result = {
        "results": geojson_by_target,
        "stats": extractor.stats,
        "targets": req.targets,
        "method": req.method,
    }

    return _extraction_result


def _extract_with_cv_dl(req: ExtractRequest):
    """使用传统 CV / DL / 混合方法提取"""
    global _extraction_result

    extractor = _get_extractor()

    # 构建配置
    config = ExtractionConfig(
        targets=req.targets,
        method=req.method,
        building_canny_low=req.building_canny_low,
        building_canny_high=req.building_canny_high,
        building_min_area_px=req.building_min_area_px,
        building_kernel_size=req.building_kernel_size,
        forest_hsv_h_range=tuple(req.forest_hsv_h_range),
        forest_hsv_s_min=req.forest_hsv_s_min,
        forest_hsv_v_min=req.forest_hsv_v_min,
        forest_min_area_px=req.forest_min_area_px,
        grassland_hsv_h_range=tuple(req.grassland_hsv_h_range),
        grassland_hsv_s_range=tuple(req.grassland_hsv_s_range),
        grassland_hsv_v_range=tuple(req.grassland_hsv_v_range),
        grassland_min_area_px=req.grassland_min_area_px,
        simplify_tolerance=req.simplify_tolerance,
        min_polygon_area=req.min_polygon_area,
    )
    extractor.config = config

    # 确定提取范围
    if req.roi:
        results = extractor.extract_region(req.roi)
    else:
        # 全图: 使用 TIF 的 bounds
        full_bounds = {
            "left": extractor.bounds.left,
            "bottom": extractor.bounds.bottom,
            "right": extractor.bounds.right,
            "top": extractor.bounds.top,
        }
        results = extractor.extract_region(full_bounds)

    # 存储结果
    geojson_by_target = {}
    for target, gdf in results.items():
        if len(gdf) > 0:
            geojson_by_target[target] = json.loads(
                gdf.to_crs("EPSG:4326").to_json()
            )
        else:
            geojson_by_target[target] = {"type": "FeatureCollection", "features": []}

    _extraction_result = {
        "results": geojson_by_target,
        "stats": extractor.stats,
        "targets": req.targets,
        "method": req.method,
    }

    return _extraction_result


# 5. 获取提取结果
@app.get("/api/result")
def get_result(target: Optional[str] = None):
    if not _extraction_result:
        return {"results": {}, "stats": {}}
    if target:
        return {
            "geojson": _extraction_result["results"].get(target, {}),
            "stats": _extraction_result["stats"].get("targets", {}).get(target, {}),
        }
    return _extraction_result


# 6. 获取统计信息
@app.get("/api/stats")
def get_stats():
    if not _extraction_result:
        return {}
    return _extraction_result.get("stats", {})


# 7. 导出结果
@app.get("/api/export/{fmt}")
def export_result(fmt: str, target: Optional[str] = None):
    if not _extraction_result:
        raise HTTPException(400, "没有可导出的结果，请先执行提取")

    method = _extraction_result.get("method", "cv")

    # 根据方法获取对应的提取器
    if method == "geoai":
        extractor = _get_geoai_extractor()
    else:
        extractor = _get_extractor()

    results = extractor.results

    if not results:
        raise HTTPException(400, "提取结果为空")

    # 选择要导出的目标
    if target and target in results:
        gdfs = {target: results[target]}
    else:
        gdfs = results

    import geopandas as gpd
    import pandas as pd

    all_gdfs = [gdf for gdf in gdfs.values() if len(gdf) > 0]
    if not all_gdfs:
        raise HTTPException(400, "没有有效的提取结果")

    merged = gpd.GeoDataFrame(
        pd.concat(all_gdfs, ignore_index=True), crs=extractor.crs
    )

    import tempfile
    tmp_dir = tempfile.mkdtemp()

    if fmt == "geojson":
        path = os.path.join(tmp_dir, "extraction.geojson")
        merged.to_crs("EPSG:4326").to_file(path, driver="GeoJSON")
        return FileResponse(path, filename="extraction.geojson",
                            media_type="application/geo+json")
    elif fmt == "gpkg":
        path = os.path.join(tmp_dir, "extraction.gpkg")
        merged.to_file(path, driver="GPKG")
        return FileResponse(path, filename="extraction.gpkg",
                            media_type="application/octet-stream")
    elif fmt == "shp":
        import shutil
        shp_dir = os.path.join(tmp_dir, "shp")
        os.makedirs(shp_dir)
        merged.to_file(os.path.join(shp_dir, "extraction.shp"),
                        driver="ESRI Shapefile")
        zip_path = os.path.join(tmp_dir, "extraction.zip")
        shutil.make_archive(os.path.join(tmp_dir, "extraction"), "zip", shp_dir)
        return FileResponse(zip_path, filename="extraction.zip",
                            media_type="application/zip")
    else:
        raise HTTPException(400, f"不支持的格式: {fmt}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
