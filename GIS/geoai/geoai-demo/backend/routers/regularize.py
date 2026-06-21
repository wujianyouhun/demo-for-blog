"""要素正则化路由"""
import json
from pathlib import Path
from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import Optional
from config import OUTPUT_DIR, REGULARIZE_CONFIG

router = APIRouter()


class RegularizeRequest(BaseModel):
    input_path: str
    simplify_tolerance: float = REGULARIZE_CONFIG["simplify_tolerance"]
    smooth_iterations: int = REGULARIZE_CONFIG["smooth_iterations"]
    min_area: float = REGULARIZE_CONFIG["min_area"]
    orthogonalize: bool = REGULARIZE_CONFIG["orthogonalize"]


@router.post("/run")
def run_regularize(req: RegularizeRequest):
    """对矢量结果执行正则化（简化、平滑、正交化）"""
    try:
        from geoai_core import FeatureRegularizer

        regularizer = FeatureRegularizer(
            simplify_tolerance=req.simplify_tolerance,
            smooth_iterations=req.smooth_iterations,
            min_area=req.min_area,
            orthogonalize=req.orthogonalize,
        )

        result = regularizer.process(input_path=req.input_path)

        # 保存结果
        input_name = Path(req.input_path).stem
        output_path = OUTPUT_DIR / f"{input_name}_regularized.geojson"
        regularizer.save(result, str(output_path))

        return {
            "status": "completed",
            "output_path": str(output_path),
            "filename": output_path.name,
            "stats": result.get("stats", {}),
            "message": f"正则化完成，结果保存至 {output_path.name}",
        }
    except Exception as e:
        return {"status": "failed", "error": str(e)}


@router.get("/preview")
def preview_regularized(
    file_path: str = Query(..., description="GeoJSON 文件路径"),
):
    """预览正则化后的 GeoJSON 数据"""
    try:
        p = Path(file_path)
        if not p.exists():
            # 尝试在 OUTPUT_DIR 中查找
            p = OUTPUT_DIR / file_path
        if not p.exists():
            return {"error": f"文件不存在: {file_path}"}

        with open(p, "r", encoding="utf-8") as f:
            geojson = json.load(f)

        # 限制返回的要素数量
        features = geojson.get("features", [])
        preview_count = min(len(features), 100)

        return {
            "type": geojson.get("type", "FeatureCollection"),
            "total_features": len(features),
            "preview_features": preview_count,
            "features": features[:preview_count],
            "crs": geojson.get("crs", "EPSG:4326"),
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/stats")
def get_stats(
    file_path: str = Query(..., description="GeoJSON 文件路径"),
):
    """获取正则化结果的统计信息"""
    try:
        p = Path(file_path)
        if not p.exists():
            p = OUTPUT_DIR / file_path
        if not p.exists():
            return {"error": f"文件不存在: {file_path}"}

        with open(p, "r", encoding="utf-8") as f:
            geojson = json.load(f)

        features = geojson.get("features", [])
        if not features:
            return {"polygon_count": 0, "message": "无要素"}

        # 统计各类别数量和面积
        class_counts = {}
        areas = []
        for feat in features:
            props = feat.get("properties", {})
            cls = props.get("class_name", props.get("class", "unknown"))
            class_counts[cls] = class_counts.get(cls, 0) + 1

            geom = feat.get("geometry", {})
            area = props.get("area", 0)
            if area > 0:
                areas.append(area)

        import numpy as np
        area_stats = {}
        if areas:
            arr = np.array(areas)
            area_stats = {
                "total_area": float(arr.sum()),
                "mean_area": float(arr.mean()),
                "median_area": float(np.median(arr)),
                "min_area": float(arr.min()),
                "max_area": float(arr.max()),
            }

        return {
            "polygon_count": len(features),
            "class_counts": class_counts,
            "area_stats": area_stats,
        }
    except Exception as e:
        return {"error": str(e)}
