"""对比分析 API"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from config import OUTPUT_DIR

router = APIRouter()


class CompareRequest(BaseModel):
    image_a: str
    image_b: str
    change_map: Optional[str] = None
    mode: str = "side_by_side"
    opacity: float = 0.5
    change_color: str = "#FF0000"


class DiffStatsRequest(BaseModel):
    change_map_path: str


@router.post("/visualize")
async def create_visualization(req: CompareRequest):
    try:
        from cdd.visualize import ChangeVisualizer
        color = tuple(int(req.change_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        out = OUTPUT_DIR / f"compare_{req.mode}.png"
        if req.mode == "heatmap":
            ChangeVisualizer.create_difference_heatmap(req.image_a, req.image_b, out)
        else:
            ChangeVisualizer.create_comparison_image(req.image_a, req.image_b, req.change_map, out, req.mode, color, req.opacity)
        return {"status": "ok", "path": str(out), "filename": out.name}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/overlay")
async def create_overlay(req: CompareRequest):
    try:
        from cdd.visualize import ChangeVisualizer
        color = tuple(int(req.change_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        out = OUTPUT_DIR / "change_overlay.png"
        ChangeVisualizer.create_change_overlay(req.image_a, req.change_map, out, color, req.opacity)
        return {"status": "ok", "path": str(out)}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/stats")
async def get_stats(req: DiffStatsRequest):
    try:
        import rasterio, numpy as np
        with rasterio.open(req.change_map_path) as src:
            chg = src.read(1).astype(np.uint8)
            prob = src.read(2) if src.count >= 2 else None
        total = chg.size
        changed = int(np.sum(chg > 0))
        stats = {"total_pixels": total, "changed_pixels": changed,
                  "unchanged_pixels": total - changed, "change_ratio": round(changed / max(total, 1) * 100, 2)}
        if prob is not None and changed > 0:
            stats["mean_probability"] = round(float(np.mean(prob[chg > 0])), 4)
            stats["max_probability"] = round(float(np.max(prob[chg > 0])), 4)
        return stats
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/preview/{filename}")
async def preview_vector(filename: str):
    import geopandas as gpd
    fpath = OUTPUT_DIR / filename
    if not fpath.exists(): raise HTTPException(404)
    gdf = gpd.read_file(fpath)
    if gdf.crs and not gdf.crs.is_geographic: gdf = gdf.to_crs(epsg=4326)
    return {"geojson": gdf.__geo_interface__, "count": len(gdf)}
