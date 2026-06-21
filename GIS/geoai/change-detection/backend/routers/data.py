"""数据管理 API"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional, List
from config import RAW_DIR, PRESET_REGIONS

router = APIRouter()
_tasks = {}


class DownloadPairRequest(BaseModel):
    region: Optional[str] = None
    bbox: Optional[List[float]] = None
    date_a: str = "2022-06-01"
    date_b: str = "2023-06-01"
    max_cloud_cover: float = 20.0
    out_name: Optional[str] = None


@router.post("/download-pair")
async def download_pair(req: DownloadPairRequest, background_tasks: BackgroundTasks):
    import uuid
    task_id = str(uuid.uuid4())[:8]
    _tasks[task_id] = {
        "status": "running", "message": "准备中...",
        "progress": 0, "stage": "init", "result": None,
    }
    bbox = req.bbox or (PRESET_REGIONS.get(req.region, {}).get("bbox") if req.region else None)
    if not bbox:
        raise HTTPException(400, "请提供 bbox 或 region")

    def _update(progress: int, stage: str, message: str):
        _tasks[task_id].update(progress=progress, stage=stage, message=message)

    def _run():
        try:
            _update(5, "searching", "正在搜索可用影像...")
            from cdd.downloader import BiTemporalDownloader
            dl = BiTemporalDownloader(RAW_DIR)

            bands = ["B04", "B03", "B02"]
            prefix = req.out_name or f"s2_{bbox[0]:.2f}_{bbox[1]:.2f}"

            _update(10, "downloading_a", "正在下载时相 A...")
            path_a = dl._download_single(
                bbox, req.date_a, req.max_cloud_cover,
                bands, dl.time_a_dir, f"{prefix}_A",
            )

            _update(50, "downloading_b", "正在下载时相 B...")
            path_b = dl._download_single(
                bbox, req.date_b, req.max_cloud_cover,
                bands, dl.time_b_dir, f"{prefix}_B",
            )

            if path_a is None or path_b is None:
                raise RuntimeError("未能下载到足够的影像")

            _update(85, "aligning", "正在对齐空间参考...")
            aligned_b = dl._align_pair(path_a, path_b)

            _update(100, "done", "下载完成")
            _tasks[task_id]["status"] = "completed"
            _tasks[task_id]["result"] = {
                "time_a": str(path_a),
                "time_b": str(aligned_b),
            }
        except Exception as e:
            _tasks[task_id].update(
                status="failed", message=str(e), result=None, progress=0, stage="error",
            )

    background_tasks.add_task(_run)
    return {"task_id": task_id, "status": "running", "message": "下载任务已启动"}


@router.get("/download-pair/{task_id}")
async def download_status(task_id: str):
    if task_id not in _tasks:
        raise HTTPException(404)
    return {"task_id": task_id, **_tasks[task_id]}


@router.get("/pairs")
async def list_pairs():
    from cdd.downloader import BiTemporalDownloader
    return {"pairs": BiTemporalDownloader(RAW_DIR).list_pairs()}


@router.get("/regions")
async def list_regions():
    return PRESET_REGIONS


@router.get("/preview/{subdir}/{filename}")
async def preview_image(subdir: str, filename: str):
    """将 GeoTIFF 转为 PNG 预览（RGB 三通道）"""
    import numpy as np
    from PIL import Image
    import io
    import rasterio

    filepath = RAW_DIR / subdir / filename
    if not filepath.exists():
        raise HTTPException(404, "文件不存在")

    with rasterio.open(filepath) as src:
        bands = []
        for i in range(min(3, src.count)):
            band = src.read(i + 1).astype(np.float32)
            # 2% 线性拉伸，提升可视化效果
            lo = np.percentile(band[band > 0], 2) if np.any(band > 0) else 0
            hi = np.percentile(band, 98) if np.any(band > 0) else 1
            if hi == lo:
                hi = lo + 1
            band = np.clip((band - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
            bands.append(band)
        while len(bands) < 3:
            bands.append(bands[-1])
        rgb = np.stack(bands, axis=-1)
        # 获取空间信息供前端使用
        bounds = src.bounds
        crs = str(src.crs) if src.crs else "EPSG:4326"

    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    return Response(
        content=buf.read(),
        media_type="image/png",
        headers={
            "X-Image-Bounds": f"{bounds.left},{bounds.bottom},{bounds.right},{bounds.top}",
            "X-Image-CRS": crs,
        },
    )
