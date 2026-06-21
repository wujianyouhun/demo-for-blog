"""数据管理路由"""
import uuid
import traceback
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List
from config import PRESET_REGIONS, RAW_DIR, STAC_API_URL

router = APIRouter()

# 后台任务状态跟踪
_tasks = {}


class DownloadRequest(BaseModel):
    region: Optional[str] = None
    bbox: Optional[List[float]] = None
    date_start: str = "2023-01-01"
    date_end: str = "2023-12-31"
    max_cloud_cover: float = 20.0


def _run_download(task_id: str, req: DownloadRequest):
    """后台执行数据下载"""
    try:
        _tasks[task_id]["status"] = "running"
        _tasks[task_id]["progress"] = 0

        from geoai_core import DataDownloader

        # 确定 bbox
        if req.region and req.region in PRESET_REGIONS:
            bbox = PRESET_REGIONS[req.region]["bbox"]
        elif req.bbox and len(req.bbox) == 4:
            bbox = req.bbox
        else:
            raise ValueError("必须提供有效的 region 或 bbox")

        downloader = DataDownloader(
            stac_url=STAC_API_URL,
            output_dir=str(RAW_DIR),
        )

        result = downloader.download(
            bbox=bbox,
            date_range=(req.date_start, req.date_end),
            max_cloud_cover=req.max_cloud_cover,
            progress_callback=lambda p: _tasks[task_id].update({"progress": p}),
        )

        _tasks[task_id]["status"] = "completed"
        _tasks[task_id]["progress"] = 100
        _tasks[task_id]["result"] = {
            "files": result.get("files", []),
            "bbox": bbox,
            "message": f"下载完成，共 {len(result.get('files', []))} 个文件",
        }
    except Exception as e:
        _tasks[task_id]["status"] = "failed"
        _tasks[task_id]["error"] = str(e)
        _tasks[task_id]["traceback"] = traceback.format_exc()


@router.post("/download")
def download_data(req: DownloadRequest, background_tasks: BackgroundTasks):
    """提交 Sentinel-2 数据下载任务"""
    task_id = str(uuid.uuid4())[:8]
    _tasks[task_id] = {"status": "pending", "progress": 0, "type": "download"}
    background_tasks.add_task(_run_download, task_id, req)
    return {"task_id": task_id, "status": "pending"}


@router.get("/download/{task_id}")
def download_status(task_id: str):
    """查询下载任务状态"""
    if task_id not in _tasks:
        return {"error": "任务不存在", "task_id": task_id}
    return _tasks[task_id]


@router.get("/files")
def list_downloaded_files():
    """列出已下载的原始数据文件"""
    files = []
    if RAW_DIR.exists():
        for f in sorted(RAW_DIR.rglob("*")):
            if f.is_file() and not f.name.startswith("."):
                files.append({
                    "name": f.name,
                    "path": str(f.relative_to(RAW_DIR)),
                    "size": f.stat().st_size,
                    "suffix": f.suffix,
                })
    return {"files": files, "directory": str(RAW_DIR)}


@router.get("/regions")
def list_regions():
    """列出预设区域"""
    return {"regions": PRESET_REGIONS}
