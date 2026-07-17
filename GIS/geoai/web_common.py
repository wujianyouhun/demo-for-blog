"""GeoAI 独立 Web 示例共用的轻量任务与文件工具。"""
from __future__ import annotations

import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable


TERMINAL_STATES = {"completed", "failed", "cancelled"}


class TaskRegistry:
    """进程内任务注册表；GPU 类项目使用一个 worker 防止显存争抢。"""

    def __init__(self, max_workers: int = 1):
        self._tasks: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="geoai-task")

    def submit(self, stage: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
        task_id = uuid.uuid4().hex[:12]
        cancel_event = threading.Event()
        task = {
            "task_id": task_id,
            "status": "pending",
            "progress": 0,
            "stage": stage,
            "message": "任务已进入队列",
            "metrics": {},
            "result": None,
            "error": None,
            "_cancel_event": cancel_event,
        }
        with self._lock:
            self._tasks[task_id] = task
        self._executor.submit(self._run, task_id, func, args, kwargs)
        return self.public(task_id)

    def _run(self, task_id: str, func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        self.update(task_id, status="running", message="任务正在执行")
        try:
            result = func(
                *args,
                task_id=task_id,
                cancel_event=self._tasks[task_id]["_cancel_event"],
                update=lambda **values: self.update(task_id, **values),
                **kwargs,
            )
            if self._tasks[task_id]["_cancel_event"].is_set():
                self.update(task_id, status="cancelled", message="任务已取消")
            else:
                self.update(task_id, status="completed", progress=100, message="任务完成", result=result)
        except Exception as exc:  # pragma: no cover - error branch is exposed to UI
            self.update(
                task_id,
                status="failed",
                message="任务失败",
                error={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
            )

    def update(self, task_id: str, **values: Any) -> None:
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].update(values)

    def public(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            if task_id not in self._tasks:
                raise KeyError(task_id)
            return {key: value for key, value in self._tasks[task_id].items() if not key.startswith("_")}

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self.public(task_id) for task_id in reversed(list(self._tasks))]

    def cancel(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            task = self._tasks[task_id]
            if task["status"] not in TERMINAL_STATES:
                task["_cancel_event"].set()
                task["message"] = "正在取消"
        return self.public(task_id)


def resolve_user_path(value: str, base_dir: Path, allowed_suffixes: set[str] | None = None) -> Path:
    """解析用户选择的本地路径并执行存在性和后缀校验。"""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    if allowed_suffixes and path.suffix.lower() not in allowed_suffixes:
        raise ValueError(f"不支持的文件格式: {path.suffix}")
    return path


def list_files(root: Path, suffixes: set[str] | None = None) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    result = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if suffixes and path.suffix.lower() not in suffixes:
            continue
        result.append({
            "name": path.name,
            "path": str(path),
            "relative_path": str(path.relative_to(root)),
            "size": path.stat().st_size,
            "suffix": path.suffix.lower(),
        })
    return result


def install_common_routes(app, registry: TaskRegistry, project_name: str, data_dir: Path,
                          output_dir: Path, config: dict[str, Any] | None = None,
                          upload_suffixes: set[str] | None = None) -> None:
    """为独立 FastAPI 示例安装统一健康检查、任务、上传和下载接口。"""
    from fastapi import File, HTTPException, UploadFile
    from fastapi.responses import FileResponse

    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    upload_dir = data_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    allowed = upload_suffixes or {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".geojson", ".gpkg", ".zip", ".shp"}

    @app.get("/api/health")
    def common_health():
        return {"status": "ok", "project": project_name}

    @app.get("/api/config")
    def common_config():
        return {"project": project_name, "data_dir": str(data_dir), "output_dir": str(output_dir), **(config or {})}

    @app.get("/api/files")
    def common_files():
        return {"files": list_files(data_dir) + list_files(output_dir)}

    @app.post("/api/uploads")
    async def common_upload(file: UploadFile = File(...)):
        name = Path(file.filename or "upload.bin").name
        if Path(name).suffix.lower() not in allowed:
            raise HTTPException(400, f"不支持的格式: {Path(name).suffix}")
        target = upload_dir / name
        target.write_bytes(await file.read())
        return {"name": target.name, "path": str(target), "size": target.stat().st_size}

    @app.get("/api/tasks/{task_id}")
    def common_task(task_id: str):
        try:
            return registry.public(task_id)
        except KeyError:
            raise HTTPException(404, "任务不存在")

    @app.post("/api/tasks/{task_id}/cancel")
    def common_cancel(task_id: str):
        try:
            return registry.cancel(task_id)
        except KeyError:
            raise HTTPException(404, "任务不存在")

    @app.get("/api/download")
    def common_download(path: str):
        candidate = Path(path).expanduser().resolve()
        roots = [data_dir.resolve(), output_dir.resolve()]
        if not candidate.is_file() or not any(candidate == root or root in candidate.parents for root in roots):
            raise HTTPException(404, "文件不存在或不允许下载")
        return FileResponse(candidate, filename=candidate.name)
