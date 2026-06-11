"""
SAM 遥感标注 Web 服务 — FastAPI 入口

提供影像加载、SAM 推理（点/框/文本）、Mask 后处理、矢量导出等 REST API。
配合 Vue3 + OpenLayers 前端使用。

启动方式:
    cd backend
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import uuid
import json
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse, FileResponse
from pydantic import BaseModel

# ── 将项目根目录加入 sys.path ──────────────────────────────
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from services import ImageService, SAMService

# ── FastAPI 应用 ────────────────────────────────────────────
app = FastAPI(
    title="SAM GeoAI 标注平台",
    description="基于 SAM 的遥感影像半自动标注 Web 服务",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 会话存储（单实例简易模式） ──────────────────────────────
_sessions: Dict[str, Dict[str, Any]] = {}


def _get_session(session_id: Optional[str]) -> Dict[str, Any]:
    """获取或创建会话。"""
    if not session_id or session_id not in _sessions:
        sid = str(uuid.uuid4())
        _sessions[sid] = {
            "id": sid,
            "image_service": None,
            "sam_service": None,
            "last_mask": None,
            "processed_mask": None,
            "image_path": None,
        }
        return _sessions[sid]
    return _sessions[session_id]


def _get_image_svc(session: Dict[str, Any]) -> ImageService:
    svc = session.get("image_service")
    if svc is None:
        raise HTTPException(status_code=400, detail="请先加载影像 (/api/image/load)")
    return svc


def _get_sam_svc(session: Dict[str, Any]) -> SAMService:
    svc = session.get("sam_service")
    if svc is None:
        svc = SAMService()
        session["sam_service"] = svc
    return svc


# ================================================================
#  请求/响应模型
# ================================================================

class LoadImageRequest(BaseModel):
    image_path: str
    model_type: str = "vit_l"
    sam_version: str = "sam1"


class PointPredictRequest(BaseModel):
    session_id: str
    points: List[List[float]]   # [[x1,y1], [x2,y2], ...] 显示坐标
    labels: List[int]            # 1=前景, 0=背景


class BoxPredictRequest(BaseModel):
    session_id: str
    box: List[float]             # [x1, y1, x2, y2] 显示坐标


class TextPredictRequest(BaseModel):
    session_id: str
    text: str
    box_threshold: float = 0.25
    text_threshold: float = 0.25


class PostProcessRequest(BaseModel):
    session_id: str
    min_size: int = 200
    fill_holes: bool = True
    smooth_sigma: float = 1.5
    opening_radius: int = 2
    closing_radius: int = 3


class ExportRequest(BaseModel):
    session_id: str
    min_area: int = 50
    output_format: str = "geojson"   # geojson / gpkg / shp


# ================================================================
#  影像管理
# ================================================================

@app.post("/api/image/load")
async def load_image(req: LoadImageRequest):
    """加载影像文件，返回元数据和会话 ID。"""
    if not os.path.isfile(req.image_path):
        raise HTTPException(status_code=404, detail=f"文件不存在: {req.image_path}")

    session = _get_session(None)
    try:
        img_svc = ImageService(req.image_path)
        info = img_svc.load()
        session["image_service"] = img_svc
        session["image_path"] = req.image_path
        session["sam_service"] = SAMService(
            model_type=req.model_type,
            sam_version=req.sam_version,
        )
        session["last_mask"] = None
        session["processed_mask"] = None

        return {"session_id": session["id"], **info}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"影像加载失败: {e}")


@app.get("/api/image/display")
async def get_display_image(session_id: str = Query(...)):
    """获取降采样显示版本的 PNG 图片。"""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    img_svc = _get_image_svc(session)
    try:
        png_bytes = img_svc.get_display_image_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/image/info")
async def get_image_info(session_id: str = Query(...)):
    """获取影像元数据。"""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    img_svc = _get_image_svc(session)
    return img_svc.get_info()


# ================================================================
#  SAM 推理
# ================================================================

def _downsample_mask(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """将原始分辨率 Mask 降采样到显示尺寸。"""
    from PIL import Image as PILImage
    if mask.ndim == 3:
        mask = mask[0]
    m = (mask > 0).astype(np.uint8) * 255
    pil = PILImage.fromarray(m, mode="L")
    pil = pil.resize((target_w, target_h), resample=PILImage.NEAREST)
    return np.array(pil) > 0


@app.post("/api/predict/point")
async def predict_point(req: PointPredictRequest):
    """点提示分割。返回 Mask 叠加 PNG。"""
    session = _sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    img_svc = _get_image_svc(session)
    sam_svc = _get_sam_svc(session)
    image_path = session["image_path"]

    # 显示坐标 → 原图坐标
    orig_points = []
    for pt in req.points:
        ox, oy = img_svc.display_to_orig(pt[0], pt[1])
        orig_points.append([ox, oy])

    try:
        masks = sam_svc.predict_by_point(
            image_path=image_path,
            points=orig_points,
            labels=req.labels,
        )
        # 保存原始 mask 到会话（用于后续后处理）
        session["last_mask"] = masks
        session["processed_mask"] = None

        # 降采样到显示尺寸并返回 PNG
        info = img_svc.get_info()
        display_mask = _downsample_mask(masks, info["display_height"], info["display_width"])
        png_bytes = SAMService.mask_to_png(display_mask)
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"点分割失败: {e}")


@app.post("/api/predict/box")
async def predict_box(req: BoxPredictRequest):
    """框提示分割。"""
    session = _sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    img_svc = _get_image_svc(session)
    sam_svc = _get_sam_svc(session)
    image_path = session["image_path"]

    # 显示坐标 → 原图坐标
    ox1, oy1 = img_svc.display_to_orig(req.box[0], req.box[1])
    ox2, oy2 = img_svc.display_to_orig(req.box[2], req.box[3])
    orig_box = [ox1, oy1, ox2, oy2]

    try:
        masks = sam_svc.predict_by_box(
            image_path=image_path,
            box=orig_box,
        )
        session["last_mask"] = masks
        session["processed_mask"] = None

        info = img_svc.get_info()
        display_mask = _downsample_mask(masks, info["display_height"], info["display_width"])
        png_bytes = SAMService.mask_to_png(display_mask)
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"框分割失败: {e}")


@app.post("/api/predict/text")
async def predict_text(req: TextPredictRequest):
    """文本提示分割 (Grounded-SAM)。"""
    session = _sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    img_svc = _get_image_svc(session)
    sam_svc = _get_sam_svc(session)
    image_path = session["image_path"]

    try:
        masks = sam_svc.predict_by_text(
            image_path=image_path,
            text=req.text,
            box_threshold=req.box_threshold,
            text_threshold=req.text_threshold,
        )
        session["last_mask"] = masks
        session["processed_mask"] = None

        info = img_svc.get_info()
        display_mask = _downsample_mask(masks, info["display_height"], info["display_width"])
        png_bytes = SAMService.mask_to_png(display_mask)
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"文本分割失败: {e}")


# ================================================================
#  Mask 后处理
# ================================================================

@app.post("/api/postprocess")
async def postprocess_mask(req: PostProcessRequest):
    """对上一次预测的 Mask 进行后处理。"""
    session = _sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    if session.get("last_mask") is None:
        raise HTTPException(status_code=400, detail="请先执行一次分割预测")

    img_svc = _get_image_svc(session)
    sam_svc = _get_sam_svc(session)

    try:
        processed = sam_svc.postprocess(
            mask=session["last_mask"],
            min_size=req.min_size,
            fill_holes=req.fill_holes,
            smooth_sigma=req.smooth_sigma,
            opening_radius=req.opening_radius,
            closing_radius=req.closing_radius,
        )
        session["processed_mask"] = processed

        info = img_svc.get_info()
        display_mask = _downsample_mask(processed, info["display_height"], info["display_width"])
        png_bytes = SAMService.mask_to_png(display_mask)
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"后处理失败: {e}")


# ================================================================
#  矢量导出
# ================================================================

@app.post("/api/export/vectorize")
async def export_vectors(req: ExportRequest):
    """将当前 Mask 矢量化并导出。"""
    session = _sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    # 优先使用处理后 mask，否则使用原始 mask
    mask = session.get("processed_mask") or session.get("last_mask")
    if mask is None:
        raise HTTPException(status_code=400, detail="没有可导出的 Mask，请先执行分割")

    sam_svc = _get_sam_svc(session)
    image_path = session["image_path"]

    try:
        output_dir = os.path.join(_project_root, "output", "web_export")
        result = sam_svc.vectorize(
            mask=mask,
            image_path=image_path,
            min_area=req.min_area,
            output_format=req.output_format,
            output_dir=output_dir,
        )
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"矢量导出失败: {e}")


@app.get("/api/export/download")
async def download_export(session_id: str = Query(...)):
    """下载上一次导出的矢量文件。"""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    export_dir = os.path.join(_project_root, "output", "web_export")
    # 查找最近的导出文件
    for ext in [".geojson", ".gpkg", ".shp"]:
        fpath = os.path.join(export_dir, f"polygons{ext}")
        if os.path.isfile(fpath):
            return FileResponse(fpath, filename=os.path.basename(fpath))

    raise HTTPException(status_code=404, detail="未找到导出文件")


# ================================================================
#  会话管理 / 状态
# ================================================================

@app.get("/api/session/status")
async def session_status(session_id: str = Query(...)):
    """查询会话状态。"""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {
        "session_id": session["id"],
        "image_loaded": session.get("image_service") is not None,
        "has_mask": session.get("last_mask") is not None,
        "has_processed_mask": session.get("processed_mask") is not None,
        "image_path": session.get("image_path"),
    }


@app.delete("/api/session/clear")
async def clear_session(session_id: str = Query(...)):
    """清除会话中的 Mask（重新标注）。"""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    session["last_mask"] = None
    session["processed_mask"] = None
    return {"ok": True}


@app.get("/")
async def root():
    return {
        "name": "SAM GeoAI 标注平台",
        "version": "2.0.0",
        "docs": "/docs",
    }
