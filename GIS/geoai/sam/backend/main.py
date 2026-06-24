"""
SAM 遥感标注 Web 服务 — FastAPI 入口

提供 TiTiler 动态瓦片 + SAM 推理（点/框/文本）+ Mask 后处理 + 矢量导出。
影像显示由 TiTiler 按需切片，浏览器仅加载视口内瓦片，支持超大 GeoTIFF。

启动方式:
    cd backend
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import uuid
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
from pydantic import BaseModel

# ── 将项目根目录加入 sys.path ──────────────────────────────
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from services import ImageService, SAMService
from config import DEFAULT_IMAGE, SAM_MODEL_TYPE, SAM_VERSION

# ── FastAPI 应用 ────────────────────────────────────────────
app = FastAPI(
    title="SAM GeoAI 标注平台",
    description="基于 SAM + TiTiler 的遥感影像半自动标注 Web 服务",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 挂载 TiTiler 动态瓦片服务 ──────────────────────────────
from titiler.core.factory import TilerFactory

cog_tiler = TilerFactory()
app.include_router(cog_tiler.router, prefix="/tiles/cog", tags=["TiTiler 瓦片"])


# ── 会话存储 ────────────────────────────────────────────────
_sessions: Dict[str, Dict[str, Any]] = {}


def _get_session(session_id: Optional[str]) -> Dict[str, Any]:
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


def _raise_predict_error(mode: str, exc: Exception):
    if isinstance(exc, ImportError):
        raise HTTPException(
            status_code=503,
            detail=f"{mode}所需依赖未安装或不可用: {exc}",
        )
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc))
    raise HTTPException(status_code=500, detail=f"{mode}失败: {exc}")


def _geo_to_pixel_or_error(img_svc: ImageService, lon: float, lat: float):
    try:
        return img_svc.geo_to_pixel(lon, lat)
    except Exception as exc:
        raise ValueError(f"坐标转换失败，请检查影像 CRS 与前端地图范围: {exc}") from exc


# ================================================================
#  请求/响应模型
# ================================================================

class LoadImageRequest(BaseModel):
    image_path: str
    model_type: str = "vit_l"
    sam_version: str = "sam1"


class PointPredictRequest(BaseModel):
    session_id: str
    # 地理坐标 [[lon1,lat1], [lon2,lat2], ...]（EPSG:4326）
    points: List[List[float]]
    labels: List[int]            # 1=前景, 0=背景


class BoxPredictRequest(BaseModel):
    session_id: str
    # 地理坐标 [lon1, lat1, lon2, lat2]
    box: List[float]


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
    output_format: str = "geojson"


# ================================================================
#  影像管理
# ================================================================

@app.post("/api/image/load")
async def load_image(req: LoadImageRequest):
    """加载影像文件，返回元数据、瓦片 URL 和会话 ID。"""
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

        abs_path = os.path.abspath(req.image_path)
        # 构造 TiTiler 瓦片 URL 模板
        info["tile_url"] = (
            f"/tiles/cog/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}.png"
            f"?url={abs_path}"
        )
        info["tile_json_url"] = (
            f"/tiles/cog/WebMercatorQuad/tilejson.json"
            f"?url={abs_path}"
        )
        info["mask_extent"] = img_svc.get_mask_extent()
        info["session_id"] = session["id"]

        return info
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"影像加载失败: {e}")


@app.get("/api/image/info")
async def get_image_info(session_id: str = Query(...)):
    """获取影像元数据。"""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    img_svc = _get_image_svc(session)
    info = img_svc.get_info()
    # 补充 EPSG:3857 extent（用于 mask 叠加定位）
    info["mask_extent"] = img_svc.get_mask_extent()
    return info


# ================================================================
#  SAM 推理（前端发送地理坐标，后端转像素坐标）
# ================================================================

# Mask 降采样最大尺寸（用于前端叠加显示）
_MASK_DISPLAY_MAX = 2048


def _downsample_mask_png(
    mask: np.ndarray,
    orig_h: int,
    orig_w: int,
    crop_offset: Optional[Tuple[int, int, int, int]] = None,
) -> bytes:
    """将原始分辨率 mask 降采样到 ≤2048px 并编码为 RGBA PNG。

    Args:
        mask: SAM 输出的 mask (可能是裁剪后的子区域)
        orig_h, orig_w: 原始影像尺寸
        crop_offset: (col_off, row_off, crop_w, crop_h) 如果是裁剪后的 mask
    """
    from PIL import Image as PILImage

    while mask.ndim > 2:
        mask = mask[0]

    scale = min(_MASK_DISPLAY_MAX / max(orig_h, orig_w), 1.0)
    dh = max(1, int(orig_h * scale))
    dw = max(1, int(orig_w * scale))

    if crop_offset is not None:
        # 裁剪后的 mask: 放置到全图降采样画布的正确位置
        col_off, row_off, crop_w, crop_h = crop_offset

        # 创建全图降采样画布
        full_canvas = np.zeros((dh, dw), dtype=bool)

        # 计算 mask 在降采样画布中的位置
        mask_h, mask_w = mask.shape
        ds_col_off = int(col_off * scale)
        ds_row_off = int(row_off * scale)
        ds_mask_w = max(1, int(mask_w * scale))
        ds_mask_h = max(1, int(mask_h * scale))

        # 降采样 mask
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        pil_mask = PILImage.fromarray(mask_uint8, mode="L")
        pil_mask = pil_mask.resize((ds_mask_w, ds_mask_h), resample=PILImage.NEAREST)
        mask_small = np.array(pil_mask) > 0

        # 放置到画布上 (注意边界裁剪)
        end_col = min(ds_col_off + ds_mask_w, dw)
        end_row = min(ds_row_off + ds_mask_h, dh)
        actual_w = end_col - ds_col_off
        actual_h = end_row - ds_row_off

        if actual_w > 0 and actual_h > 0:
            full_canvas[ds_row_off:end_row, ds_col_off:end_col] = mask_small[:actual_h, :actual_w]

        m_small = full_canvas
    else:
        # 完整 mask: 直接降采样
        m = (mask > 0).astype(np.uint8) * 255
        pil = PILImage.fromarray(m, mode="L")
        pil = pil.resize((dw, dh), resample=PILImage.NEAREST)
        m_small = np.array(pil) > 0

    rgba = np.zeros((*m_small.shape, 4), dtype=np.uint8)
    rgba[m_small] = [50, 230, 80, 100]  # 绿色半透明
    out = PILImage.fromarray(rgba, mode="RGBA")
    import io
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


@app.post("/api/predict/point")
async def predict_point(req: PointPredictRequest):
    """点提示分割。前端发送 EPSG:4326 地理坐标。"""
    session = _sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    img_svc = _get_image_svc(session)
    sam_svc = _get_sam_svc(session)
    image_path = session["image_path"]

    try:
        # 地理坐标 → 像素坐标
        if not req.points:
            raise ValueError("点标注至少需要 1 个提示点")
        if len(req.points) != len(req.labels):
            raise ValueError("点坐标数量必须与标签数量一致")

        pixel_points = []
        for pt in req.points:
            col, row = _geo_to_pixel_or_error(img_svc, pt[0], pt[1])
            pixel_points.append([col, row])

        masks = sam_svc.predict_by_points(
            image_path=image_path,
            points=pixel_points,
            labels=req.labels,
        )
        session["last_mask"] = masks
        session["processed_mask"] = None
        session["crop_offset"] = sam_svc._crop_offset

        info = img_svc.get_info()
        png_bytes = _downsample_mask_png(masks, info["height"], info["width"], sam_svc._crop_offset)
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        traceback.print_exc()
        _raise_predict_error("点标注分割", e)


@app.post("/api/predict/box")
async def predict_box(req: BoxPredictRequest):
    """框提示分割。前端发送 [lon1, lat1, lon2, lat2]。"""
    session = _sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    img_svc = _get_image_svc(session)
    sam_svc = _get_sam_svc(session)
    image_path = session["image_path"]

    try:
        if len(req.box) != 4:
            raise ValueError("框标注需要 4 个坐标值: [lon1, lat1, lon2, lat2]")

        col1, row1 = _geo_to_pixel_or_error(img_svc, req.box[0], req.box[1])
        col2, row2 = _geo_to_pixel_or_error(img_svc, req.box[2], req.box[3])
        pixel_box = [
            min(col1, col2),
            min(row1, row2),
            max(col1, col2),
            max(row1, row2),
        ]

        masks = sam_svc.predict_by_box(
            image_path=image_path,
            box=pixel_box,
        )
        session["last_mask"] = masks
        session["processed_mask"] = None
        session["crop_offset"] = sam_svc._crop_offset

        info = img_svc.get_info()
        png_bytes = _downsample_mask_png(masks, info["height"], info["width"], sam_svc._crop_offset)
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        traceback.print_exc()
        _raise_predict_error("框标注分割", e)


@app.post("/api/predict/text")
async def predict_text(req: TextPredictRequest):
    """文本提示分割 (Grounded-SAM)。"""
    session = _sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

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

        img_svc = _get_image_svc(session)
        info = img_svc.get_info()
        png_bytes = _downsample_mask_png(masks, info["height"], info["width"])
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        traceback.print_exc()
        _raise_predict_error("文本标注分割", e)


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
        crop_offset = session.get("crop_offset")
        png_bytes = _downsample_mask_png(processed, info["height"], info["width"], crop_offset)
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
    for ext in [".geojson", ".gpkg", ".shp"]:
        fpath = os.path.join(export_dir, f"polygons{ext}")
        if os.path.isfile(fpath):
            return FileResponse(fpath, filename=os.path.basename(fpath))

    raise HTTPException(status_code=404, detail="未找到导出文件")


# ================================================================
#  会话管理
# ================================================================

@app.get("/api/session/status")
async def session_status(session_id: str = Query(...)):
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
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    session["last_mask"] = None
    session["processed_mask"] = None
    return {"ok": True}


@app.get("/api/config")
async def get_config():
    return {
        "default_image": DEFAULT_IMAGE,
        "default_model_type": SAM_MODEL_TYPE,
        "default_sam_version": SAM_VERSION,
    }


@app.get("/")
async def root():
    return {
        "name": "SAM GeoAI 标注平台",
        "version": "2.1.0",
        "tiles": "/tiles/cog",
        "docs": "/docs",
    }
