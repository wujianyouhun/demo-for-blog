"""
SAM 模型服务 + 影像服务

封装 SAM 推理、影像加载/降采样、Mask 后处理等核心业务逻辑。
"""

import os
import io
import sys
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from PIL import Image

# 将项目根目录加入 path，以便导入 geoai_sam
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


class ImageService:
    """影像加载与降采样服务。"""

    MAX_DISPLAY = 4096  # 显示用图片最大边长

    def __init__(self, image_path: str):
        self.image_path = os.path.abspath(image_path)
        self._orig_w = 0
        self._orig_h = 0
        self._crs = None
        self._display_img: Optional[np.ndarray] = None
        self._scale_x = 1.0
        self._scale_y = 1.0
        self._loaded = False

    def load(self) -> Dict[str, Any]:
        """加载影像元数据并生成降采样显示版本。"""
        if self._loaded:
            return self.get_info()

        try:
            import rasterio
            from rasterio.enums import Resampling

            with rasterio.open(self.image_path) as src:
                self._orig_w = src.width
                self._orig_h = src.height
                self._crs = str(src.crs) if src.crs else None
                bands = min(src.count, 3)

                # 计算降采样尺寸
                scale = min(self.MAX_DISPLAY / max(self._orig_h, self._orig_w), 1.0)
                dh = max(1, int(self._orig_h * scale))
                dw = max(1, int(self._orig_w * scale))

                # 使用 rasterio 内置降采样读取
                img = src.read(
                    list(range(1, bands + 1)),
                    out_shape=(bands, dh, dw),
                    resampling=Resampling.average,
                )
                img = np.transpose(img, (1, 2, 0))
        except ImportError:
            import cv2
            raw = cv2.imread(self.image_path)
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            self._orig_h, self._orig_w = raw.shape[:2]

            scale = min(self.MAX_DISPLAY / max(self._orig_h, self._orig_w), 1.0)
            dh = max(1, int(self._orig_h * scale))
            dw = max(1, int(self._orig_w * scale))
            img = cv2.resize(raw, (dw, dh), interpolation=cv2.INTER_AREA)

        # 归一化到 0-255 uint8
        img = img.astype(np.float32)
        lo, hi = img.min(), img.max()
        if hi > lo:
            img = ((img - lo) / (hi - lo) * 255).astype(np.uint8)
        else:
            img = np.zeros_like(img, dtype=np.uint8)

        self._display_img = img
        self._scale_x = self._orig_w / img.shape[1]
        self._scale_y = self._orig_h / img.shape[0]
        self._loaded = True

        return self.get_info()

    def get_info(self) -> Dict[str, Any]:
        return {
            "width": self._orig_w,
            "height": self._orig_h,
            "display_width": self._display_img.shape[1] if self._display_img is not None else 0,
            "display_height": self._display_img.shape[0] if self._display_img is not None else 0,
            "scale_x": self._scale_x,
            "scale_y": self._scale_y,
            "crs": self._crs,
            "filename": os.path.basename(self.image_path),
        }

    def get_display_image_png(self) -> bytes:
        """返回降采样显示版本的 PNG 字节。"""
        if self._display_img is None:
            raise RuntimeError("影像未加载，请先调用 load()")
        img = self._display_img
        if img.shape[2] == 1:
            pil_img = Image.fromarray(img[:, :, 0], mode="L")
        else:
            pil_img = Image.fromarray(img[:, :, :3])
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return buf.getvalue()

    def display_to_orig(self, dx: float, dy: float) -> Tuple[float, float]:
        """显示坐标 → 原图像素坐标。"""
        ox = dx * self._scale_x
        oy = dy * self._scale_y
        ox = max(0, min(ox, self._orig_w - 1))
        oy = max(0, min(oy, self._orig_h - 1))
        return ox, oy


class SAMService:
    """SAM 模型推理服务（延迟加载）。"""

    def __init__(self, model_type: str = "vit_l", sam_version: str = "sam1"):
        self.model_type = model_type
        self.sam_version = sam_version
        self._sam = None
        self._gsam = None
        self._image_set = False

    def _ensure_sam(self):
        if self._sam is not None:
            return
        from geoai_sam import SAMWrapper
        self._sam = SAMWrapper(
            model_type=self.model_type,
            sam_version=self.sam_version,
            automatic=False,
        )

    def _ensure_image(self, image_path: str):
        self._ensure_sam()
        if not self._image_set:
            self._sam.set_image(image_path)
            self._image_set = True

    def predict_by_points(
        self,
        image_path: str,
        points: List[List[float]],
        labels: List[int],
    ) -> np.ndarray:
        """点提示分割。"""
        self._ensure_image(image_path)
        masks = self._sam.generate_masks_by_points(
            points=points,
            point_labels=labels,
        )
        return masks

    def predict_by_box(
        self,
        image_path: str,
        box: List[float],
    ) -> np.ndarray:
        """框提示分割。"""
        self._ensure_image(image_path)
        masks = self._sam.generate_masks_by_box(box=box)
        return masks

    def predict_by_boxes(
        self,
        image_path: str,
        boxes: List[List[float]],
    ) -> np.ndarray:
        """多框批量分割。"""
        self._ensure_image(image_path)
        masks = self._sam.generate_masks_by_boxes(boxes=boxes)
        return masks

    def predict_by_text(
        self,
        image_path: str,
        text: str,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ) -> np.ndarray:
        """文本提示分割 (Grounded-SAM)。"""
        if self._gsam is None:
            from geoai_sam import GroundedSAMWrapper
            self._gsam = GroundedSAMWrapper(
                sam_model_type=self.model_type,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
            self._gsam.set_image(image_path)

        masks = self._gsam.segment_by_text(
            text,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        return masks

    def postprocess(
        self,
        mask: np.ndarray,
        min_size: int = 200,
        fill_holes: bool = True,
        smooth_sigma: float = 1.5,
        opening_radius: int = 2,
        closing_radius: int = 3,
    ) -> np.ndarray:
        """Mask 后处理。"""
        from geoai_sam import MaskPostProcessor
        return MaskPostProcessor.default_pipeline(
            mask,
            min_size=min_size,
            fill_holes_flag=fill_holes,
            smooth_sigma=smooth_sigma,
            opening_radius=opening_radius,
            closing_radius=closing_radius,
        )

    def vectorize(
        self,
        mask: np.ndarray,
        image_path: str,
        min_area: int = 50,
        output_format: str = "geojson",
        output_dir: str = "output/web_export",
    ) -> Dict[str, Any]:
        """Mask 矢量化并导出。"""
        from geoai_sam import MaskVectorizer

        os.makedirs(output_dir, exist_ok=True)
        vec = MaskVectorizer()
        gdf = vec.vectorize(mask, reference_image=image_path, min_area=min_area)

        results = {"polygon_count": vec.get_polygon_count()}

        ext_map = {
            "geojson": ".geojson",
            "gpkg": ".gpkg",
            "shp": ".shp",
        }
        ext = ext_map.get(output_format, ".geojson")
        out_path = os.path.join(output_dir, f"polygons{ext}")
        try:
            vec.save(out_path)
            results["file"] = out_path
        except Exception as e:
            results["error"] = str(e)

        return results

    @staticmethod
    def mask_to_png(mask: np.ndarray, alpha: float = 0.4) -> bytes:
        """将二值 Mask 转为带透明度的 PNG（用于前端叠加显示）。"""
        if mask.ndim == 3:
            mask = mask[0]
        m = mask.astype(bool)
        # RGBA: 绿色半透明
        rgba = np.zeros((*m.shape, 4), dtype=np.uint8)
        rgba[m] = [50, 230, 80, int(alpha * 255)]
        pil = Image.fromarray(rgba, mode="RGBA")
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return buf.getvalue()
