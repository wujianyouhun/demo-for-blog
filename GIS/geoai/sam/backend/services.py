"""
SAM 模型服务 + 影像服务

封装 SAM 推理、影像加载/降采样、Mask 后处理等核心业务逻辑。
影像显示由 TiTiler 动态瓦片负责，ImageService 仅提供元数据和坐标转换。
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
    """影像元数据与坐标转换服务。

    不再加载全图到内存（显示由 TiTiler 瓦片处理），
    仅读取元数据并提供 地理坐标 ↔ 像素坐标 的转换。
    """

    def __init__(self, image_path: str):
        self.image_path = os.path.abspath(image_path)
        self._orig_w = 0
        self._orig_h = 0
        self._crs: Optional[str] = None
        self._bounds: Optional[Tuple[float, float, float, float]] = None  # (left, bottom, right, top)
        self._transform = None  # rasterio Affine
        self._band_count = 0
        self._loaded = False

    def load(self) -> Dict[str, Any]:
        """读取影像元数据（不加载像素）。"""
        if self._loaded:
            return self.get_info()

        import rasterio
        with rasterio.open(self.image_path) as src:
            self._orig_w = src.width
            self._orig_h = src.height
            self._crs = str(src.crs) if src.crs else None
            self._band_count = src.count
            self._transform = src.transform
            if src.bounds:
                self._bounds = (src.bounds.left, src.bounds.bottom,
                                src.bounds.right, src.bounds.top)
            else:
                self._bounds = None

        self._loaded = True
        return self.get_info()

    def get_info(self) -> Dict[str, Any]:
        """返回前端所需的全部影像元数据。"""
        info: Dict[str, Any] = {
            "width": self._orig_w,
            "height": self._orig_h,
            "crs": self._crs,
            "band_count": self._band_count,
            "filename": os.path.basename(self.image_path),
            "has_georef": self._crs is not None and self._bounds is not None,
        }
        if self._bounds:
            info["bounds"] = list(self._bounds)  # [left, bottom, right, top]
        return info

    def geo_to_pixel(self, lon: float, lat: float) -> Tuple[float, float]:
        """地理坐标 (lon/lat in image CRS) → 像素坐标 (col, row)。

        前端发送 EPSG:4326 坐标；若影像 CRS 不同则先做投影转换。
        """
        import rasterio
        from rasterio.warp import transform as warp_transform

        if self._transform is None:
            raise RuntimeError("影像未加载或无仿射变换")

        # 如果前端坐标 (WGS84) 与影像 CRS 不同，先转投影
        src_lon, src_lat = lon, lat
        if self._crs and self._crs.upper() not in ("EPSG:4326",):
            xs, ys = warp_transform("EPSG:4326", self._crs, [lon], [lat])
            src_lon, src_lat = xs[0], ys[0]

        # 仿射逆变换: (lon, lat) → (col, row)
        inv = ~self._transform
        col, row = inv * (src_lon, src_lat)

        # 裁剪到有效范围
        col = max(0.0, min(float(col), self._orig_w - 1))
        row = max(0.0, min(float(row), self._orig_h - 1))
        return col, row

    def display_to_orig(self, dx: float, dy: float) -> Tuple[float, float]:
        """保留向后兼容（不再使用，由 geo_to_pixel 替代）。"""
        return dx, dy

    def get_mask_extent(self) -> Optional[List[float]]:
        """返回 mask 叠加层所需的 EPSG:3857 范围 [minX, minY, maxX, maxY]。"""
        if not self._bounds:
            return None
        from pyproj import Transformer
        transformer = Transformer.from_crs(
            self._crs or "EPSG:4326", "EPSG:3857", always_xy=True
        )
        left, bottom, right, top = self._bounds
        x_min, y_min = transformer.transform(left, bottom)
        x_max, y_max = transformer.transform(right, top)
        return [x_min, y_min, x_max, y_max]


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
