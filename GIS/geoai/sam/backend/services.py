"""
SAM 模型服务 + 影像服务

封装 SAM 推理、影像加载/降采样、Mask 后处理等核心业务逻辑。
影像显示由 TiTiler 动态瓦片负责，ImageService 仅提供元数据和坐标转换。
"""

import os
import io
import sys
import importlib.util
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

    # 大影像阈值: 超过此尺寸时自动裁剪子区域
    _MAX_SAM_DIM = 2048
    # 裁剪边距 (像素)
    _CROP_PADDING = 200

    def __init__(self, model_type: str = "vit_l", sam_version: str = "sam1"):
        self.model_type = model_type
        self.sam_version = sam_version
        self._sam = None
        self._gsam = None
        self._image_set = False
        self._image_path = None
        # 当前裁剪信息: (col_off, row_off, crop_w, crop_h) 或 None
        self._crop_offset = None
        # 原始影像尺寸
        self._orig_w = 0
        self._orig_h = 0

    _SAM1_CHECKPOINTS = {
        "vit_h": ("sam_vit_h_4b8939.pth", 2566786440),
        "vit_l": ("sam_vit_l_0b3195.pth", 1249599818),
        "vit_b": ("sam_vit_b_01ec64.pth", 393891747),
    }

    def _validate_sam_runtime(self):
        missing = [
            name for name in ("torch", "samgeo")
            if importlib.util.find_spec(name) is None
        ]
        if missing:
            raise ImportError(f"缺少基础 SAM 依赖: {', '.join(missing)}")

        if self.sam_version != "sam1":
            return

        entry = self._SAM1_CHECKPOINTS.get(self.model_type)
        if entry is None:
            return

        filename, expected_size = entry
        local_path = os.path.join(_project_root, "models", filename)
        if not os.path.isfile(local_path):
            raise ImportError(f"SAM1 模型文件不存在: {local_path}，请先运行 python download_models.py --{self.model_type}")

        actual_size = os.path.getsize(local_path)
        if actual_size < expected_size * 0.95:
            actual_mb = actual_size / 1024 / 1024
            expected_mb = expected_size / 1024 / 1024
            raise ImportError(
                f"SAM1 {self.model_type} 模型文件不完整: {local_path} "
                f"({actual_mb:.1f}MB / 期望约 {expected_mb:.1f}MB)，请删除该文件后重新下载"
            )

    @staticmethod
    def _normalize_masks(masks: Any, mode: str) -> np.ndarray:
        """Normalize different SAM wrappers to a non-empty numpy mask array."""
        if masks is None:
            raise ValueError(f"{mode}未生成 Mask，请检查提示参数或模型依赖")

        if isinstance(masks, dict):
            masks = masks.get("masks", masks.get("mask"))
            if masks is None:
                raise ValueError(f"{mode}未生成 Mask，请检查提示参数或模型依赖")
        elif isinstance(masks, list):
            if not masks:
                raise ValueError(f"{mode}未生成 Mask，请调整提示参数后重试")
            if isinstance(masks[0], dict) and "segmentation" in masks[0]:
                masks = np.stack([m["segmentation"] for m in masks], axis=0)

        masks = np.asarray(masks)
        if masks.size == 0:
            raise ValueError(f"{mode}生成了空 Mask，请调整提示参数后重试")
        return masks

    def predict_by_point(
        self,
        image_path: str,
        points: List[List[float]],
        labels: List[int],
    ) -> np.ndarray:
        """Backward-compatible alias for older callers."""
        return self.predict_by_points(image_path=image_path, points=points, labels=labels)

    def _ensure_sam(self):
        if self._sam is not None:
            return
        self._validate_sam_runtime()
        from geoai_sam import SAMWrapper
        self._sam = SAMWrapper(
            model_type=self.model_type,
            sam_version=self.sam_version,
            automatic=False,
        )

    def _get_image_dimensions(self, image_path: str) -> Tuple[int, int]:
        """读取影像宽高（不加载像素数据）。"""
        import rasterio
        with rasterio.open(image_path) as src:
            return src.width, src.height

    def _needs_cropping(self, image_path: str) -> bool:
        """判断影像是否需要裁剪（尺寸过大）。"""
        w, h = self._get_image_dimensions(image_path)
        self._orig_w = w
        self._orig_h = h
        return max(w, h) > self._MAX_SAM_DIM

    def _crop_around_region(
        self,
        image_path: str,
        cx: float,
        cy: float,
        region_w: float = 0,
        region_h: float = 0,
    ) -> Tuple[str, int, int, int, int]:
        """裁取提示区域周围的子影像。

        Args:
            image_path: 原始影像路径
            cx, cy: 提示区域中心 (像素坐标)
            region_w, region_h: 提示区域宽高 (像素, 框提示时使用)

        Returns:
            (temp_path, col_off, row_off, crop_w, crop_h)
        """
        import rasterio
        from rasterio.windows import Window
        import tempfile

        w, h = self._orig_w, self._orig_h
        padding = self._CROP_PADDING

        # 计算裁剪窗口大小
        if region_w > 0 and region_h > 0:
            # 框提示: 窗口 = 框大小 + 边距, 但不超过阈值
            crop_w = min(int(region_w) + padding * 2, self._MAX_SAM_DIM, w)
            crop_h = min(int(region_h) + padding * 2, self._MAX_SAM_DIM, h)
        else:
            # 点提示: 使用最大允许尺寸
            crop_w = min(self._MAX_SAM_DIM, w)
            crop_h = min(self._MAX_SAM_DIM, h)

        # 计算裁剪起点 (居中于提示区域)
        col_off = int(cx - crop_w / 2)
        row_off = int(cy - crop_h / 2)

        # 裁剪到有效范围
        col_off = max(0, min(col_off, w - crop_w))
        row_off = max(0, min(row_off, h - crop_h))

        # 读取窗口并保存
        with rasterio.open(image_path) as src:
            win = Window(col_off=col_off, row_off=row_off,
                         width=crop_w, height=crop_h)
            data = src.read(window=win)

            profile = src.profile.copy()
            profile.update(
                width=crop_w,
                height=crop_h,
                transform=rasterio.windows.transform(win, src.transform),
            )

            # 保存到临时文件
            temp_dir = os.path.join(_project_root, "output", "tmp")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, f"crop_{col_off}_{row_off}.tif")

            with rasterio.open(temp_path, "w", **profile) as dst:
                dst.write(data)

        print(f"[SAMService] 裁剪子影像: ({col_off}, {row_off}) {crop_w}x{crop_h} -> {temp_path}")
        return temp_path, col_off, row_off, crop_w, crop_h

    def _ensure_image(self, image_path: str):
        """加载影像到 SAM。大影像由调用方通过 crop 处理。"""
        self._ensure_sam()
        if not self._image_set or self._image_path != image_path:
            self._sam.set_image(image_path)
            self._image_set = True
            self._image_path = image_path

    def predict_by_points(
        self,
        image_path: str,
        points: List[List[float]],
        labels: List[int],
    ) -> np.ndarray:
        """点提示分割（大影像自动裁剪子区域）。"""
        self._ensure_sam()

        # 检查是否需要裁剪
        if self._needs_cropping(image_path):
            # 计算提示点中心
            cx = sum(p[0] for p in points) / len(points)
            cy = sum(p[1] for p in points) / len(points)

            # 裁剪子影像
            temp_path, col_off, row_off, crop_w, crop_h = self._crop_around_region(
                image_path, cx, cy
            )
            self._crop_offset = (col_off, row_off, crop_w, crop_h)

            # 调整点坐标到裁剪后影像的局部坐标
            local_points = [[p[0] - col_off, p[1] - row_off] for p in points]

            # 在子影像上运行 SAM
            self._ensure_image(temp_path)
            masks = self._sam.generate_masks_by_points(
                points=local_points,
                point_labels=labels,
            )
        else:
            self._crop_offset = None
            self._ensure_image(image_path)
            masks = self._sam.generate_masks_by_points(
                points=points,
                point_labels=labels,
            )

        return self._normalize_masks(masks, "点标注")

    def predict_by_box(
        self,
        image_path: str,
        box: List[float],
    ) -> np.ndarray:
        """框提示分割（大影像自动裁剪子区域）。"""
        self._ensure_sam()

        # 检查是否需要裁剪
        if self._needs_cropping(image_path):
            # 框中心和尺寸
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            region_w = box[2] - box[0]
            region_h = box[3] - box[1]

            # 裁剪子影像
            temp_path, col_off, row_off, crop_w, crop_h = self._crop_around_region(
                image_path, cx, cy, region_w, region_h
            )
            self._crop_offset = (col_off, row_off, crop_w, crop_h)

            # 调整框坐标到局部坐标
            local_box = [
                box[0] - col_off,
                box[1] - row_off,
                box[2] - col_off,
                box[3] - row_off,
            ]

            # 在子影像上运行 SAM
            self._ensure_image(temp_path)
            masks = self._sam.generate_masks_by_box(box=local_box)
        else:
            self._crop_offset = None
            self._ensure_image(image_path)
            masks = self._sam.generate_masks_by_box(box=box)

        return self._normalize_masks(masks, "框标注")

    def predict_by_boxes(
        self,
        image_path: str,
        boxes: List[List[float]],
    ) -> np.ndarray:
        """多框批量分割（大影像自动裁剪子区域）。"""
        self._ensure_sam()

        if self._needs_cropping(image_path):
            # 计算所有框的总范围
            all_x1 = min(b[0] for b in boxes)
            all_y1 = min(b[1] for b in boxes)
            all_x2 = max(b[2] for b in boxes)
            all_y2 = max(b[3] for b in boxes)
            cx = (all_x1 + all_x2) / 2
            cy = (all_y1 + all_y2) / 2
            region_w = all_x2 - all_x1
            region_h = all_y2 - all_y1

            temp_path, col_off, row_off, crop_w, crop_h = self._crop_around_region(
                image_path, cx, cy, region_w, region_h
            )
            self._crop_offset = (col_off, row_off, crop_w, crop_h)

            local_boxes = [
                [b[0] - col_off, b[1] - row_off, b[2] - col_off, b[3] - row_off]
                for b in boxes
            ]

            self._ensure_image(temp_path)
            masks = self._sam.generate_masks_by_boxes(boxes=local_boxes)
        else:
            self._crop_offset = None
            self._ensure_image(image_path)
            masks = self._sam.generate_masks_by_boxes(boxes=boxes)

        return self._normalize_masks(masks, "框标注")

    def predict_by_text(
        self,
        image_path: str,
        text: str,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ) -> np.ndarray:
        """文本提示分割 (Grounded-SAM)。"""
        self._validate_sam_runtime()
        if self._gsam is None:
            from geoai_sam import GroundedSAMWrapper
            self._gsam = GroundedSAMWrapper(
                sam_model_type=self.model_type,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
        if getattr(self._gsam, "_image_path", None) != image_path:
            self._gsam.set_image(image_path)

        masks = self._gsam.segment_by_text(
            text,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        return self._normalize_masks(masks, "文本标注")

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
        """Mask 矢量化并导出。

        如果存在裁剪偏移，使用裁剪后的子影像作为参考，
        以确保矢量坐标正确。
        """
        from geoai_sam import MaskVectorizer

        os.makedirs(output_dir, exist_ok=True)

        # 如果有裁剪偏移，使用裁剪后的子影像作为参考
        ref_image = image_path
        if self._crop_offset is not None:
            col_off, row_off, crop_w, crop_h = self._crop_offset
            temp_dir = os.path.join(_project_root, "output", "tmp")
            ref_image = os.path.join(temp_dir, f"crop_{col_off}_{row_off}.tif")
            if not os.path.exists(ref_image):
                ref_image = image_path

        vec = MaskVectorizer()
        gdf = vec.vectorize(mask, reference_image=ref_image, min_area=min_area)

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
