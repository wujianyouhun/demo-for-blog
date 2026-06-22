"""
GeoAI 要素提取引擎
====================
支持多种提取方式：
  1. 传统 CV：HSV 颜色空间 + 形态学操作
  2. 深度学习：DeepLabV3+ (ResNet34) 语义分割
  3. 混合模式：CV 预筛选 + DL 精细分割

提取类别：建筑 (building)、林地 (forest)、草地 (grassland)
"""

import numpy as np
import cv2
import rasterio
from rasterio.features import shapes as rasterio_shapes
from rasterio.windows import Window
from shapely.geometry import shape as shapely_shape, box as shapely_box
from shapely.ops import unary_union
import geopandas as gpd
import pandas as pd
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
import time
import os


@dataclass
class ExtractionConfig:
    """提取配置"""
    # 目标类别
    targets: List[str] = field(default_factory=lambda: ["building"])
    # 提取方法: cv / dl / hybrid
    method: str = "cv"
    # CV 参数
    building_canny_low: int = 50
    building_canny_high: int = 150
    building_min_area_px: int = 100
    building_kernel_size: int = 5
    forest_hsv_h_range: Tuple[int, int] = (35, 85)
    forest_hsv_s_min: int = 40
    forest_hsv_v_min: int = 40
    forest_min_area_px: int = 200
    grassland_hsv_h_range: Tuple[int, int] = (25, 75)
    grassland_hsv_s_range: Tuple[int, int] = (20, 120)
    grassland_hsv_v_range: Tuple[int, int] = (60, 200)
    grassland_min_area_px: int = 300
    # DL 参数
    dl_model: str = "deeplabv3plus"
    dl_backbone: str = "resnet34"
    dl_tile_size: int = 512
    dl_batch_size: int = 4
    # 通用
    simplify_tolerance: float = 1.0
    min_polygon_area: float = 10.0  # 平方米


class GeoAIExtractor:
    """GeoAI 要素提取器"""

    def __init__(self, tif_path: str, config: ExtractionConfig = None):
        self.tif_path = tif_path
        self.config = config or ExtractionConfig()
        self.results: Dict[str, gpd.GeoDataFrame] = {}
        self.stats: Dict[str, Any] = {}
        self._ds = None
        self._dl_model = None

        with rasterio.open(tif_path) as ds:
            self.crs = ds.crs
            self.transform = ds.transform
            self.width = ds.width
            self.height = ds.height
            self.bounds = ds.bounds
            self.res = ds.res

    def extract_region(self, bounds_latlon: dict,
                       progress_callback=None) -> Dict[str, gpd.GeoDataFrame]:
        """提取指定经纬度范围内的要素

        Args:
            bounds_latlon: {"left": lon, "bottom": lat, "right": lon, "top": lat}
            progress_callback: fn(stage, pct) 进度回调
        """
        start = time.time()
        self.results = {}

        # 1. 读取区域影像
        if progress_callback:
            progress_callback("reading", 0)
        data, window_transform = self._read_region(bounds_latlon)
        if data is None:
            return {}

        if progress_callback:
            progress_callback("reading", 100)

        # 2. 执行提取
        for i, target in enumerate(self.config.targets):
            pct_base = int((i / len(self.config.targets)) * 80) + 10
            if progress_callback:
                progress_callback(f"extracting_{target}", pct_base)

            mask = self._extract_target(data, target)

            if progress_callback:
                progress_callback(f"vectorizing_{target}", pct_base + 5)

            gdf = self._vectorize_mask(mask, window_transform, target)
            self.results[target] = gdf

        # 3. 统计
        elapsed = time.time() - start
        self.stats = {
            "elapsed_seconds": round(elapsed, 2),
            "method": self.config.method,
            "targets": {},
        }
        for target, gdf in self.results.items():
            self.stats["targets"][target] = {
                "count": len(gdf),
                "total_area_m2": round(gdf["area_m2"].sum(), 2) if len(gdf) > 0 else 0,
            }

        if progress_callback:
            progress_callback("done", 100)

        return self.results

    def extract_tiles(self, tile_bounds_list: List[dict],
                      progress_callback=None) -> Dict[str, gpd.GeoDataFrame]:
        """批量提取多个瓦片区域的要素，合并结果"""
        all_results: Dict[str, List[gpd.GeoDataFrame]] = {
            t: [] for t in self.config.targets
        }

        total = len(tile_bounds_list)
        for i, bounds in enumerate(tile_bounds_list):
            if progress_callback:
                progress_callback("processing", int(i / total * 100))
            results = self.extract_region(bounds)
            for target, gdf in results.items():
                if len(gdf) > 0:
                    all_results[target].append(gdf)

        # 合并
        for target in self.config.targets:
            if all_results[target]:
                merged = gpd.GeoDataFrame(
                    pd.concat(all_results[target], ignore_index=True),
                    crs=self.crs,
                )
                # 去重 (相邻瓦片边界处的重复)
                merged = merged.drop_duplicates(subset="geometry", keep="first")
                merged = merged.reset_index(drop=True)
                self.results[target] = merged
            else:
                self.results[target] = gpd.GeoDataFrame(
                    columns=["class", "area_m2", "geometry"],
                    geometry="geometry", crs=self.crs,
                )

        return self.results

    # ══════════════════════════════════════════════════
    #  提取方法实现
    # ══════════════════════════════════════════════════

    def _extract_target(self, data: np.ndarray, target: str) -> np.ndarray:
        """根据目标类别选择提取方法"""
        method = self.config.method

        if method == "cv":
            return self._extract_cv(data, target)
        elif method == "dl":
            return self._extract_dl(data, target)
        elif method == "hybrid":
            # 混合模式: CV 粗提取 + DL 精修
            cv_mask = self._extract_cv(data, target)
            dl_mask = self._extract_dl(data, target)
            # 取并集
            return np.logical_or(cv_mask, dl_mask).astype(np.uint8)
        else:
            raise ValueError(f"未知的提取方法: {method}")

    def _extract_cv(self, data: np.ndarray, target: str) -> np.ndarray:
        """传统 CV 提取"""
        # (bands, h, w) → (h, w, bands) → BGR for OpenCV
        img = np.transpose(data[:3], (1, 2, 0))
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if target == "building":
            return self._cv_extract_building(bgr)
        elif target == "forest":
            return self._cv_extract_forest(bgr)
        elif target == "grassland":
            return self._cv_extract_grassland(bgr)
        else:
            raise ValueError(f"不支持的类别: {target}")

    def _cv_extract_building(self, bgr: np.ndarray) -> np.ndarray:
        """CV 建筑提取: Canny 边缘 + 形态学"""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 高斯模糊降噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny 边缘检测
        cfg = self.config
        edges = cv2.Canny(blurred, cfg.building_canny_low, cfg.building_canny_high)

        # 形态学闭运算连接边缘
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (cfg.building_kernel_size, cfg.building_kernel_size)
        )
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 膨胀扩展
        dilated = cv2.dilate(closed, kernel, iterations=1)

        # 填充内部
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= cfg.building_min_area_px:
                cv2.drawContours(mask, [cnt], -1, 255, -1)

        return mask

    def _cv_extract_forest(self, bgr: np.ndarray) -> np.ndarray:
        """CV 林地提取: HSV 颜色阈值"""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        cfg = self.config

        lower = np.array([
            cfg.forest_hsv_h_range[0],
            cfg.forest_hsv_s_min,
            cfg.forest_hsv_v_min,
        ])
        upper = np.array([
            cfg.forest_hsv_h_range[1], 255, 255,
        ])

        mask = cv2.inRange(hsv, lower, upper)

        # 形态学降噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 过滤小区域
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = np.zeros_like(mask)
        for cnt in contours:
            if cv2.contourArea(cnt) >= cfg.forest_min_area_px:
                cv2.drawContours(filtered, [cnt], -1, 255, -1)

        return filtered

    def _cv_extract_grassland(self, bgr: np.ndarray) -> np.ndarray:
        """CV 草地提取: HSV 颜色阈值 (低饱和度)"""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        cfg = self.config

        lower = np.array([
            cfg.grassland_hsv_h_range[0],
            cfg.grassland_hsv_s_range[0],
            cfg.grassland_hsv_v_range[0],
        ])
        upper = np.array([
            cfg.grassland_hsv_h_range[1],
            cfg.grassland_hsv_s_range[1],
            cfg.grassland_hsv_v_range[1],
        ])

        mask = cv2.inRange(hsv, lower, upper)

        # 与林地排除: 林地饱和度更高，草地饱和度低
        # 排除已经是林地的区域
        forest_mask = self._cv_extract_forest(bgr)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(forest_mask))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 过滤小区域
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = np.zeros_like(mask)
        for cnt in contours:
            if cv2.contourArea(cnt) >= cfg.grassland_min_area_px:
                cv2.drawContours(filtered, [cnt], -1, 255, -1)

        return filtered

    def _extract_dl(self, data: np.ndarray, target: str) -> np.ndarray:
        """深度学习提取: DeepLabV3+ 语义分割"""
        model = self._load_dl_model()
        if model is None:
            print("[DL] 模型不可用，回退到 CV 方法")
            return self._extract_cv(data, target)

        # 分块处理
        tile_size = self.config.dl_tile_size
        _, h, w = data.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        img = np.transpose(data[:3], (1, 2, 0))
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        import torch
        model.eval()
        with torch.no_grad():
            for row in range(0, h, tile_size):
                for col in range(0, w, tile_size):
                    r_end = min(row + tile_size, h)
                    c_end = min(col + tile_size, w)
                    tile = img[row:r_end, col:c_end]

                    # 补全到 tile_size
                    pad_h = tile_size - tile.shape[0]
                    pad_w = tile_size - tile.shape[1]
                    if pad_h > 0 or pad_w > 0:
                        tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)))

                    # 推理
                    tensor = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    output = model(tensor)
                    pred = output.argmax(dim=1).squeeze(0).numpy()

                    # 裁剪回原始大小
                    tile_mask = pred[:r_end - row, :c_end - col]

                    # 映射类别
                    class_mask = self._map_dl_classes(tile_mask, target)
                    mask[row:r_end, col:c_end] = np.maximum(
                        mask[row:r_end, col:c_end], class_mask
                    )

        return mask

    def _load_dl_model(self):
        """加载深度学习模型"""
        if self._dl_model is not None:
            return self._dl_model

        try:
            import torch
            import segmentation_models_pytorch as smp

            # 使用 ADE20K 预训练的 DeepLabV3+
            # ADE20K 有 150 类，包含建筑、树、草等
            model = smp.DeepLabV3Plus(
                encoder_name=self.config.dl_backbone,
                encoder_weights="imagenet",
                in_channels=3,
                classes=150,
            )
            model.eval()
            self._dl_model = model
            print(f"[DL] 加载 DeepLabV3+ ({self.config.dl_backbone}) 成功")
            return model
        except Exception as e:
            print(f"[DL] 模型加载失败: {e}")
            return None

    # ADE20K 类别映射 (部分关键类别)
    _ADE20K_BUILDING = [6]      # building, edifice
    _ADE20K_TREE = [14, 74]     # tree, palm
    _ADE20K_GRASS = [9, 59]     # grass, lawn

    def _map_dl_classes(self, pred: np.ndarray, target: str) -> np.ndarray:
        """将 DeepLabV3+ (ADE20K) 预测映射到目标类别"""
        if target == "building":
            classes = self._ADE20K_BUILDING
        elif target == "forest":
            classes = self._ADE20K_TREE
        elif target == "grassland":
            classes = self._ADE20K_GRASS
        else:
            return np.zeros_like(pred, dtype=np.uint8)

        mask = np.zeros_like(pred, dtype=np.uint8)
        for cls_id in classes:
            mask[pred == cls_id] = 255
        return mask

    # ══════════════════════════════════════════════════
    #  矢量化
    # ══════════════════════════════════════════════════

    def _read_region(self, bounds_latlon: dict) -> Tuple[Optional[np.ndarray], Any]:
        """读取区域影像并返回数据和变换矩阵"""
        # 经纬度 → 像素坐标
        left_px = int((bounds_latlon["left"] - self.bounds.left) / self.res[0])
        top_px = int((self.bounds.top - bounds_latlon["top"]) / self.res[1])
        right_px = int((bounds_latlon["right"] - self.bounds.left) / self.res[0])
        bottom_px = int((self.bounds.top - bounds_latlon["bottom"]) / self.res[1])

        col_off = max(0, min(left_px, right_px))
        row_off = max(0, min(top_px, bottom_px))
        w = abs(right_px - left_px)
        h = abs(bottom_px - top_px)

        if w < 1 or h < 1:
            return None, None

        # 限制范围
        col_off = min(col_off, self.width - 1)
        row_off = min(row_off, self.height - 1)
        w = min(w, self.width - col_off)
        h = min(h, self.height - row_off)

        # 限制最大读取尺寸 (防止内存溢出)
        MAX_DIM = 4096
        scale_factor = 1.0
        if w > MAX_DIM or h > MAX_DIM:
            scale_factor = min(MAX_DIM / w, MAX_DIM / h)
            read_w = int(w * scale_factor)
            read_h = int(h * scale_factor)
        else:
            read_w, read_h = w, h

        window = Window(col_off, row_off, w, h)
        with rasterio.open(self.tif_path) as ds:
            # 读取完整区域但缩放到固定尺寸
            data = ds.read(
                window=window,
                out_shape=(ds.count, read_h, read_w),
                resampling=rasterio.enums.Resampling.bilinear,
            )
            # 计算缩放后的变换矩阵
            scaled_transform = ds.window_transform(
                Window(col_off, row_off, read_w, read_h)
            )
            if scale_factor < 1.0:
                from rasterio.transform import Affine
                sx = w / read_w
                sy = h / read_h
                t = ds.window_transform(Window(col_off, row_off, 1, 1))
                scaled_transform = Affine(
                    t.a * sx, t.b, t.c,
                    t.d, t.e * sy, t.f,
                )
            window_transform = scaled_transform

        print(f"[提取] 读取区域: {w}x{h} → {read_w}x{read_h} 像素 (scale={scale_factor:.2f})")
        return data, window_transform

    def _vectorize_mask(self, mask: np.ndarray, transform,
                        target: str) -> gpd.GeoDataFrame:
        """将二值 mask 转为 GeoDataFrame"""
        mask_uint8 = (mask > 128).astype(np.uint8)

        polygons = []
        for geom, value in rasterio_shapes(mask_uint8, transform=transform):
            if value == 1:
                poly = shapely_shape(geom)
                if poly.is_valid and poly.area > 0:
                    polygons.append(poly)

        # 简化
        if self.config.simplify_tolerance > 0:
            polygons = [p.simplify(self.config.simplify_tolerance) for p in polygons]

        # 投影到 UTM 计算面积
        try:
            from pyproj import Transformer
            if self.crs and self.crs.is_geographic:
                utm_crs = self._estimate_utm_crs()
                transformer = Transformer.from_crs(
                    self.crs, utm_crs, always_xy=True
                )
                projected = []
                for p in polygons:
                    proj_p = shapely_transform(transformer, p)
                    projected.append((p, proj_p.area))
            else:
                projected = [(p, p.area) for p in polygons]
        except Exception:
            projected = [(p, p.area) for p in polygons]

        # 过滤小面积
        min_area = self.config.min_polygon_area
        records = []
        for orig_poly, area_m2 in projected:
            if area_m2 >= min_area:
                records.append({
                    "class": target,
                    "area_m2": round(area_m2, 2),
                    "geometry": orig_poly,
                })

        gdf = gpd.GeoDataFrame(records, crs=self.crs)
        print(f"[矢量化] {target}: {len(gdf)} 个多边形")
        return gdf

    def _estimate_utm_crs(self):
        """估算 UTM CRS"""
        center_lon = (self.bounds.left + self.bounds.right) / 2
        center_lat = (self.bounds.top + self.bounds.bottom) / 2
        zone = int((center_lon + 180) / 6) + 1
        if center_lat >= 0:
            return rasterio.crs.CRS.from_epsg(32600 + zone)
        else:
            return rasterio.crs.CRS.from_epsg(32700 + zone)

    def get_results_geojson(self, target: str = None) -> dict:
        """获取结果的 GeoJSON (WGS84)"""
        if target:
            gdf = self.results.get(target)
            if gdf is None or len(gdf) == 0:
                return {"type": "FeatureCollection", "features": []}
            return gdf.to_crs("EPSG:4326").__geo_interface__

        # 合并所有类别
        all_gdfs = []
        for t, gdf in self.results.items():
            if len(gdf) > 0:
                all_gdfs.append(gdf)

        if not all_gdfs:
            return {"type": "FeatureCollection", "features": []}

        merged = gpd.GeoDataFrame(
            pd.concat(all_gdfs, ignore_index=True), crs=self.crs
        )
        return merged.to_crs("EPSG:4326").__geo_interface__


def shapely_transform(transformer, geom):
    """使用 pyproj Transformer 转换 Shapely 几何的坐标"""
    from shapely.ops import transform
    return transform(lambda x, y: transformer.transform(x, y), geom)
