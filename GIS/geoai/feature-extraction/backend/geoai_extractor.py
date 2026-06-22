"""
GeoAI 要素提取引擎（基于 geoai-py 库）
=========================================
使用 geoai 库提供的深度学习模型进行遥感影像要素提取：
  - 建筑提取：BuildingFootprintExtractor（预训练建筑轮廓模型）
  - 林地/草地提取：GroundedSAM（Grounding DINO + SAM 零样本分割）
  - 备选方案：semantic_segmentation 语义分割

依赖：geoai-py >= 0.40.1, numpy < 2
"""

import os
import sys
import json
import time
import tempfile
import warnings
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes as rasterio_shapes
from shapely.geometry import shape as shapely_shape
from shapely.ops import transform as shapely_transform
import geopandas as gpd
import pandas as pd

# ══════════════════════════════════════════════════
#  geoai-py 库导入（核心依赖）
# ══════════════════════════════════════════════════
try:
    import geoai
    from geoai.extract import BuildingFootprintExtractor, ObjectDetector
    from geoai import GroundedSAM

    GEOAI_AVAILABLE = True
    GEOAI_VERSION = getattr(geoai, "__version__", "unknown")
    print(f"[GeoAI] geoai-py {GEOAI_VERSION} 加载成功")
except ImportError as e:
    GEOAI_AVAILABLE = False
    GEOAI_VERSION = None
    print(f"[GeoAI] 警告: geoai-py 库不可用 ({e})，将回退到 HSV 方法")
    print("[GeoAI] 安装方式: pip install geoai-py \"numpy<2\"")

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class GeoAIConfig:
    """GeoAI 库提取配置"""
    # 提取目标
    targets: List[str] = field(default_factory=lambda: ["building"])
    # 推理设备: cpu / cuda:0 / auto
    device: str = "cpu"
    # 建筑提取参数
    building_model_path: str = "building_footprints_usa.pth"
    building_repo_id: Optional[str] = None
    building_batch_size: int = 4
    building_confidence: float = 0.5
    building_filter_edges: bool = True
    building_edge_buffer: int = 20
    building_regularize: bool = True
    building_simplify_tolerance: float = 1.0
    building_min_area: float = 10.0  # 平方米
    # GroundedSAM 参数（林地/草地）
    grounded_sam_detector_id: str = "IDEA-Research/grounding-dino-tiny"
    grounded_sam_segmenter_id: str = "facebook/sam-vit-base"
    grounded_sam_tile_size: int = 1024
    grounded_sam_overlap: int = 128
    grounded_sam_threshold: float = 0.3
    # 文本提示词（用于 GroundedSAM 零样本分割）
    forest_text_prompt: str = "trees. forest. woodland. dense vegetation."
    grassland_text_prompt: str = "grass. lawn. meadow. grassland."
    # 通用参数
    min_polygon_area: float = 10.0  # 平方米
    simplify_tolerance: float = 1.0
    # ROI 裁剪后的临时文件最大尺寸
    max_clip_dimension: int = 4096


class GeoAILibraryExtractor:
    """
    基于 geoai-py 库的要素提取器

    核心流程:
    1. 根据 ROI 裁剪 TIF 为临时文件（若全图则直接使用原文件）
    2. 调用 BuildingFootprintExtractor 提取建筑轮廓
    3. 调用 GroundedSAM 进行零样本分割（林地/草地）
    4. 矢量化、面积过滤、正则化
    5. 返回 GeoDataFrame
    """

    def __init__(self, tif_path: str, config: GeoAIConfig = None):
        self.tif_path = tif_path
        self.config = config or GeoAIConfig()
        self.results: Dict[str, gpd.GeoDataFrame] = {}
        self.stats: Dict[str, Any] = {}
        self._tmp_dir = None

        # 读取 TIF 元数据
        with rasterio.open(tif_path) as ds:
            self.crs = ds.crs
            self.transform = ds.transform
            self.width = ds.width
            self.height = ds.height
            self.bounds = ds.bounds
            self.res = ds.res
            self.count = ds.count

        # 模型缓存
        self._building_extractor = None
        self._grounded_sam = None

    def _ensure_tmp_dir(self):
        """确保临时目录存在"""
        if self._tmp_dir is None or not os.path.exists(self._tmp_dir):
            self._tmp_dir = tempfile.mkdtemp(prefix="geoai_clip_")
        return self._tmp_dir

    def extract(
        self,
        bounds_latlon: Optional[dict] = None,
        progress_callback=None,
    ) -> Dict[str, gpd.GeoDataFrame]:
        """
        执行提取

        Args:
            bounds_latlon: ROI 经纬度范围 {"left","bottom","right","top"}
                           None 表示使用全图
            progress_callback: fn(stage, pct) 进度回调
        """
        start = time.time()
        self.results = {}
        cfg = self.config

        # 1. 准备影像文件
        self._ensure_tmp_dir()  # 确保临时目录存在
        if progress_callback:
            progress_callback("preparing", 5)

        if bounds_latlon:
            raster_path = self._clip_to_roi(bounds_latlon)
            if raster_path is None:
                print("[GeoAI] ROI 裁剪失败，使用全图")
                raster_path = self.tif_path
        else:
            raster_path = self.tif_path

        print(f"[GeoAI] 输入影像: {raster_path}")
        print(f"[GeoAI] 提取目标: {cfg.targets}")
        print(f"[GeoAI] 推理设备: {cfg.device}")

        # 2. 逐目标提取
        for i, target in enumerate(cfg.targets):
            pct_base = int(10 + (i / len(cfg.targets)) * 80)
            if progress_callback:
                progress_callback(f"extracting_{target}", pct_base)

            try:
                if target == "building":
                    gdf = self._extract_building(raster_path)
                elif target in ("forest", "grassland"):
                    gdf = self._extract_land_cover(raster_path, target)
                else:
                    print(f"[GeoAI] 不支持的目标: {target}，跳过")
                    gdf = gpd.GeoDataFrame(
                        columns=["class", "area_m2", "geometry"],
                        geometry="geometry",
                        crs=self.crs,
                    )

                self.results[target] = gdf
                print(f"[GeoAI] {target}: 提取 {len(gdf)} 个要素")

            except Exception as e:
                print(f"[GeoAI] {target} 提取失败: {e}")
                import traceback
                traceback.print_exc()
                self.results[target] = gpd.GeoDataFrame(
                    columns=["class", "area_m2", "geometry"],
                    geometry="geometry",
                    crs=self.crs,
                )

        # 3. 统计
        elapsed = time.time() - start
        self.stats = {
            "elapsed_seconds": round(elapsed, 2),
            "method": "geoai",
            "library": "geoai-py",
            "device": cfg.device,
            "targets": {},
        }
        for target, gdf in self.results.items():
            self.stats["targets"][target] = {
                "count": len(gdf),
                "total_area_m2": round(gdf["area_m2"].sum(), 2) if len(gdf) > 0 else 0,
            }

        if progress_callback:
            progress_callback("done", 100)

        # 4. 清理临时文件
        self._cleanup_tmp()

        return self.results

    # ══════════════════════════════════════════════════
    #  建筑提取 — BuildingFootprintExtractor
    # ══════════════════════════════════════════════════

    def _extract_building(self, raster_path: str) -> gpd.GeoDataFrame:
        """使用 geoai.BuildingFootprintExtractor 提取建筑轮廓"""
        cfg = self.config

        if not GEOAI_AVAILABLE:
            print("[GeoAI] geoai-py 不可用，回退到 HSV 方法")
            return self._fallback_hsv_extract(raster_path, "building")

        # 懒加载模型
        if self._building_extractor is None:
            print("[GeoAI] 正在加载 BuildingFootprintExtractor 模型...")
            self._building_extractor = BuildingFootprintExtractor(
                model_path=cfg.building_model_path,
                repo_id=cfg.building_repo_id,
                device=cfg.device,
            )
            print("[GeoAI] BuildingFootprintExtractor 模型加载完成")

        extractor = self._building_extractor

        # 方法1：直接 process_raster（推荐，一步到位）
        print("[GeoAI] 正在提取建筑轮廓 (process_raster)...")
        gdf = extractor.process_raster(
            raster_path,
            batch_size=cfg.building_batch_size,
            filter_edges=cfg.building_filter_edges,
            edge_buffer=cfg.building_edge_buffer,
        )

        if gdf is None or len(gdf) == 0:
            print("[GeoAI] process_raster 返回空结果，尝试 generate_masks + masks_to_vector")
            return self._extract_building_via_masks(raster_path)

        # 确保 CRS 正确
        if gdf.crs is None:
            gdf = gdf.set_crs(self.crs)

        # 正则化（直角化建筑轮廓）
        if cfg.building_regularize and len(gdf) > 0:
            try:
                print("[GeoAI] 正在正则化建筑轮廓...")
                gdf = extractor.regularize_buildings(
                    gdf,
                    min_area=int(cfg.building_min_area),
                    angle_threshold=15,
                    rectangularity_threshold=0.7,
                )
            except Exception as e:
                print(f"[GeoAI] 正则化失败（保留原始结果）: {e}")

        # 简化
        if cfg.building_simplify_tolerance > 0 and len(gdf) > 0:
            gdf = gdf.copy()
            gdf["geometry"] = gdf["geometry"].simplify(cfg.building_simplify_tolerance)

        # 计算面积并过滤
        gdf = self._add_area_and_filter(gdf, "building", cfg.building_min_area)

        return gdf

    def _extract_building_via_masks(self, raster_path: str) -> gpd.GeoDataFrame:
        """备选方案：通过 mask 中间步骤提取建筑"""
        cfg = self.config
        extractor = self._building_extractor

        # 生成 mask
        print("[GeoAI] 正在生成建筑 mask...")
        mask_path = extractor.generate_masks(
            raster_path,
            min_object_area=10,
            overlap=0.25,
            batch_size=cfg.building_batch_size,
            verbose=True,
        )

        if not os.path.exists(mask_path):
            print(f"[GeoAI] mask 文件不存在: {mask_path}")
            return self._empty_gdf()

        # mask → 矢量
        print("[GeoAI] 正在将 mask 转为矢量...")
        gdf = extractor.masks_to_vector(
            mask_path,
            simplify_tolerance=cfg.building_simplify_tolerance,
            min_object_area=int(cfg.building_min_area),
            regularize=cfg.building_regularize,
        )

        if gdf.crs is None:
            gdf = gdf.set_crs(self.crs)

        # 面积过滤
        gdf = self._add_area_and_filter(gdf, "building", cfg.building_min_area)

        return gdf

    # ══════════════════════════════════════════════════
    #  林地/草地提取 — GroundedSAM 零样本分割
    # ══════════════════════════════════════════════════

    def _extract_land_cover(self, raster_path: str, target: str) -> gpd.GeoDataFrame:
        """
        使用 GroundedSAM（Grounding DINO + SAM）进行零样本分割
        
        原理：通过文本提示词引导 Grounding DINO 检测目标区域，
        然后 SAM 对检测区域进行精细分割，最终生成矢量多边形。
        """
        cfg = self.config

        # 选择文本提示词
        if target == "forest":
            text_prompt = cfg.forest_text_prompt
        elif target == "grassland":
            text_prompt = cfg.grassland_text_prompt
        else:
            text_prompt = f"{target}."

        # 懒加载 GroundedSAM
        if self._grounded_sam is None:
            if not GEOAI_AVAILABLE:
                print("[GeoAI] geoai-py 不可用，回退到 HSV 方法")
                return self._fallback_hsv_extract(raster_path, target)

            print("[GeoAI] 正在加载 GroundedSAM 模型...")
            try:
                self._grounded_sam = GroundedSAM(
                    detector_id=cfg.grounded_sam_detector_id,
                    segmenter_id=cfg.grounded_sam_segmenter_id,
                    device=cfg.device,
                    tile_size=cfg.grounded_sam_tile_size,
                    overlap=cfg.grounded_sam_overlap,
                    threshold=cfg.grounded_sam_threshold,
                )
                print("[GeoAI] GroundedSAM 模型加载完成")
            except Exception as e:
                print(f"[GeoAI] GroundedSAM 加载失败: {e}")
                print("[GeoAI] 回退到 HSV 颜色提取方法")
                return self._fallback_hsv_extract(raster_path, target)

        # 准备输出路径
        output_path = os.path.join(self._tmp_dir, f"{target}_segmentation.tif")

        # 执行分割
        print(f"[GeoAI] 正在用 GroundedSAM 分割 {target}...")
        print(f"[GeoAI] 文本提示: {text_prompt}")

        try:
            result = self._grounded_sam.segment_image(
                input_path=raster_path,
                output_path=output_path,
                text_prompts=text_prompt,
                export_polygons=True,
                min_polygon_area=max(50, int(cfg.min_polygon_area)),
                simplify_tolerance=cfg.simplify_tolerance,
                smoothing_sigma=1.0,
                nms_threshold=0.5,
            )

            # result 是一个 dict，可能包含输出路径
            if isinstance(result, dict):
                print(f"[GeoAI] 分割结果: {result}")
                # 尝试从结果中获取输出路径
                actual_output = result.get("output_path", output_path)
                if os.path.exists(actual_output):
                    output_path = actual_output

        except Exception as e:
            print(f"[GeoAI] GroundedSAM 分割失败: {e}")
            import traceback
            traceback.print_exc()
            print("[GeoAI] 回退到 HSV 颜色提取方法")
            return self._fallback_hsv_extract(raster_path, target)

        # 矢量化分割结果
        if os.path.exists(output_path):
            return self._vectorize_segmentation(output_path, target)
        else:
            print(f"[GeoAI] 分割输出文件不存在: {output_path}")
            return self._fallback_hsv_extract(raster_path, target)

    def _vectorize_segmentation(
        self, seg_raster_path: str, target: str
    ) -> gpd.GeoDataFrame:
        """将分割结果栅格转为矢量"""
        polygons = []

        try:
            with rasterio.open(seg_raster_path) as ds:
                # 读取分割结果（通常是单波段，非零值=目标）
                data = ds.read(1)
                seg_transform = ds.transform
                seg_crs = ds.crs or self.crs

                # 二值化
                binary = (data > 0).astype(np.uint8)

                # 矢量化
                for geom, value in rasterio_shapes(binary, transform=seg_transform):
                    if value == 1:
                        poly = shapely_shape(geom)
                        if poly.is_valid and poly.area > 0:
                            polygons.append(poly)

            print(f"[GeoAI] {target} 矢量化: {len(polygons)} 个原始多边形")

        except Exception as e:
            print(f"[GeoAI] 分割结果矢量化失败: {e}")
            return self._empty_gdf()

        if not polygons:
            return self._empty_gdf()

        # 构建 GeoDataFrame
        records = [{"class": target, "geometry": p} for p in polygons]
        gdf = gpd.GeoDataFrame(records, crs=seg_crs)

        # 简化
        if self.config.simplify_tolerance > 0:
            gdf = gdf.copy()
            gdf["geometry"] = gdf["geometry"].simplify(self.config.simplify_tolerance)

        # 面积计算与过滤
        gdf = self._add_area_and_filter(gdf, target, self.config.min_polygon_area)

        return gdf

    # ══════════════════════════════════════════════════
    #  备选方案：HSV 颜色提取
    # ══════════════════════════════════════════════════

    def _fallback_hsv_extract(
        self, raster_path: str, target: str
    ) -> gpd.GeoDataFrame:
        """当 GroundedSAM 不可用时的 HSV 回退方案"""
        import cv2

        print(f"[GeoAI] 使用 HSV 回退方案提取 {target}...")

        try:
            with rasterio.open(raster_path) as ds:
                data = ds.read()
                raster_transform = ds.transform

            # (bands, h, w) → BGR
            img = np.transpose(data[:3], (1, 2, 0))
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

            if target == "forest":
                lower = np.array([35, 40, 40])
                upper = np.array([85, 255, 255])
                min_area_px = 200
            elif target == "grassland":
                lower = np.array([25, 20, 60])
                upper = np.array([75, 120, 200])
                min_area_px = 300
            else:
                return self._empty_gdf()

            mask = cv2.inRange(hsv, lower, upper)

            # 形态学降噪
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            # 矢量化
            polygons = []
            for geom, value in rasterio_shapes(mask, transform=raster_transform):
                if value == 255:
                    poly = shapely_shape(geom)
                    if poly.is_valid and poly.area > 0:
                        polygons.append(poly)

            # 构建 GeoDataFrame
            records = [{"class": target, "geometry": p} for p in polygons]
            gdf = gpd.GeoDataFrame(records, crs=self.crs)

            # 简化
            if self.config.simplify_tolerance > 0:
                gdf["geometry"] = gdf["geometry"].simplify(self.config.simplify_tolerance)

            # 面积过滤
            gdf = self._add_area_and_filter(gdf, target, self.config.min_polygon_area)

            print(f"[GeoAI] HSV 回退 {target}: {len(gdf)} 个要素")
            return gdf

        except Exception as e:
            print(f"[GeoAI] HSV 回退失败: {e}")
            return self._empty_gdf()

    # ══════════════════════════════════════════════════
    #  ROI 裁剪
    # ══════════════════════════════════════════════════

    def _clip_to_roi(self, bounds_latlon: dict) -> Optional[str]:
        """将 TIF 裁剪到 ROI 范围，返回临时文件路径"""
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
            return None

        # 限制范围
        col_off = min(col_off, self.width - 1)
        row_off = min(row_off, self.height - 1)
        w = min(w, self.width - col_off)
        h = min(h, self.height - row_off)

        # 缩放过大尺寸
        max_dim = self.config.max_clip_dimension
        out_w, out_h = w, h
        scale = 1.0
        if w > max_dim or h > max_dim:
            scale = min(max_dim / w, max_dim / h)
            out_w = int(w * scale)
            out_h = int(h * scale)

        window = Window(col_off, row_off, w, h)

        # 写入临时 TIF
        clip_path = os.path.join(self._tmp_dir, "roi_clip.tif")

        try:
            with rasterio.open(self.tif_path) as src:
                # 读取数据（可能缩放）
                data = src.read(
                    window=window,
                    out_shape=(src.count, out_h, out_w),
                    resampling=rasterio.enums.Resampling.bilinear,
                )

                # 计算缩放后的变换
                win_transform = src.window_transform(window)
                if scale < 1.0:
                    from rasterio.transform import Affine
                    sx = w / out_w
                    sy = h / out_h
                    win_transform = Affine(
                        win_transform.a * sx,
                        win_transform.b,
                        win_transform.c,
                        win_transform.d,
                        win_transform.e * sy,
                        win_transform.f,
                    )

                # 写出
                profile = src.profile.copy()
                profile.update(
                    width=out_w,
                    height=out_h,
                    transform=win_transform,
                    count=src.count,
                    dtype=data.dtype,
                )

                with rasterio.open(clip_path, "w", **profile) as dst:
                    dst.write(data)

            file_size_mb = os.path.getsize(clip_path) / 1024 / 1024
            print(f"[GeoAI] ROI 裁剪完成: {out_w}x{out_h} ({file_size_mb:.1f} MB)")
            return clip_path

        except Exception as e:
            print(f"[GeoAI] ROI 裁剪失败: {e}")
            return None

    # ══════════════════════════════════════════════════
    #  工具方法
    # ══════════════════════════════════════════════════

    def _add_area_and_filter(
        self, gdf: gpd.GeoDataFrame, target: str, min_area_m2: float
    ) -> gpd.GeoDataFrame:
        """计算面积（平方米）并过滤小面积要素"""
        if len(gdf) == 0:
            gdf["area_m2"] = pd.Series(dtype=float)
            return gdf

        # 投影到 UTM 计算面积
        try:
            from pyproj import Transformer

            if gdf.crs and gdf.crs.is_geographic:
                utm_crs = self._estimate_utm_crs()
                transformer = Transformer.from_crs(gdf.crs, utm_crs, always_xy=True)
                areas = []
                for geom in gdf.geometry:
                    proj_geom = shapely_transform(
                        lambda x, y: transformer.transform(x, y), geom
                    )
                    areas.append(proj_geom.area)
                gdf = gdf.copy()
                gdf["area_m2"] = areas
            else:
                gdf = gdf.copy()
                gdf["area_m2"] = gdf.geometry.area
        except Exception:
            gdf = gdf.copy()
            gdf["area_m2"] = gdf.geometry.area

        gdf["area_m2"] = gdf["area_m2"].round(2)
        before = len(gdf)
        gdf = gdf[gdf["area_m2"] >= min_area_m2].reset_index(drop=True)
        if before > len(gdf):
            print(f"[GeoAI] {target} 面积过滤: {before} → {len(gdf)}")

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

    def _empty_gdf(self) -> gpd.GeoDataFrame:
        """返回空 GeoDataFrame"""
        return gpd.GeoDataFrame(
            columns=["class", "area_m2", "geometry"],
            geometry="geometry",
            crs=self.crs,
        )

    def _cleanup_tmp(self):
        """清理临时文件"""
        try:
            import shutil
            if os.path.exists(self._tmp_dir):
                shutil.rmtree(self._tmp_dir, ignore_errors=True)
        except Exception:
            pass

    def get_results_geojson(self, target: str = None) -> dict:
        """获取结果的 GeoJSON (WGS84)"""
        if target:
            gdf = self.results.get(target)
            if gdf is None or len(gdf) == 0:
                return {"type": "FeatureCollection", "features": []}
            return json.loads(gdf.to_crs("EPSG:4326").to_json())

        all_gdfs = [gdf for gdf in self.results.values() if len(gdf) > 0]
        if not all_gdfs:
            return {"type": "FeatureCollection", "features": []}

        merged = gpd.GeoDataFrame(
            pd.concat(all_gdfs, ignore_index=True), crs=self.crs
        )
        return json.loads(merged.to_crs("EPSG:4326").to_json())
