"""
Mask 矢量化模块

将 SAM 输出的 Raster Mask 转换为 Polygon 矢量数据，
支持输出为 GeoJSON、Shapefile、GeoPackage、GeoParquet 等格式。
"""

import os
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path


class MaskVectorizer:
    """
    Mask 矢量化工具类。

    将栅格 Mask 转为矢量 Polygon，并保留地理参考信息。

    典型用法:
        >>> vectorizer = MaskVectorizer()
        >>> gdf = vectorizer.vectorize(mask, reference_image="image.tif")
        >>> vectorizer.save("output/polygons.geojson")
    """

    def __init__(self):
        self._gdf = None
        self._polygons = []
        self._polygon_pixel_areas = []
        self._reference_crs = None
        self._reference_transform = None

    def vectorize(
        self,
        mask: np.ndarray,
        reference_image: Optional[str] = None,
        min_area: int = 10,
        simplify_tolerance: float = 0,
    ) -> Any:
        """
        将 Mask 转为矢量 Polygon。

        Args:
            mask: 二值 Mask 数组 (bool 或 uint8)
            reference_image: 参考影像路径（用于获取地理参考）
            min_area: 最小多边形面积（像素数），低于此值的多边形将被过滤
            simplify_tolerance: 简化容差（0 为不简化）

        Returns:
            GeoDataFrame 或包含多边形的字典列表
        """
        # 确保 mask 是二值的
        if mask.dtype == bool:
            mask_uint8 = mask.astype("uint8")
        elif mask.max() > 1:
            mask_uint8 = (mask > 128).astype("uint8")
        else:
            mask_uint8 = mask.astype("uint8")

        # 获取地理参考
        transform = None
        crs = None
        if reference_image:
            try:
                import rasterio
                with rasterio.open(reference_image) as src:
                    transform = src.transform
                    crs = src.crs
                    self._reference_crs = crs
                    self._reference_transform = transform
            except ImportError:
                print("[矢量化] rasterio 未安装，将不保留地理参考")
            except Exception as e:
                print(f"[矢量化] 读取地理参考失败: {e}")

        # 栅格转矢量
        try:
            from rasterio.features import shapes
            from shapely.geometry import shape

            polygons = []
            polygon_pixel_areas = []
            pixel_area_units = 1.0
            if transform is not None:
                pixel_area_units = abs(transform.a * transform.e)
                if pixel_area_units == 0:
                    pixel_area_units = 1.0

            for geom, value in shapes(mask_uint8, transform=transform):
                if value == 1:
                    poly = shape(geom)
                    pixel_area = poly.area / pixel_area_units
                    if pixel_area >= min_area:
                        polygons.append(poly)
                        polygon_pixel_areas.append(pixel_area)

            print(f"[矢量化] 生成 {len(polygons)} 个多边形 (过滤 min_area={min_area})")

        except ImportError:
            print("[矢量化] rasterio/shapely 未安装，使用简化矢量化")
            polygons = self._simple_vectorize(mask_uint8, min_area)
            polygon_pixel_areas = [poly.area for poly in polygons]

        self._polygons = polygons
        self._polygon_pixel_areas = polygon_pixel_areas

        # 简化多边形
        if simplify_tolerance > 0:
            self._polygons = [p.simplify(simplify_tolerance) for p in self._polygons]
            print(f"[矢量化] 多边形简化: tolerance={simplify_tolerance}")

        # 尝试创建 GeoDataFrame
        try:
            import geopandas as gpd
            from shapely.geometry import Polygon

            if len(self._polygons) == 0:
                print("[矢量化] 未生成任何多边形，返回空 GeoDataFrame")
                self._gdf = gpd.GeoDataFrame(
                    columns=["id", "area_pixels", "geometry"],
                    geometry="geometry",
                    crs=crs,
                )
                return self._gdf

            records = []
            for i, poly in enumerate(self._polygons):
                records.append({
                    "id": i,
                    "area_pixels": self._polygon_pixel_areas[i],
                    "geometry": poly,
                })

            self._gdf = gpd.GeoDataFrame(records, crs=crs)
            print(f"[矢量化] GeoDataFrame 创建完成, CRS: {crs}")
            return self._gdf

        except ImportError:
            print("[矢量化] geopandas 未安装，返回多边形列表")
            return self._polygons

    def _simple_vectorize(
        self,
        mask: np.ndarray,
        min_area: int,
    ) -> list:
        """简化版矢量化（不依赖 rasterio）。"""
        try:
            from scipy.ndimage import label
            from shapely.geometry import Polygon

            labeled, num_features = label(mask)
            polygons = []

            for i in range(1, num_features + 1):
                component = labeled == i
                if component.sum() < min_area:
                    continue

                # 获取边界坐标
                coords = np.argwhere(component)
                if len(coords) < 3:
                    continue

                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0) + 1

                # 创建简单的边界矩形多边形
                poly = Polygon([
                    (x_min, y_min), (x_max, y_min),
                    (x_max, y_max), (x_min, y_max),
                    (x_min, y_min),
                ])
                polygons.append(poly)

            return polygons

        except ImportError:
            print("[矢量化] scipy/shapely 均未安装，无法矢量化")
            return []

    def save(
        self,
        output_path: str,
        driver: Optional[str] = None,
    ) -> str:
        """
        保存矢量结果。

        Args:
            output_path: 输出文件路径
            driver: 驱动类型，None 时根据扩展名自动选择:
                - .geojson → GeoJSON
                - .shp → ESRI Shapefile
                - .gpkg → GeoPackage
                - .parquet → GeoParquet

        Returns:
            实际保存路径
        """
        if self._gdf is None and not self._polygons:
            raise RuntimeError("没有可保存的矢量数据，请先调用 vectorize()")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        if driver is None:
            ext = Path(output_path).suffix.lower()
            driver_map = {
                ".geojson": "GeoJSON",
                ".json": "GeoJSON",
                ".shp": "ESRI Shapefile",
                ".gpkg": "GPKG",
                ".parquet": "GeoParquet",
                ".fgb": "FlatGeobuf",
            }
            driver = driver_map.get(ext, "GeoJSON")

        if self._gdf is not None:
            try:
                self._gdf.to_file(output_path, driver=driver)
            except Exception:
                # 回退: 尝试不同驱动
                if driver == "GeoParquet":
                    try:
                        self._gdf.to_parquet(output_path)
                    except Exception as e:
                        # 最终回退到 GeoJSON
                        output_path = output_path.rsplit(".", 1)[0] + ".geojson"
                        self._gdf.to_file(output_path, driver="GeoJSON")
                else:
                    output_path = output_path.rsplit(".", 1)[0] + ".geojson"
                    self._gdf.to_file(output_path, driver="GeoJSON")
        else:
            # 无 geopandas，手动保存为 GeoJSON
            import json
            features = []
            for i, poly in enumerate(self._polygons):
                features.append({
                    "type": "Feature",
                    "properties": {"id": i, "area": poly.area},
                    "geometry": poly.__geo_interface__,
                })

            geojson = {"type": "FeatureCollection", "features": features}
            output_path = output_path.rsplit(".", 1)[0] + ".geojson"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(geojson, f, indent=2, ensure_ascii=False)

        print(f"[矢量化] 已保存: {output_path} (驱动: {driver})")
        return output_path

    def get_polygon_count(self) -> int:
        """获取多边形数量。"""
        return len(self._polygons)

    def get_total_area(self) -> float:
        """获取所有多边形的总面积。"""
        return sum(p.area for p in self._polygons)

    def get_gdf(self):
        """获取 GeoDataFrame。"""
        return self._gdf

    def add_attribute(self, name: str, values: list) -> "MaskVectorizer":
        """
        为多边形添加属性字段。

        Args:
            name: 字段名
            values: 属性值列表

        Returns:
            self
        """
        if self._gdf is not None:
            self._gdf[name] = values
            print(f"[矢量化] 添加属性字段: {name}")
        return self

    def filter_by_area(
        self,
        min_area: Optional[float] = None,
        max_area: Optional[float] = None,
    ) -> "MaskVectorizer":
        """
        按面积过滤多边形。

        Args:
            min_area: 最小面积
            max_area: 最大面积

        Returns:
            self
        """
        if self._gdf is not None:
            mask = np.ones(len(self._gdf), dtype=bool)
            if min_area is not None:
                mask &= self._gdf["area_pixels"] >= min_area
            if max_area is not None:
                mask &= self._gdf["area_pixels"] <= max_area
            self._gdf = self._gdf[mask].reset_index(drop=True)
            print(f"[矢量化] 面积过滤后剩余: {len(self._gdf)} 个多边形")
        return self

    def __repr__(self) -> str:
        count = len(self._polygons)
        return f"MaskVectorizer(polygons={count}, crs={self._reference_crs})"
