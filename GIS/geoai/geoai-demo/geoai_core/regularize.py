"""GeoAI Core - 矢量要素后处理 (简化、平滑、正交化、属性计算)"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

logger = logging.getLogger(__name__)


class FeatureRegularizer:
    """
    对矢量化后的地物多边形进行后处理,
    包括简化、平滑、正交化和属性计算。
    """

    def __init__(self, gdf: gpd.GeoDataFrame):
        if gdf is None or len(gdf) == 0:
            logger.warning("输入 GeoDataFrame 为空")
        self.gdf = gdf.copy() if gdf is not None else gpd.GeoDataFrame()

    def simplify(self, tolerance: float = 2.0) -> "FeatureRegularizer":
        """
        简化多边形几何 (Douglas-Peucker 算法)。

        Args:
            tolerance: 简化容差 (坐标单位)

        Returns:
            self (链式调用)
        """
        if len(self.gdf) == 0:
            return self

        before = len(self.gdf)
        self.gdf["geometry"] = self.gdf.geometry.simplify(
            tolerance, preserve_topology=True
        )
        # 移除简化后变为空的几何
        self.gdf = self.gdf[~self.gdf.geometry.is_empty].copy()
        logger.info(
            "简化完成: tolerance=%.1f, %d -> %d 要素",
            tolerance, before, len(self.gdf),
        )
        return self

    def smooth(self, iterations: int = 3) -> "FeatureRegularizer":
        """
        使用 Chaikin 算法平滑多边形边界。

        Args:
            iterations: 平滑迭代次数

        Returns:
            self (链式调用)
        """
        if len(self.gdf) == 0:
            return self

        def _chaikin_smooth(coords, iterations=1):
            """Chaikin 曲线平滑。"""
            for _ in range(iterations):
                new_coords = [coords[0]]
                for i in range(len(coords) - 1):
                    p0 = np.array(coords[i])
                    p1 = np.array(coords[i + 1])
                    q = 0.75 * p0 + 0.25 * p1
                    r = 0.25 * p0 + 0.75 * p1
                    new_coords.extend([tuple(q), tuple(r)])
                new_coords.append(coords[-1])
                coords = new_coords
            return coords

        def _smooth_polygon(geom, iters):
            if isinstance(geom, Polygon):
                ext = list(geom.exterior.coords)
                ext = _chaikin_smooth(ext, iters)
                holes = []
                for interior in geom.interiors:
                    hole = list(interior.coords)
                    hole = _chaikin_smooth(hole, iters)
                    holes.append(hole)
                try:
                    return Polygon(ext, holes)
                except Exception:
                    return geom
            elif isinstance(geom, MultiPolygon):
                polys = [_smooth_polygon(p, iters) for p in geom.geoms]
                return MultiPolygon(polys)
            return geom

        self.gdf["geometry"] = self.gdf.geometry.apply(
            lambda g: _smooth_polygon(g, iterations)
        )
        logger.info("平滑完成: %d 次迭代", iterations)
        return self

    def orthogonalize(self) -> "FeatureRegularizer":
        """
        正交化: 使建筑物角点更接近直角。

        对每个多边形, 检测角点并微调使其更接近 90/180 度。

        Returns:
            self (链式调用)
        """
        if len(self.gdf) == 0:
            return self

        def _orthogonalize_polygon(geom: Polygon) -> Polygon:
            if not isinstance(geom, Polygon) or not geom.is_valid:
                return geom

            coords = np.array(geom.exterior.coords[:-1])
            if len(coords) < 4:
                return geom

            n = len(coords)
            new_coords = coords.copy()

            for i in range(n):
                p_prev = coords[(i - 1) % n]
                p_curr = coords[i]
                p_next = coords[(i + 1) % n]

                v1 = p_prev - p_curr
                v2 = p_next - p_curr

                # 计算角度
                cos_angle = np.dot(v1, v2) / (
                    np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10
                )
                angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

                # 如果角点接近 90 度, 进行微调
                if 70 < angle < 110:
                    # 计算垂直方向
                    v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
                    v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)

                    # 调整 v2 使其与 v1 正交
                    v2_ortho = v2_norm - np.dot(v2_norm, v1_norm) * v1_norm
                    v2_ortho = v2_ortho / (np.linalg.norm(v2_ortho) + 1e-10)

                    dist = np.linalg.norm(p_next - p_curr)
                    new_coords[(i + 1) % n] = p_curr + v2_ortho * dist

            new_coords_closed = np.vstack([new_coords, new_coords[0]])
            try:
                result = Polygon(new_coords_closed)
                if result.is_valid and result.area > 0:
                    return result
            except Exception:
                pass
            return geom

        # 只对 building 类别应用正交化
        if "class_name" in self.gdf.columns:
            mask = self.gdf["class_name"] == "building"
            self.gdf.loc[mask, "geometry"] = self.gdf.loc[mask, "geometry"].apply(
                _orthogonalize_polygon
            )
            logger.info("正交化完成: %d 个建筑要素", mask.sum())
        else:
            self.gdf["geometry"] = self.gdf.geometry.apply(
                _orthogonalize_polygon
            )
            logger.info("正交化完成: %d 个要素", len(self.gdf))

        return self

    def filter_small(self, min_area: float = 50.0) -> "FeatureRegularizer":
        """
        过滤掉面积小于阈值的图斑。

        Args:
            min_area: 最小面积 (坐标单位, 如平方米)

        Returns:
            self (链式调用)
        """
        if len(self.gdf) == 0:
            return self

        before = len(self.gdf)
        areas = self.gdf.geometry.area
        self.gdf = self.gdf[areas >= min_area].copy()
        removed = before - len(self.gdf)
        logger.info(
            "面积过滤: min_area=%.1f, 移除 %d 个, 剩余 %d 个",
            min_area, removed, len(self.gdf),
        )
        return self

    def add_attributes(self) -> "FeatureRegularizer":
        """
        计算并添加面积、周长、紧凑度属性列。

        Returns:
            self (链式调用)
        """
        if len(self.gdf) == 0:
            return self

        self.gdf["area"] = self.gdf.geometry.area
        self.gdf["perimeter"] = self.gdf.geometry.length

        # 紧凑度: 4 * pi * area / perimeter^2
        perim_sq = self.gdf["perimeter"] ** 2
        perim_sq = perim_sq.replace(0, np.nan)
        self.gdf["compactness"] = (
            4 * np.pi * self.gdf["area"] / perim_sq
        ).fillna(0.0)

        logger.info("属性计算完成: area, perimeter, compactness")
        return self

    def run(self, config: dict) -> gpd.GeoDataFrame:
        """
        根据配置依次执行所有后处理步骤。

        Args:
            config: 配置字典, 可包含以下键:
                - simplify_tolerance (float): 简化容差
                - smooth_iterations (int): 平滑迭代次数
                - min_area (float): 最小面积
                - orthogonalize (bool): 是否正交化

        Returns:
            处理后的 GeoDataFrame
        """
        if len(self.gdf) == 0:
            logger.warning("无数据可处理")
            return self.gdf

        logger.info("开始后处理: %d 个要素, 配置=%s", len(self.gdf), config)

        # 1. 简化
        tol = config.get("simplify_tolerance", 0)
        if tol and tol > 0:
            self.simplify(tol)

        # 2. 平滑
        iters = config.get("smooth_iterations", 0)
        if iters and iters > 0:
            self.smooth(iters)

        # 3. 正交化
        if config.get("orthogonalize", False):
            self.orthogonalize()

        # 4. 面积过滤
        min_area = config.get("min_area", 0)
        if min_area and min_area > 0:
            self.filter_small(min_area)

        # 5. 添加属性
        self.add_attributes()

        # 修复可能无效的几何
        self.gdf["geometry"] = self.gdf.geometry.buffer(0)
        self.gdf = self.gdf[self.gdf.geometry.notnull()].copy()

        logger.info("后处理完成: %d 个要素", len(self.gdf))
        return self.gdf

    def save(self, output_path: str | Path, driver: str = "GPKG") -> Path:
        """保存结果到文件。"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.gdf.to_file(output_path, driver=driver)
        logger.info("结果已保存: %s (%d 要素)", output_path, len(self.gdf))
        return output_path
