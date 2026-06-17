"""
建筑物轮廓正则化核心算法。

实现完整的正则化流水线：
  面积过滤 → Douglas-Peucker 简化 → 主方向检测 → 直角化 → 对称化 → 边界平滑 → 拓扑检查
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString, Point
from shapely.affinity import rotate, translate
from shapely.ops import unary_union, transform as shapely_transform

try:
    import pyproj
    _HAS_PYPROJ = True
except ImportError:
    _HAS_PYPROJ = False


# ──────────────────────────────────────────────
# 坐标投影 (WGS84 ↔ UTM 米制)
# ──────────────────────────────────────────────
def _detect_utm_zone(lon: float, lat: float) -> str:
    """根据经纬度自动检测 UTM zone EPSG 代码。"""
    zone = int((lon + 180) // 6) + 1
    ns = "north" if lat >= 0 else "south"
    # EPSG: 326xx (北半球) / 327xx (南半球)
    code = 32600 + zone if ns == "north" else 32700 + zone
    return f"EPSG:{code}"


def _project_polys(polys: List[Polygon], src_crs: str = "EPSG:4326") -> Tuple[List[Polygon], str]:
    """将 WGS84 多边形投影到 UTM (米制)。返回 (投影后多边形, 目标 CRS)。"""
    if not _HAS_PYPROJ or not polys:
        return polys, src_crs

    # 用第一个多边形的质心检测 UTM zone
    c = polys[0].centroid
    dst_crs = _detect_utm_zone(c.x, c.y)
    transformer = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    projected = []
    for p in polys:
        pp = shapely_transform(transformer.transform, p)
        projected.append(pp)
    return projected, dst_crs


def _unproject_polys(polys: List[Polygon], dst_crs: str, src_crs: str = "EPSG:4326") -> List[Polygon]:
    """将 UTM 多边形反投影回 WGS84。"""
    if not _HAS_PYPROJ or not polys:
        return polys

    transformer = pyproj.Transformer.from_crs(dst_crs, src_crs, always_xy=True)
    unprojected = []
    for p in polys:
        up = shapely_transform(transformer.transform, p)
        unprojected.append(up)
    return unprojected


# ──────────────────────────────────────────────
# 参数配置
# ──────────────────────────────────────────────
@dataclass
class RegularizeConfig:
    """正则化流水线各步骤参数。"""

    # 面积过滤
    min_area: float = 20.0          # 最小面积 (m²)

    # Douglas-Peucker 简化
    dp_tolerance: float = 0.5       # 简化容差 (m)

    # 主方向检测
    use_pca: bool = False           # True = PCA, False = 最小外接矩形

    # 直角化
    angle_threshold: float = 10.0   # 角度吸附阈值 (°)
    snap_angles: List[float] = field(default_factory=lambda: [0, 45, 90, 135])

    # 对称化
    enable_symmetry: bool = False   # 是否启用对称修正
    symmetry_tolerance: float = 2.0 # 对称判定容差 (m)

    # 边界平滑 (Chaikin)
    smooth_iterations: int = 0      # 0 = 不平滑, 建议 1~3
    smooth_ratio: float = 0.25      # Chaikin 切角比例

    # 拓扑
    fix_topology: bool = True


# ──────────────────────────────────────────────
# 步骤 0：拓扑修复
# ──────────────────────────────────────────────
def fix_topology(polygon: Polygon) -> Polygon:
    """修复自相交和无效几何。"""
    if polygon.is_valid:
        return polygon
    fixed = polygon.buffer(0)
    if isinstance(fixed, MultiPolygon):
        # 取最大面
        fixed = max(fixed.geoms, key=lambda g: g.area)
    return fixed


# ──────────────────────────────────────────────
# 步骤 1：最小面积过滤
# ──────────────────────────────────────────────
def filter_by_area(polygons: List[Polygon], min_area: float) -> List[Polygon]:
    """删除面积小于 min_area 的多边形。"""
    return [p for p in polygons if p.area >= min_area]


# ──────────────────────────────────────────────
# 步骤 2：Douglas-Peucker 顶点简化
# ──────────────────────────────────────────────
def simplify_polygon(polygon: Polygon, tolerance: float = 0.5) -> Polygon:
    """Douglas-Peucker 简化，保留拓扑。"""
    simplified = polygon.simplify(tolerance, preserve_topology=True)
    # 退化检查
    if simplified.is_empty or not isinstance(simplified, (Polygon,)):
        return polygon
    return simplified


# ──────────────────────────────────────────────
# 步骤 3：主方向检测
# ──────────────────────────────────────────────
def get_main_direction_mrr(polygon: Polygon) -> float:
    """通过最小外接矩形 (Minimum Rotated Rectangle) 获取建筑主方向角度 (0~180°)。"""
    mrr = polygon.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)
    # 取最长边的方向
    edges = []
    for i in range(len(coords) - 1):
        dx = coords[i + 1][0] - coords[i][0]
        dy = coords[i + 1][1] - coords[i][1]
        length = math.hypot(dx, dy)
        angle = math.degrees(math.atan2(dy, dx)) % 180
        edges.append((length, angle))
    edges.sort(key=lambda e: e[0], reverse=True)
    return edges[0][1]


def get_main_direction_pca(polygon: Polygon) -> float:
    """通过 PCA 第一主成分获取建筑主方向角度 (0~180°)。"""
    from sklearn.decomposition import PCA

    coords = np.array(polygon.exterior.coords)
    pca = PCA(n_components=2)
    pca.fit(coords)
    v = pca.components_[0]
    angle = math.degrees(math.atan2(v[1], v[0])) % 180
    return angle


def get_main_direction(polygon: Polygon, use_pca: bool = False) -> float:
    """获取建筑主方向角度 (0~180°)。"""
    if use_pca:
        return get_main_direction_pca(polygon)
    return get_main_direction_mrr(polygon)


# ──────────────────────────────────────────────
# 步骤 4：直角化 (Orthogonalization)
# ──────────────────────────────────────────────
def _snap_angle(angle_deg: float, snap_angles: List[float], threshold: float) -> float:
    """将角度吸附到最近的 snap_angle。"""
    # 映射到 0~180°
    a = angle_deg % 180
    best = a
    best_diff = float("inf")
    for sa in snap_angles:
        diff = abs(a - sa)
        diff = min(diff, 180 - diff)  # 环形距离
        if diff < best_diff:
            best_diff = diff
            best = sa
    if best_diff <= threshold:
        return best
    return a


def _intersect_lines(
    p1: Tuple[float, float], d1: Tuple[float, float],
    p2: Tuple[float, float], d2: Tuple[float, float],
) -> Tuple[float, float]:
    """求两条直线的交点。p 为线上一点, d 为方向向量。"""
    x1, y1 = p1
    dx1, dy1 = d1
    x2, y2 = p2
    dx2, dy2 = d2

    denom = dx1 * dy2 - dy1 * dx2
    if abs(denom) < 1e-12:
        # 平行线，取中点
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    t = ((x2 - x1) * dy2 - (y2 - y1) * dx2) / denom
    return (x1 + t * dx1, y1 + t * dy1)


def orthogonalize(polygon: Polygon, angle_threshold: float = 10.0,
                  snap_angles: Optional[List[float]] = None) -> Polygon:
    """
    直角化：将接近吸附角度的边修正到精确方向，重新计算顶点。

    算法：
    1. 提取每条边的方向角
    2. 对接近吸附角度的边进行角度修正
    3. 用修正后的边重新计算相邻边交点作为新顶点
    """
    if snap_angles is None:
        snap_angles = [0, 45, 90, 135]

    coords = list(polygon.exterior.coords)[:-1]  # 去掉闭合点
    n = len(coords)
    if n < 4:
        return polygon

    # 计算每条边的方向和角度
    edges = []
    for i in range(n):
        p0 = coords[i]
        p1 = coords[(i + 1) % n]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        angle = math.degrees(math.atan2(dy, dx))
        length = math.hypot(dx, dy)
        edges.append({
            "p0": p0, "p1": p1,
            "dx": dx, "dy": dy,
            "angle": angle,
            "length": length,
        })

    # 对每条边进行角度吸附
    new_edges = []
    for edge in edges:
        orig_angle = edge["angle"]
        snapped = _snap_angle(orig_angle, snap_angles, angle_threshold)
        if abs(snapped - (orig_angle % 180)) < 1e-6 or \
           abs(abs(snapped - (orig_angle % 180)) - 180) < 1e-6:
            # 未吸附
            new_edges.append(edge)
        else:
            rad = math.radians(snapped)
            # 保持边长和中心点不变
            cx = (edge["p0"][0] + edge["p1"][0]) / 2
            cy = (edge["p0"][1] + edge["p1"][1]) / 2
            half_len = edge["length"] / 2
            ndx = math.cos(rad)
            ndy = math.sin(rad)
            # 方向可能需要翻转
            dot = ndx * edge["dx"] + ndy * edge["dy"]
            if dot < 0:
                ndx, ndy = -ndx, -ndy
            new_p0 = (cx - half_len * ndx, cy - half_len * ndy)
            new_p1 = (cx + half_len * ndx, cy + half_len * ndy)
            new_edges.append({
                "p0": new_p0, "p1": new_p1,
                "dx": ndx, "dy": ndy,
                "angle": snapped,
                "length": edge["length"],
            })

    # 重新计算交点作为新顶点
    new_coords = []
    for i in range(n):
        e_prev = new_edges[(i - 1) % n]
        e_curr = new_edges[i]
        intersection = _intersect_lines(
            e_prev["p1"], (e_prev["dx"], e_prev["dy"]),
            e_curr["p0"], (e_curr["dx"], e_curr["dy"]),
        )
        new_coords.append(intersection)

    try:
        result = Polygon(new_coords)
        if not result.is_valid:
            result = result.buffer(0)
            if isinstance(result, MultiPolygon):
                result = max(result.geoms, key=lambda g: g.area)
        # 确保面积不退化
        if result.area < polygon.area * 0.5:
            return polygon
        return result
    except Exception:
        return polygon


# ──────────────────────────────────────────────
# 步骤 5：对称化
# ──────────────────────────────────────────────
def symmetrize(polygon: Polygon, tolerance: float = 2.0) -> Polygon:
    """
    建筑对称化：以主方向为对称轴，将两侧差异较大的顶点做镜像修正。
    仅适用于近似矩形的建筑。
    """
    coords = np.array(polygon.exterior.coords[:-1])
    n = len(coords)
    if n < 4 or n > 20:
        return polygon

    # 获取主方向
    angle = get_main_direction_mrr(polygon)
    rad = math.radians(angle)

    # 旋转使主方向对齐 X 轴
    cos_a, sin_a = math.cos(-rad), math.sin(-rad)
    rotated = []
    for x, y in coords:
        rx = x * cos_a - y * sin_a
        ry = x * sin_a + y * cos_a
        rotated.append((rx, ry))

    # 计算中心
    cx = sum(r[0] for r in rotated) / n
    cy = sum(r[1] for r in rotated) / n

    # 以 Y 轴为对称轴做镜像平均
    new_rotated = []
    for rx, ry in rotated:
        mirror_rx = 2 * cx - rx
        # 寻找最近的镜像点
        best_dist = float("inf")
        best_ry = ry
        for orx, ory in rotated:
            d = abs(orx - mirror_rx) + abs(ory - ry) * 0.1
            if d < best_dist:
                best_dist = d
                best_ry = ory
        if best_dist < tolerance:
            avg_ry = (ry + best_ry) / 2
            new_rotated.append((rx, avg_ry))
        else:
            new_rotated.append((rx, ry))

    # 反旋转
    cos_b, sin_b = math.cos(rad), math.sin(rad)
    result_coords = []
    for rx, ry in new_rotated:
        x = rx * cos_b - ry * sin_b
        y = rx * sin_b + ry * cos_b
        result_coords.append((x, y))

    try:
        result = Polygon(result_coords)
        if result.is_valid and result.area > polygon.area * 0.8:
            return result
    except Exception:
        pass
    return polygon


# ──────────────────────────────────────────────
# 步骤 6：Chaikin 边界平滑
# ──────────────────────────────────────────────
def chaikin_smooth(polygon: Polygon, iterations: int = 2,
                   ratio: float = 0.25) -> Polygon:
    """Chaikin 切角平滑算法。"""
    if iterations <= 0:
        return polygon

    coords = list(polygon.exterior.coords[:-1])
    for _ in range(iterations):
        new_coords = []
        n = len(coords)
        for i in range(n):
            p0 = coords[i]
            p1 = coords[(i + 1) % n]
            q = (
                (1 - ratio) * p0[0] + ratio * p1[0],
                (1 - ratio) * p0[1] + ratio * p1[1],
            )
            r = (
                ratio * p0[0] + (1 - ratio) * p1[0],
                ratio * p0[1] + (1 - ratio) * p1[1],
            )
            new_coords.extend([q, r])
        coords = new_coords

    try:
        result = Polygon(coords)
        if result.is_valid:
            return result
    except Exception:
        pass
    return polygon


# ──────────────────────────────────────────────
# 完整流水线
# ──────────────────────────────────────────────
class RegularizePipeline:
    """
    建筑物轮廓正则化流水线。

    用法::

        pipeline = RegularizePipeline(config)
        results = pipeline.run(raw_polygons)

    每个步骤的中间结果可通过 ``steps`` 属性获取。
    """

    def __init__(self, config: Optional[RegularizeConfig] = None):
        self.config = config or RegularizeConfig()
        self.steps: dict = {}  # step_name -> List[Polygon]

    def _record(self, name: str, polys: List[Polygon]):
        self.steps[name] = polys

    def run(self, polygons: List[Polygon]) -> List[Polygon]:
        cfg = self.config
        self.directions: List[float] = []

        # ── 坐标投影：WGS84 → UTM (米制) ──
        # 记录原始 WGS84 结果（给前端用）
        self._record("0_topology_fix", list(polygons))

        projected, dst_crs = _project_polys(polygons, "EPSG:4326")
        polygons = projected

        # 步骤 0: 拓扑修复
        if cfg.fix_topology:
            polygons = [fix_topology(p) for p in polygons]

        # 步骤 1: 面积过滤 (米制)
        polygons = filter_by_area(polygons, cfg.min_area)
        self._record("1_area_filter", _unproject_polys(polygons, dst_crs))

        # 步骤 2: Douglas-Peucker 简化 (米制)
        polygons = [simplify_polygon(p, cfg.dp_tolerance) for p in polygons]
        self._record("2_dp_simplify", _unproject_polys(polygons, dst_crs))

        # 步骤 3: 主方向检测
        self.directions = []
        for p in polygons:
            d = get_main_direction(p, cfg.use_pca)
            self.directions.append(d)
        self._record("3_direction_detect", _unproject_polys(polygons, dst_crs))

        # 步骤 4: 直角化 (米制)
        polygons = [
            orthogonalize(p, cfg.angle_threshold, cfg.snap_angles)
            for p in polygons
        ]
        self._record("4_orthogonalize", _unproject_polys(polygons, dst_crs))

        # 步骤 5: 对称化 (可选, 米制)
        if cfg.enable_symmetry:
            polygons = [
                symmetrize(p, cfg.symmetry_tolerance) for p in polygons
            ]
        self._record("5_symmetry", _unproject_polys(polygons, dst_crs))

        # 步骤 6: 边界平滑 (米制)
        if cfg.smooth_iterations > 0:
            polygons = [
                chaikin_smooth(p, cfg.smooth_iterations, cfg.smooth_ratio)
                for p in polygons
            ]
        self._record("6_smooth", _unproject_polys(polygons, dst_crs))

        # 最终拓扑检查
        polygons = [fix_topology(p) for p in polygons]
        result_wgs = _unproject_polys(polygons, dst_crs)
        self._record("7_final", result_wgs)

        return result_wgs
