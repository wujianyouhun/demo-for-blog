"""
生成演示数据：模拟 GeoAI 提取的带噪声建筑轮廓。

在局部平面坐标系 (近似米制) 中生成建筑物，
然后偏移到真实的经纬度位置（以西安某区域为中心）。

生成三组数据集:
  1. residential.geojson  - 规则矩形住宅楼群
  2. commercial.geojson   - L 型 / U 型商业建筑
  3. mixed.geojson        - 混合场景（含小碎片误检测）
"""

import json
import math
import os
import random
from typing import List, Tuple

import numpy as np
from shapely.geometry import Polygon, mapping

random.seed(42)
np.random.seed(42)

# 西安某区域中心坐标 (经度, 纬度)
CENTER_LON = 108.945
CENTER_LAT = 34.265
# 1 度纬度 ≈ 111320m；1 度经度 ≈ 111320 * cos(lat) m
M_PER_DEG_LAT = 111320.0
M_PER_DEG_LON = 111320.0 * math.cos(math.radians(CENTER_LAT))


def m_to_deg(dx_m: float, dy_m: float) -> Tuple[float, float]:
    """局部米制偏移 → 经纬度偏移。"""
    return dx_m / M_PER_DEG_LON, dy_m / M_PER_DEG_LAT


def offset_coords(cx: float, cy: float, local_pts: List[Tuple[float, float]]):
    """将局部坐标 (米) 转为经纬度并偏移到中心点。"""
    result = []
    for x, y in local_pts:
        dlon, dlat = m_to_deg(x, y)
        result.append((cx + dlon, cy + dlat))
    return result


# ──────────────────────────────────────────────
# 噪声函数
# ──────────────────────────────────────────────
def add_jitter(pts, sigma=0.3):
    """给每个顶点添加高斯噪声 (单位: 米)。"""
    noisy = []
    for x, y in pts:
        nx = x + random.gauss(0, sigma)
        ny = y + random.gauss(0, sigma)
        noisy.append((nx, ny))
    return noisy


def add_extra_vertices(pts, density=0.5):
    """在边上随机插入额外顶点 (模拟过密的矢量化)。"""
    result = []
    for i in range(len(pts)):
        p0 = pts[i]
        p1 = pts[(i + 1) % len(pts)]
        result.append(p0)
        dist = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        n_extra = max(1, int(dist * density))
        for k in range(1, n_extra + 1):
            t = k / (n_extra + 1)
            mx = p0[0] + t * (p1[0] - p0[0]) + random.gauss(0, 0.15)
            my = p0[1] + t * (p1[1] - p0[1]) + random.gauss(0, 0.15)
            result.append((mx, my))
    return result


def add_edge_serration(pts, amplitude=0.4):
    """给边添加锯齿状波动 (模拟像素边界)。"""
    result = []
    for i in range(len(pts)):
        p0 = pts[i]
        p1 = pts[(i + 1) % len(pts)]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            result.append(p0)
            continue
        # 法向量
        nx, ny = -dy / length, dx / length
        result.append(p0)
        n_mid = max(1, int(length / 1.5))
        for k in range(1, n_mid + 1):
            t = k / (n_mid + 1)
            mx = p0[0] + t * dx + amplitude * nx * random.choice([-1, 1])
            my = p0[1] + t * dy + amplitude * ny * random.choice([-1, 1])
            result.append((mx, my))
    return result


def make_noisy_building(pts, noise_level="medium"):
    """对干净建筑顶点施加不同等级的噪声。"""
    if noise_level == "low":
        pts = add_extra_vertices(pts, density=0.3)
        pts = add_jitter(pts, sigma=0.15)
    elif noise_level == "medium":
        pts = add_edge_serration(pts, amplitude=0.35)
        pts = add_extra_vertices(pts, density=0.4)
        pts = add_jitter(pts, sigma=0.25)
    else:
        pts = add_edge_serration(pts, amplitude=0.6)
        pts = add_extra_vertices(pts, density=0.6)
        pts = add_jitter(pts, sigma=0.4)
    return pts


# ──────────────────────────────────────────────
# 建筑形状生成器
# ──────────────────────────────────────────────
def rect_building(w: float, h: float) -> List[Tuple[float, float]]:
    """矩形建筑 (局部坐标, 米)。"""
    return [(0, 0), (w, 0), (w, h), (0, h)]


def l_building(w: float, h: float, arm_w: float, arm_h: float) -> List[Tuple[float, float]]:
    """L 型建筑。"""
    return [
        (0, 0), (w, 0), (w, arm_h), (arm_w, arm_h),
        (arm_w, h), (0, h),
    ]


def u_building(w: float, h: float, inner_w: float, inner_h: float) -> List[Tuple[float, float]]:
    """U 型建筑。"""
    side = (w - inner_w) / 2
    return [
        (0, 0), (w, 0), (w, h), (w - side, h),
        (w - side, inner_h), (side, inner_h),
        (side, h), (0, h),
    ]


def t_building(w: float, h: float, stem_w: float, stem_h: float) -> List[Tuple[float, float]]:
    """T 型建筑。"""
    side = (w - stem_w) / 2
    return [
        (0, h), (w, h), (w, h - stem_h), (side + stem_w, h - stem_h),
        (side + stem_w, 0), (side, 0), (side, h - stem_h), (0, h - stem_h),
    ]


# ──────────────────────────────────────────────
# 数据集生成
# ──────────────────────────────────────────────
def generate_dataset(buildings, center_lon, center_lat, filename, noise="medium"):
    """生成一组带噪声的建筑数据。"""
    features = []
    for i, (bx, by, pts) in enumerate(buildings):
        # 添加噪声
        noisy_pts = make_noisy_building(pts, noise)
        # 偏移到经纬度
        geo_pts = offset_coords(
            center_lon + bx / M_PER_DEG_LON,
            center_lat + by / M_PER_DEG_LAT,
            noisy_pts,
        )
        geo_pts.append(geo_pts[0])  # 闭合

        try:
            poly = Polygon(geo_pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            features.append({
                "type": "Feature",
                "id": i,
                "geometry": mapping(poly) if isinstance(poly, Polygon) else mapping(poly.geoms[0]),
                "properties": {
                    "building_id": i,
                    "type": "building",
                    "noisy": True,
                },
            })
        except Exception as e:
            print(f"  跳过建筑 {i}: {e}")

    geojson = {"type": "FeatureCollection", "features": features}
    out_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)
    print(f"  已生成: {out_path} ({len(features)} 个建筑)")


def main():
    print("=== 生成演示数据 ===\n")

    # ---- 1. 规则住宅楼群 ----
    print("[1/3] 住宅楼群 (矩形)...")
    res_buildings = []
    for row in range(4):
        for col in range(5):
            bx = col * 35 + random.uniform(-2, 2)
            by = row * 25 + random.uniform(-2, 2)
            w = random.uniform(15, 22)
            h = random.uniform(10, 15)
            pts = rect_building(w, h)
            # 随机旋转 0~5° (模拟轻微偏转)
            angle = random.uniform(-5, 5)
            rad = math.radians(angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            rotated = [(x * cos_a - y * sin_a, x * sin_a + y * cos_a) for x, y in pts]
            res_buildings.append((bx, by, rotated))

    generate_dataset(res_buildings, CENTER_LON, CENTER_LAT, "residential.geojson", "medium")

    # ---- 2. 商业建筑 (L/U/T 型) ----
    print("[2/3] 商业建筑 (L/U/T 型)...")
    com_buildings = []
    positions = [
        (0, 0), (50, 0), (0, 50), (50, 50), (25, 25),
        (0, 100), (50, 100), (100, 50),
    ]
    for bx, by in positions:
        shape_type = random.choice(["L", "U", "T", "rect"])
        if shape_type == "L":
            pts = l_building(25, 20, 10, 10)
        elif shape_type == "U":
            pts = u_building(25, 20, 12, 10)
        elif shape_type == "T":
            pts = t_building(25, 20, 8, 10)
        else:
            pts = rect_building(20, 15)
        angle = random.uniform(-8, 8)
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        rotated = [(x * cos_a - y * sin_a, x * sin_a + y * cos_a) for x, y in pts]
        com_buildings.append((bx + random.uniform(-3, 3), by + random.uniform(-3, 3), rotated))

    generate_dataset(com_buildings, CENTER_LON, CENTER_LAT, "commercial.geojson", "high")

    # ---- 3. 混合场景 (含小碎片) ----
    print("[3/3] 混合场景 (含小碎片误检测)...")
    mixed = []
    # 正常建筑
    for i in range(8):
        bx = random.uniform(0, 150)
        by = random.uniform(0, 100)
        w = random.uniform(12, 25)
        h = random.uniform(8, 18)
        pts = rect_building(w, h)
        angle = random.uniform(-10, 10)
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        rotated = [(x * cos_a - y * sin_a, x * sin_a + y * cos_a) for x, y in pts]
        mixed.append((bx, by, rotated))

    # 小碎片 (误检测: 汽车、树冠、阴影)
    for i in range(12):
        bx = random.uniform(0, 150)
        by = random.uniform(0, 100)
        w = random.uniform(2, 6)
        h = random.uniform(2, 5)
        pts = rect_building(w, h)
        angle = random.uniform(0, 360)
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        rotated = [(x * cos_a - y * sin_a, x * sin_a + y * cos_a) for x, y in pts]
        mixed.append((bx, by, rotated))

    generate_dataset(mixed, CENTER_LON, CENTER_LAT, "mixed.geojson", "medium")

    print("\n=== 演示数据生成完成 ===")


if __name__ == "__main__":
    main()
