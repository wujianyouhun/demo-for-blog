"""
演示数据生成器 —— 生成包含各类拓扑错误的矢量数据
=================================================
3 组场景:
  1. ai_segmentation  — 模拟 AI 分割结果 (自相交、重叠、碎片)
  2. building_footprints — 建筑轮廓 (孔洞、重叠、多部件)
  3. mixed_errors — 混合错误场景 (所有类型)

中心坐标: 西安 (108.945°E, 34.265°N)
"""

import json
import math
import os
import random
import numpy as np

random.seed(42)
np.random.seed(42)

# ── 基础参数 ──
CX, CY = 108.945, 34.265   # 中心经纬度
SCALE = 0.0003              # 约 30m 的建筑尺度


def _to_geojson_feature(coords, properties=None):
    return {
        "type": "Feature",
        "properties": properties or {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [coords],
        },
    }


def _to_geojson_feature_multi(polys, properties=None):
    return {
        "type": "Feature",
        "properties": properties or {},
        "geometry": {
            "type": "MultiPolygon",
            "coordinates": [[c] for c in polys],
        },
    }


def _to_geojson_feature_with_holes(exterior, holes, properties=None):
    return {
        "type": "Feature",
        "properties": properties or {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [exterior] + holes,
        },
    }


# ── 形状生成工具 ──

def rect(cx, cy, w, h, angle=0):
    """生成矩形坐标 (WGS84)"""
    hw, hh = w / 2, h / 2
    pts = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh), (-hw, -hh)]
    if angle != 0:
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        pts = [(x * cos_a - y * sin_a, x * sin_a + y * cos_a) for x, y in pts]
    return [(cx + dx, cy + dy) for dx, dy in pts]


def bowtie(cx, cy, size):
    """生成蝴蝶结多边形 (自相交)"""
    s = size
    return [
        (cx - s, cy - s),
        (cx + s, cy + s),
        (cx - s, cy + s),
        (cx + s, cy - s),
        (cx - s, cy - s),
    ]


def l_shape(cx, cy, w, h, notch_w, notch_h):
    """L 形建筑"""
    return [
        (cx, cy), (cx + w, cy), (cx + w, cy + h),
        (cx + notch_w, cy + h), (cx + notch_w, cy + notch_h),
        (cx, cy + notch_h), (cx, cy),
    ]


# ══════════════════════════════════════════════════════════
#  场景 1: AI 分割结果 (含自相交、重叠、碎片)
# ══════════════════════════════════════════════════════════

def generate_ai_segmentation():
    features = []
    base_x, base_y = CX - 0.002, CY - 0.001

    # 正常建筑 (8 个)
    for i in range(8):
        row, col = divmod(i, 4)
        cx = base_x + col * 0.0012
        cy = base_y + row * 0.001
        w = SCALE * random.uniform(0.7, 1.3)
        h = SCALE * random.uniform(0.7, 1.3)
        coords = rect(cx, cy, w, h, random.uniform(-5, 5))
        features.append(_to_geojson_feature(coords, {
            "id": i, "type": "building", "source": "normal"
        }))

    # 自相交蝴蝶结 (3 个)
    for i in range(3):
        cx = base_x + (i + 1) * 0.0015
        cy = base_y + 0.0025
        coords = bowtie(cx, cy, SCALE * 0.6)
        features.append(_to_geojson_feature(coords, {
            "id": 8 + i, "type": "building", "source": "self_intersection"
        }))

    # 重叠多边形 (2 对, 4 个)
    for i in range(2):
        cx = base_x + 0.001 + i * 0.002
        cy = base_y - 0.0005
        w, h = SCALE * 1.0, SCALE * 0.8
        # 原始
        coords1 = rect(cx, cy, w, h, 0)
        features.append(_to_geojson_feature(coords1, {
            "id": 11 + i * 2, "type": "building", "source": "overlap_pair"
        }))
        # 重叠偏移
        coords2 = rect(cx + w * 0.3, cy + h * 0.2, w * 0.9, h * 0.9, 5)
        features.append(_to_geojson_feature(coords2, {
            "id": 12 + i * 2, "type": "building", "source": "overlap_pair"
        }))

    # 碎片多边形 (5 个, 极小面积)
    for i in range(5):
        cx = base_x + random.uniform(0, 0.005)
        cy = base_y + random.uniform(-0.002, 0.003)
        s = 0.00001 * random.uniform(0.5, 2.0)  # 极小
        coords = rect(cx, cy, s, s * random.uniform(0.3, 3.0))
        features.append(_to_geojson_feature(coords, {
            "id": 15 + i, "type": "fragment", "source": "sliver"
        }))

    # 重复几何 (2 个完全相同)
    cx, cy = base_x + 0.004, base_y + 0.001
    coords = rect(cx, cy, SCALE, SCALE * 0.7, 10)
    features.append(_to_geojson_feature(coords, {
        "id": 20, "type": "building", "source": "duplicate_original"
    }))
    features.append(_to_geojson_feature(coords, {
        "id": 21, "type": "building", "source": "duplicate_copy"
    }))

    return {"type": "FeatureCollection", "features": features}


# ══════════════════════════════════════════════════════════
#  场景 2: 建筑轮廓 (含孔洞、多部件、重叠)
# ══════════════════════════════════════════════════════════

def generate_building_footprints():
    features = []
    base_x, base_y = CX - 0.001, CY - 0.001

    # 正常建筑 (6 个)
    for i in range(6):
        row, col = divmod(i, 3)
        cx = base_x + col * 0.0015
        cy = base_y + row * 0.0012
        w = SCALE * random.uniform(0.8, 1.2)
        h = SCALE * random.uniform(0.8, 1.2)
        coords = rect(cx, cy, w, h, random.uniform(-3, 3))
        features.append(_to_geojson_feature(coords, {
            "id": i, "type": "building", "source": "normal"
        }))

    # 带孔洞的建筑 (3 个)
    for i in range(3):
        cx = base_x + 0.001 + i * 0.0012
        cy = base_y + 0.003
        w, h = SCALE * 1.5, SCALE * 1.2
        exterior = rect(cx, cy, w, h, 0)
        # 内部孔洞
        hole_w, hole_h = w * 0.25, h * 0.25
        hole = rect(cx, cy, hole_w, hole_h, 0)
        # 孔洞需要反向 (顺时针)
        hole_reversed = list(reversed(hole))
        features.append(_to_geojson_feature_with_holes(exterior, [hole_reversed], {
            "id": 6 + i, "type": "building", "source": "with_hole"
        }))

    # 多部件建筑 (2 个)
    for i in range(2):
        cx = base_x + i * 0.002
        cy = base_y - 0.001
        poly1 = rect(cx, cy, SCALE * 0.8, SCALE * 0.6, 0)
        poly2 = rect(cx + SCALE * 1.5, cy + SCALE * 0.5, SCALE * 0.5, SCALE * 0.4, 15)
        features.append(_to_geojson_feature_multi([poly1, poly2], {
            "id": 9 + i, "type": "building", "source": "multipart"
        }))

    # 重叠建筑对 (2 对)
    for i in range(2):
        cx = base_x + 0.004 + i * 0.0015
        cy = base_y + 0.001
        w, h = SCALE, SCALE * 0.8
        coords1 = rect(cx, cy, w, h, 0)
        coords2 = rect(cx + w * 0.4, cy + h * 0.3, w * 0.85, h * 0.85, 0)
        features.append(_to_geojson_feature(coords1, {
            "id": 11 + i * 2, "type": "building", "source": "overlap"
        }))
        features.append(_to_geojson_feature(coords2, {
            "id": 12 + i * 2, "type": "building", "source": "overlap"
        }))

    # 带自相交的 L 形 (2 个)
    for i in range(2):
        cx = base_x + 0.002 + i * 0.002
        cy = base_y - 0.002
        # 制造一个有自相交问题的多边形
        coords = [
            (cx, cy), (cx + SCALE, cy),
            (cx + SCALE * 0.3, cy + SCALE * 0.7),  # 交叉点
            (cx + SCALE, cy + SCALE),
            (cx, cy + SCALE),
            (cx + SCALE * 0.7, cy + SCALE * 0.3),  # 交叉点
            (cx, cy),
        ]
        features.append(_to_geojson_feature(coords, {
            "id": 15 + i, "type": "building", "source": "self_intersection"
        }))

    return {"type": "FeatureCollection", "features": features}


# ══════════════════════════════════════════════════════════
#  场景 3: 混合错误场景 (所有类型问题)
# ══════════════════════════════════════════════════════════

def generate_mixed_errors():
    features = []
    base_x, base_y = CX - 0.002, CY - 0.002

    # 正常建筑 (5 个)
    for i in range(5):
        cx = base_x + (i % 3) * 0.0015
        cy = base_y + (i // 3) * 0.0012
        w = SCALE * random.uniform(0.8, 1.3)
        h = SCALE * random.uniform(0.8, 1.3)
        coords = rect(cx, cy, w, h, random.uniform(-5, 5))
        features.append(_to_geojson_feature(coords, {
            "id": i, "type": "building", "source": "normal"
        }))

    # 蝴蝶结自相交 (3 个)
    for i in range(3):
        cx = base_x + 0.005 + i * 0.001
        cy = base_y + 0.001
        coords = bowtie(cx, cy, SCALE * random.uniform(0.4, 0.8))
        features.append(_to_geojson_feature(coords, {
            "id": 5 + i, "type": "building", "source": "self_intersection"
        }))

    # 带孔洞 (2 个)
    for i in range(2):
        cx = base_x + 0.001 + i * 0.002
        cy = base_y + 0.003
        w, h = SCALE * 1.4, SCALE * 1.1
        exterior = rect(cx, cy, w, h, 0)
        hole = rect(cx, cy, w * 0.3, h * 0.3, 0)
        hole_reversed = list(reversed(hole))
        features.append(_to_geojson_feature_with_holes(exterior, [hole_reversed], {
            "id": 8 + i, "type": "building", "source": "with_hole"
        }))

    # 重叠对 (2 对)
    for i in range(2):
        cx = base_x + 0.003 + i * 0.002
        cy = base_y + 0.003
        w, h = SCALE, SCALE * 0.7
        coords1 = rect(cx, cy, w, h, 0)
        coords2 = rect(cx + w * 0.35, cy + h * 0.25, w * 0.9, h * 0.85, 3)
        features.append(_to_geojson_feature(coords1, {
            "id": 10 + i * 2, "type": "building", "source": "overlap"
        }))
        features.append(_to_geojson_feature(coords2, {
            "id": 11 + i * 2, "type": "building", "source": "overlap"
        }))

    # 碎片 (6 个)
    for i in range(6):
        cx = base_x + random.uniform(0, 0.006)
        cy = base_y + random.uniform(-0.001, 0.004)
        s = 0.00001 * random.uniform(0.3, 2.5)
        coords = rect(cx, cy, s, s * random.uniform(0.2, 4.0))
        features.append(_to_geojson_feature(coords, {
            "id": 14 + i, "type": "fragment", "source": "sliver"
        }))

    # 多部件 (2 个)
    for i in range(2):
        cx = base_x + 0.001 + i * 0.003
        cy = base_y - 0.001
        p1 = rect(cx, cy, SCALE * 0.7, SCALE * 0.5, 0)
        p2 = rect(cx + SCALE * 2, cy + SCALE * 0.3, SCALE * 0.4, SCALE * 0.3, 20)
        features.append(_to_geojson_feature_multi([p1, p2], {
            "id": 20 + i, "type": "building", "source": "multipart"
        }))

    # 重复 (1 对)
    cx, cy = base_x + 0.005, base_y - 0.001
    coords = rect(cx, cy, SCALE * 0.9, SCALE * 0.7, 8)
    features.append(_to_geojson_feature(coords, {
        "id": 22, "type": "building", "source": "duplicate"
    }))
    features.append(_to_geojson_feature(coords, {
        "id": 23, "type": "building", "source": "duplicate"
    }))

    return {"type": "FeatureCollection", "features": features}


# ══════════════════════════════════════════════════════════
#  主函数
# ══════════════════════════════════════════════════════════

def main():
    out_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(out_dir, exist_ok=True)

    datasets = {
        "ai_segmentation": ("AI分割结果", generate_ai_segmentation),
        "building_footprints": ("建筑轮廓", generate_building_footprints),
        "mixed_errors": ("混合错误", generate_mixed_errors),
    }

    for name, (label, gen_fn) in datasets.items():
        geojson = gen_fn()
        path = os.path.join(out_dir, f"{name}.geojson")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, ensure_ascii=False)
        n = len(geojson["features"])
        print(f"[OK] {label}: {n} 个要素 → {path}")

    print(f"\n共生成 {len(datasets)} 组演示数据")


if __name__ == "__main__":
    main()
