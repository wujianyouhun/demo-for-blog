"""
矢量数据质量自动检查与一键修复引擎
====================================
基于 GeoPandas + Shapely (GEOS) 的完整拓扑质控流程：
  - VectorQualityChecker: 7 项自动检测
  - VectorAutoRepair:    8 步一键修复
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid, explain_validity
from typing import Optional, List, Dict, Any
import warnings

warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════
#  质检引擎
# ════════════════════════════════════════════════════════════

class VectorQualityChecker:
    """矢量数据质量检查器 —— 7 项检查全覆盖"""

    def __init__(self, gdf: gpd.GeoDataFrame):
        self.gdf = gdf.copy()
        self.issues: List[Dict[str, Any]] = []
        self.report: Dict[str, Any] = {}

    # ── 运行所有检查 ────────────────────────────────────
    def run_all_checks(self, sliver_min_area: float = 5.0,
                       overlap_threshold: float = 0.05) -> List[Dict]:
        self.issues = []
        print("=" * 60)
        print("  矢量数据质量自动检查")
        print("=" * 60)
        print(f"  输入要素数: {len(self.gdf)}")
        print(f"  坐标系: {self.gdf.crs}")
        print("=" * 60)

        self.check_validity()
        self.check_self_intersection()
        self.check_holes()
        self.check_overlaps(overlap_threshold=overlap_threshold)
        self.check_slivers(min_area=sliver_min_area)
        self.check_multipart()
        self.check_duplicate_geometry()

        self.generate_report()
        return self.issues

    # ── 1. 几何合法性检查 ──────────────────────────────
    def check_validity(self):
        invalid_mask = ~self.gdf.geometry.is_valid
        for idx in self.gdf[invalid_mask].index:
            geom = self.gdf.loc[idx, "geometry"]
            self.issues.append({
                "fid": int(idx),
                "error_type": "Invalid Geometry",
                "severity": "HIGH",
                "detail": str(explain_validity(geom)),
                "auto_fixable": True,
            })
        print(f"[check] 几何合法性检查: {invalid_mask.sum()} 个问题")

    # ── 2. 自相交检查 ──────────────────────────────────
    def check_self_intersection(self):
        count = 0
        for idx, row in self.gdf.iterrows():
            geom = row.geometry
            if geom is not None and not geom.is_valid:
                reason = str(explain_validity(geom))
                if "Self-intersection" in reason:
                    count += 1
                    self.issues.append({
                        "fid": int(idx),
                        "error_type": "Self-Intersection",
                        "severity": "HIGH",
                        "detail": reason,
                        "auto_fixable": True,
                    })
        print(f"[check] 自相交检查: {count} 个问题")

    # ── 3. 孔洞检查 ────────────────────────────────────
    def check_holes(self):
        count = 0
        for idx, row in self.gdf.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
            for poly in polys:
                if poly.geom_type == "Polygon" and len(list(poly.interiors)) > 0:
                    count += 1
                    self.issues.append({
                        "fid": int(idx),
                        "error_type": "Hole",
                        "severity": "MEDIUM",
                        "detail": f"包含 {len(list(poly.interiors))} 个孔洞",
                        "auto_fixable": True,
                    })
        print(f"[check] 孔洞检查: {count} 个问题")

    # ── 4. 重叠检查 (R-Tree 空间索引加速) ──────────────
    def check_overlaps(self, overlap_threshold: float = 0.05):
        sindex = self.gdf.sindex
        checked = set()
        count = 0

        for idx, row in self.gdf.iterrows():
            geom = row.geometry
            if geom is None or not geom.is_valid:
                continue
            candidates = list(sindex.intersection(geom.bounds))
            for pos in candidates:
                other_idx = self.gdf.index[pos]
                pair = tuple(sorted([int(idx), int(other_idx)]))
                if pair in checked or idx == other_idx:
                    continue
                checked.add(pair)

                other = self.gdf.loc[other_idx, "geometry"]
                if other is None or not other.is_valid:
                    continue
                try:
                    if geom.intersects(other):
                        inter = geom.intersection(other)
                        if inter.area > 0:
                            min_area = min(geom.area, other.area)
                            ratio = inter.area / min_area if min_area > 0 else 0
                            if ratio > overlap_threshold:
                                count += 1
                                self.issues.append({
                                    "fid": f"{int(idx)}&{int(other_idx)}",
                                    "fid_1": int(idx),
                                    "fid_2": int(other_idx),
                                    "error_type": "Overlap",
                                    "severity": "HIGH",
                                    "detail": f"重叠比例: {ratio:.2%}",
                                    "overlap_ratio": round(ratio, 4),
                                    "auto_fixable": True,
                                })
                except Exception:
                    continue
        print(f"[check] 重叠检查: {count} 个问题")

    # ── 5. 碎片检查 ────────────────────────────────────
    def check_slivers(self, min_area: float = 5.0):
        count = 0
        for idx, row in self.gdf.iterrows():
            geom = row.geometry
            if geom is not None and geom.area < min_area:
                count += 1
                self.issues.append({
                    "fid": int(idx),
                    "error_type": "Sliver",
                    "severity": "LOW",
                    "detail": f"面积: {geom.area:.2f} sq.m",
                    "auto_fixable": True,
                })
        print(f"[check] 碎片检查: {count} 个问题")

    # ── 6. 多部件检查 ──────────────────────────────────
    def check_multipart(self):
        count = 0
        for idx, row in self.gdf.iterrows():
            geom = row.geometry
            if geom is not None and geom.geom_type == "MultiPolygon":
                count += 1
                self.issues.append({
                    "fid": int(idx),
                    "error_type": "MultiPart",
                    "severity": "LOW",
                    "detail": f"包含 {len(list(geom.geoms))} 个部件",
                    "auto_fixable": True,
                })
        print(f"[check] 多部件检查: {count} 个问题")

    # ── 7. 重复几何检查 ────────────────────────────────
    def check_duplicate_geometry(self):
        count = 0
        seen = {}
        for idx, row in self.gdf.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            wkb = geom.wkb_hex
            if wkb in seen:
                count += 1
                self.issues.append({
                    "fid": int(idx),
                    "error_type": "Duplicate",
                    "severity": "MEDIUM",
                    "detail": f"与要素 {seen[wkb]} 几何完全相同",
                    "auto_fixable": True,
                })
            else:
                seen[wkb] = int(idx)
        print(f"[check] 重复几何检查: {count} 个问题")

    # ── 生成报告 ───────────────────────────────────────
    def generate_report(self) -> Dict:
        if not self.issues:
            self.report = {"total_issues": 0, "by_type": {}, "by_severity": {}}
            return self.report

        df = pd.DataFrame(self.issues)
        by_type = df.groupby("error_type").size().to_dict()
        by_severity = df.groupby("severity").size().to_dict()

        self.report = {
            "total_issues": len(df),
            "by_type": by_type,
            "by_severity": by_severity,
            "fixable_rate": float(df["auto_fixable"].mean()) if "auto_fixable" in df.columns else 1.0,
        }

        print(f"\n{'=' * 60}")
        print(f"  质检报告汇总")
        print(f"{'=' * 60}")
        print(f"  总问题数: {self.report['total_issues']}")
        print(f"  可自动修复: {self.report['fixable_rate']:.0%}")
        for t, c in by_type.items():
            print(f"    {t}: {c}")
        print(f"{'=' * 60}")
        return self.report

    # ── 获取问题要素的 GeoJSON (供前端可视化) ────────────
    def get_issues_geojson(self) -> dict:
        """将所有问题标记导出为 GeoJSON，每个问题包含原始几何"""
        features = []
        for issue in self.issues:
            fid = issue.get("fid")
            geom = None
            if isinstance(fid, int) and fid in self.gdf.index:
                geom = self.gdf.loc[fid, "geometry"]
            elif isinstance(fid, str) and "&" in fid:
                # 重叠问题取两个要素的并集
                parts = fid.split("&")
                try:
                    g1 = self.gdf.loc[int(parts[0]), "geometry"]
                    g2 = self.gdf.loc[int(parts[1]), "geometry"]
                    geom = g1.union(g2)
                except Exception:
                    pass
            if geom is not None:
                features.append({
                    "type": "Feature",
                    "properties": {
                        "error_type": issue["error_type"],
                        "severity": issue["severity"],
                        "detail": issue["detail"],
                        "fid": str(fid),
                    },
                    "geometry": geom.__geo_interface__,
                })
        return {"type": "FeatureCollection", "features": features}


# ════════════════════════════════════════════════════════════
#  一键修复引擎
# ════════════════════════════════════════════════════════════

DEFAULT_REPAIR_CONFIG = {
    "repair_invalid": True,
    "fill_holes": True,
    "max_hole_area": None,
    "remove_overlaps": True,
    "overlap_method": "area",      # area / clip
    "overlap_threshold": 0.4,
    "remove_slivers": True,
    "min_area": 10,
    "explode_multipart": True,
    "remove_duplicates": True,
    "simplify": False,
    "simplify_tolerance": 0.5,
    "preserve_topology": True,
}


class VectorAutoRepair:
    """矢量数据一键修复引擎 —— 8 步完整流程"""

    def __init__(self, gdf: gpd.GeoDataFrame, config: dict = None):
        self.gdf = gdf.copy()
        self.config = {**DEFAULT_REPAIR_CONFIG, **(config or {})}
        self.repair_log: List[str] = []
        self.step_results: Dict[str, gpd.GeoDataFrame] = {}  # 每步结果

    def repair_all(self) -> gpd.GeoDataFrame:
        print("=" * 60)
        print("  矢量数据一键修复")
        print("=" * 60)

        before_count = len(self.gdf)
        before_invalid = int((~self.gdf.geometry.is_valid).sum())

        if self.config["repair_invalid"]:
            self._repair_invalid_geometry()
        if self.config["fill_holes"]:
            self._fill_holes()
        if self.config["remove_overlaps"]:
            self._remove_overlaps()
        if self.config["remove_slivers"]:
            self._remove_slivers()
        if self.config["explode_multipart"]:
            self._explode_multipart()
        if self.config["remove_duplicates"]:
            self._remove_duplicates()
        if self.config["simplify"]:
            self._simplify()
        self._final_validation()

        after_count = len(self.gdf)
        after_invalid = int((~self.gdf.geometry.is_valid).sum())

        print(f"\n  修复前: {before_count} 要素 ({before_invalid} 无效)")
        print(f"  修复后: {after_count} 要素 ({after_invalid} 无效)")
        if after_count > 0:
            print(f"  有效率: {(1 - after_invalid / after_count) * 100:.2f}%")
        print("=" * 60)

        return self.gdf

    # ── Step 1: 修复无效几何 ──────────────────────────
    def _repair_invalid_geometry(self):
        count_before = int((~self.gdf.geometry.is_valid).sum())
        self.gdf["geometry"] = self.gdf.geometry.apply(
            lambda g: make_valid(g) if g is not None and not g.is_valid else g
        )
        count_after = int((~self.gdf.geometry.is_valid).sum())
        fixed = count_before - count_after
        self.repair_log.append(f"修复无效几何: {fixed} 个 (剩余 {count_after})")
        print(f"[Step 1] 修复无效几何: {fixed} 个已修复")
        self.step_results["repair_invalid"] = self.gdf.copy()

    # ── Step 2: 填充孔洞 ──────────────────────────────
    def _fill_holes(self):
        filled = 0
        max_area = self.config["max_hole_area"]

        def _do_fill(geom):
            nonlocal filled
            if geom is None:
                return geom
            if geom.geom_type == "Polygon":
                new_interiors = []
                for interior in geom.interiors:
                    hole = Polygon(interior)
                    if max_area is None or hole.area <= max_area:
                        filled += 1
                    else:
                        new_interiors.append(interior)
                return Polygon(geom.exterior, new_interiors)
            elif geom.geom_type == "MultiPolygon":
                return MultiPolygon([_do_fill(p) for p in geom.geoms])
            return geom

        self.gdf["geometry"] = self.gdf.geometry.apply(_do_fill)
        self.repair_log.append(f"填充孔洞: {filled} 个")
        print(f"[Step 2] 填充孔洞: {filled} 个")
        self.step_results["fill_holes"] = self.gdf.copy()

    # ── Step 3: 去除重叠 (面积优先去重) ───────────────
    def _remove_overlaps(self):
        before = len(self.gdf)
        threshold = self.config["overlap_threshold"]
        method = self.config["overlap_method"]

        if method == "area":
            self.gdf = self._remove_overlaps_by_area(threshold)
        else:
            self.gdf = self._clip_overlaps()

        removed = before - len(self.gdf)
        self.repair_log.append(f"去除重叠: 移除 {removed} 个 (方法: {method})")
        print(f"[Step 3] 去除重叠: 移除 {removed} 个")
        self.step_results["remove_overlaps"] = self.gdf.copy()

    def _remove_overlaps_by_area(self, threshold: float) -> gpd.GeoDataFrame:
        gdf = self.gdf.copy()
        gdf["area"] = gdf.geometry.area
        gdf = gdf.sort_values("area", ascending=False).reset_index(drop=True)
        keep = [True] * len(gdf)
        sindex = gdf.sindex

        for i in range(len(gdf)):
            if not keep[i]:
                continue
            geom_i = gdf.loc[i, "geometry"]
            if geom_i is None or not geom_i.is_valid:
                continue
            candidates = list(sindex.intersection(geom_i.bounds))
            for j in candidates:
                if j <= i or not keep[j]:
                    continue
                geom_j = gdf.loc[j, "geometry"]
                if geom_j is None or not geom_j.is_valid:
                    continue
                try:
                    if geom_i.intersects(geom_j):
                        inter_area = geom_i.intersection(geom_j).area
                        iou = inter_area / (geom_i.area + geom_j.area - inter_area)
                        if iou > threshold:
                            keep[j] = False
                except Exception:
                    continue

        return gdf[keep].drop(columns=["area"]).reset_index(drop=True)

    def _clip_overlaps(self) -> gpd.GeoDataFrame:
        gdf = self.gdf.copy()
        gdf["area"] = gdf.geometry.area
        gdf = gdf.sort_values("area", ascending=False).reset_index(drop=True)

        for i in range(len(gdf)):
            geom_i = gdf.loc[i, "geometry"]
            if geom_i is None:
                continue
            for j in range(i + 1, len(gdf)):
                geom_j = gdf.loc[j, "geometry"]
                if geom_j is None:
                    continue
                try:
                    if geom_i.intersects(geom_j):
                        clipped = geom_j.difference(geom_i)
                        if not clipped.is_empty:
                            gdf.loc[j, "geometry"] = clipped
                except Exception:
                    continue
        return gdf.drop(columns=["area"]).reset_index(drop=True)

    # ── Step 4: 去除碎片 ──────────────────────────────
    def _remove_slivers(self):
        before = len(self.gdf)
        self.gdf = self.gdf[self.gdf.geometry.area >= self.config["min_area"]]
        self.gdf = self.gdf.reset_index(drop=True)
        removed = before - len(self.gdf)
        self.repair_log.append(f"去除碎片: 移除 {removed} 个 (min_area={self.config['min_area']})")
        print(f"[Step 4] 去除碎片: 移除 {removed} 个")
        self.step_results["remove_slivers"] = self.gdf.copy()

    # ── Step 5: 拆分多部件 ────────────────────────────
    def _explode_multipart(self):
        before = len(self.gdf)
        self.gdf = self.gdf.explode(index_parts=False).reset_index(drop=True)
        after = len(self.gdf)
        added = after - before
        self.repair_log.append(f"拆分多部件: 新增 {added} 个要素")
        print(f"[Step 5] 拆分多部件: 新增 {added} 个要素")
        self.step_results["explode_multipart"] = self.gdf.copy()

    # ── Step 6: 去除重复 ──────────────────────────────
    def _remove_duplicates(self):
        before = len(self.gdf)
        self.gdf = self.gdf.drop_duplicates(subset="geometry", keep="first").reset_index(drop=True)
        removed = before - len(self.gdf)
        self.repair_log.append(f"去除重复: 移除 {removed} 个")
        print(f"[Step 6] 去除重复: 移除 {removed} 个")
        self.step_results["remove_duplicates"] = self.gdf.copy()

    # ── Step 7: 几何简化 (可选) ──────────────────────
    def _simplify(self):
        tol = self.config["simplify_tolerance"]
        preserve = self.config["preserve_topology"]
        self.gdf["geometry"] = self.gdf.geometry.apply(
            lambda g: g.simplify(tol, preserve_topology=preserve) if g else g
        )
        self.repair_log.append(f"几何简化: tolerance={tol}")
        print(f"[Step 7] 几何简化: 完成")
        self.step_results["simplify"] = self.gdf.copy()

    # ── Step 8: 最终验证 ──────────────────────────────
    def _final_validation(self):
        invalid = int((~self.gdf.geometry.is_valid).sum())
        empty = int(self.gdf.geometry.is_empty.sum())
        null = int(self.gdf.geometry.isna().sum())
        self.gdf = self.gdf[~self.gdf.geometry.is_empty & self.gdf.geometry.notna()]
        self.gdf = self.gdf.reset_index(drop=True)
        self.repair_log.append(f"最终验证: {invalid} 无效, {empty} 空, {null} 空值")
        print(f"[Step 8] 最终验证: 清理完成")
        self.step_results["final"] = self.gdf.copy()

    def get_repair_log(self) -> List[str]:
        return self.repair_log

    def get_step_geojson(self, step_key: str) -> Optional[dict]:
        """获取某步结果的 GeoJSON"""
        if step_key in self.step_results:
            return self.step_results[step_key].__geo_interface__
        return None
