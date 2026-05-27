import json
import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import mapping

TOOLS_METADATA = [
    {
        "name": "buffer_analysis",
        "description": "对矢量图层进行缓冲区分析，围绕每个要素生成指定距离的缓冲多边形",
        "category": "vector",
        "parameters": {
            "input_path": "str: 输入矢量文件路径（支持 shp, geojson, gpkg 等）",
            "distance": "float: 缓冲区距离（单位与输入数据坐标系一致）",
            "output_path": "str: 输出结果路径",
        },
        "returns": {"type": "dict", "description": "包含 output 路径和要素数量"},
    },
    {
        "name": "clip_vector",
        "description": "使用裁剪图层裁剪目标矢量图层，保留裁剪范围内的要素",
        "category": "vector",
        "parameters": {
            "input_path": "str: 待裁剪的矢量文件路径",
            "clip_path": "str: 裁剪范围矢量文件路径",
            "output_path": "str: 输出结果路径",
        },
        "returns": {"type": "dict", "description": "包含 output 路径和要素数量"},
    },
    {
        "name": "intersect_vector",
        "description": "计算两个矢量图层的交集，生成保留两者属性的新图层",
        "category": "vector",
        "parameters": {
            "input_path": "str: 第一个输入矢量文件路径",
            "overlay_path": "str: 第二个输入矢量文件路径",
            "output_path": "str: 输出结果路径",
        },
        "returns": {"type": "dict", "description": "包含 output 路径和要素数量"},
    },
    {
        "name": "union_vector",
        "description": "计算两个矢量图层的并集，合并两者几何与属性",
        "category": "vector",
        "parameters": {
            "input_path": "str: 第一个输入矢量文件路径",
            "overlay_path": "str: 第二个输入矢量文件路径",
            "output_path": "str: 输出结果路径",
        },
        "returns": {"type": "dict", "description": "包含 output 路径和要素数量"},
    },
    {
        "name": "difference_vector",
        "description": "计算两个矢量图层的差集，保留在第一个图层中但不在第二个图层区域内的要素",
        "category": "vector",
        "parameters": {
            "input_path": "str: 第一个输入矢量文件路径",
            "overlay_path": "str: 第二个输入矢量文件路径",
            "output_path": "str: 输出结果路径",
        },
        "returns": {"type": "dict", "description": "包含 output 路径和要素数量"},
    },
    {
        "name": "dissolve_vector",
        "description": "按指定属性字段融合矢量要素，合并相同属性值的相邻多边形",
        "category": "vector",
        "parameters": {
            "input_path": "str: 输入矢量文件路径",
            "by_field": "str: 用于分组的字段名",
            "output_path": "str: 输出结果路径",
        },
        "returns": {"type": "dict", "description": "包含 output 路径和要素数量"},
    },
    {
        "name": "reproject_vector",
        "description": "将矢量图层投影转换到目标坐标系",
        "category": "vector",
        "parameters": {
            "input_path": "str: 输入矢量文件路径",
            "target_epsg": "int: 目标 EPSG 代码，如 4326(WGS84)、3857(Web Mercator)",
            "output_path": "str: 输出结果路径",
        },
        "returns": {"type": "dict", "description": "包含 output 路径和目标坐标系信息"},
    },
    {
        "name": "spatial_join",
        "description": "基于空间关系将两个图层的属性关联起来（如统计落在每个行政区的 POI 数量）",
        "category": "vector",
        "parameters": {
            "target_path": "str: 目标图层路径（保留其几何）",
            "join_path": "str: 关联图层路径（提供属性）",
            "how": "str: 空间关系，可选 intersects/contains/within，默认 intersects",
            "output_path": "str: 输出结果路径",
        },
        "returns": {"type": "dict", "description": "包含 output 路径和要素数量"},
    },
    {
        "name": "centroid",
        "description": "计算矢量图层中每个要素的几何中心点",
        "category": "vector",
        "parameters": {
            "input_path": "str: 输入矢量文件路径",
            "output_path": "str: 输出点图层路径",
        },
        "returns": {"type": "dict", "description": "包含 output 路径和要素数量"},
    },
    {
        "name": "simplify_vector",
        "description": "简化矢量几何，减少顶点数量，用于降精度展示或减少数据量",
        "category": "vector",
        "parameters": {
            "input_path": "str: 输入矢量文件路径",
            "tolerance": "float: 简化容差（单位与数据坐标系一致）",
            "output_path": "str: 输出结果路径",
        },
        "returns": {"type": "dict", "description": "包含 output 路径和要素数量"},
    },
    {
        "name": "select_by_attribute",
        "description": "按属性条件筛选矢量要素，如选出 population > 10000 的要素",
        "category": "vector",
        "parameters": {
            "input_path": "str: 输入矢量文件路径",
            "field": "str: 用于筛选的字段名",
            "operator": "str: 比较运算符，支持 == != > >= < <=",
            "value": "str: 比较值",
            "output_path": "str: 输出结果路径",
        },
        "returns": {"type": "dict", "description": "包含 output 路径和筛选后的要素数量"},
    },
    {
        "name": "select_by_location",
        "description": "按空间位置关系筛选要素，如选出在某区域内的所有要素",
        "category": "vector",
        "parameters": {
            "input_path": "str: 待筛选的矢量文件路径",
            "by_path": "str: 筛选范围的矢量文件路径",
            "relation": "str: 空间关系，可选 intersects/contains/within/crosses/touches",
            "output_path": "str: 输出结果路径",
        },
        "returns": {"type": "dict", "description": "包含 output 路径和筛选后的要素数量"},
    },
    {
        "name": "export_geojson",
        "description": "将矢量文件导出为 GeoJSON 格式",
        "category": "vector",
        "parameters": {
            "input_path": "str: 输入矢量文件路径",
            "output_path": "str: 输出 GeoJSON 路径",
        },
        "returns": {"type": "dict", "description": "包含 output 路径和要素数量"},
    },
    {
        "name": "get_vector_info",
        "description": "获取矢量图层的元数据信息（坐标系、字段列表、要素数量、范围等）",
        "category": "vector",
        "parameters": {
            "input_path": "str: 输入矢量文件路径",
        },
        "returns": {"type": "dict", "description": "包含 crs、字段、数量、范围等元数据"},
    },
    {
        "name": "area_calculate",
        "description": "计算面要素的面积并添加为新字段",
        "category": "vector",
        "parameters": {
            "input_path": "str: 输入矢量文件路径",
            "output_path": "str: 输出结果路径",
            "field_name": "str: 存储面积的字段名，默认 area",
            "unit": "str: 面积单位，可选 sq_m/sq_km/ha，默认 sq_m",
        },
        "returns": {"type": "dict", "description": "包含 output 路径和要素数量"},
    },
    {
        "name": "distance_calculate",
        "description": "计算两个点图层之间最近要素的距离，结果写入输出图层的属性表",
        "category": "vector",
        "parameters": {
            "input_path": "str: 源点图层路径",
            "target_path": "str: 目标点图层路径",
            "output_path": "str: 输出结果路径",
            "field_name": "str: 存储距离的字段名，默认 nearest_dist",
        },
        "returns": {"type": "dict", "description": "包含 output 路径和要素数量"},
    },
]

OPERATOR_MAP = {
    "==": lambda s, v: s == v,
    "!=": lambda s, v: s != v,
    ">":  lambda s, v: s > v,
    ">=": lambda s, v: s >= v,
    "<":  lambda s, v: s < v,
    "<=": lambda s, v: s <= v,
}


def _auto_reproject(gdf, target_crs=None):
    if gdf.crs is None:
        return gdf
    if gdf.crs.is_geographic:
        gdf = gdf.to_crs(epsg=3857)
    if target_crs is not None:
        gdf = gdf.to_crs(target_crs)
    return gdf


def _validate_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")


def buffer_analysis(input_path, distance, output_path):
    gdf = gpd.read_file(input_path)
    if gdf.crs is not None and gdf.crs.is_geographic:
        gdf = gdf.to_crs(epsg=3857)
    gdf["geometry"] = gdf.buffer(distance)
    gdf.to_file(output_path)
    return {"output": output_path, "feature_count": len(gdf)}


def clip_vector(input_path, clip_path, output_path):
    gdf = gpd.read_file(input_path)
    clip_gdf = gpd.read_file(clip_path)
    if gdf.crs != clip_gdf.crs and clip_gdf.crs is not None:
        gdf = gdf.to_crs(clip_gdf.crs)
    result = gpd.clip(gdf, clip_gdf)
    result.to_file(output_path)
    return {"output": output_path, "feature_count": len(result)}


def intersect_vector(input_path, overlay_path, output_path):
    gdf1 = gpd.read_file(input_path)
    gdf2 = gpd.read_file(overlay_path)
    if gdf1.crs != gdf2.crs:
        gdf2 = gdf2.to_crs(gdf1.crs)
    result = gpd.overlay(gdf1, gdf2, how="intersection")
    result.to_file(output_path)
    return {"output": output_path, "feature_count": len(result)}


def union_vector(input_path, overlay_path, output_path):
    gdf1 = gpd.read_file(input_path)
    gdf2 = gpd.read_file(overlay_path)
    if gdf1.crs != gdf2.crs:
        gdf2 = gdf2.to_crs(gdf1.crs)
    result = gpd.overlay(gdf1, gdf2, how="union")
    result.to_file(output_path)
    return {"output": output_path, "feature_count": len(result)}


def difference_vector(input_path, overlay_path, output_path):
    gdf1 = gpd.read_file(input_path)
    gdf2 = gpd.read_file(overlay_path)
    if gdf1.crs != gdf2.crs:
        gdf2 = gdf2.to_crs(gdf1.crs)
    result = gpd.overlay(gdf1, gdf2, how="difference")
    result.to_file(output_path)
    return {"output": output_path, "feature_count": len(result)}


def dissolve_vector(input_path, by_field, output_path):
    gdf = gpd.read_file(input_path)
    result = gdf.dissolve(by=by_field)
    result.to_file(output_path)
    return {"output": output_path, "feature_count": len(result)}


def reproject_vector(input_path, target_epsg, output_path):
    gdf = gpd.read_file(input_path)
    gdf = gdf.to_crs(epsg=target_epsg)
    gdf.to_file(output_path)
    old_crs = str(gdf.crs) if gdf.crs else "undefined"
    return {"output": output_path, "target_crs": f"EPSG:{target_epsg}", "feature_count": len(gdf)}


def spatial_join(target_path, join_path, output_path, how="intersects"):
    target = gpd.read_file(target_path)
    join_df = gpd.read_file(join_path)
    if target.crs != join_df.crs:
        join_df = join_df.to_crs(target.crs)
    how_map = {
        "intersects": "intersects",
        "contains": "contains",
        "within": "within",
    }
    predicate = how_map.get(how, "intersects")
    result = gpd.sjoin(target, join_df, how="inner", predicate=predicate)
    result.to_file(output_path)
    return {"output": output_path, "feature_count": len(result)}


def centroid(input_path, output_path):
    gdf = gpd.read_file(input_path)
    result = gdf.copy()
    result["geometry"] = gdf.geometry.centroid
    result.to_file(output_path)
    return {"output": output_path, "feature_count": len(result)}


def simplify_vector(input_path, tolerance, output_path):
    gdf = gpd.read_file(input_path)
    result = gdf.copy()
    result["geometry"] = gdf.geometry.simplify(tolerance, preserve_topology=True)
    result.to_file(output_path)
    return {"output": output_path, "feature_count": len(result)}


def select_by_attribute(input_path, field, operator, value, output_path):
    gdf = gpd.read_file(input_path)
    ops = OPERATOR_MAP
    if operator not in ops:
        supported = ", ".join(ops.keys())
        raise ValueError(f"Unsupported operator '{operator}'. Supported: {supported}")

    op_func = ops[operator]
    col = gdf[field]

    try:
        numeric_col = pd.to_numeric(col, errors="coerce")
        numeric_value = float(value)
        if numeric_col.notna().all():
            col = numeric_col
            value = numeric_value
    except (ValueError, TypeError):
        pass

    mask = op_func(col, value)
    result = gdf[mask]
    result.to_file(output_path)
    return {"output": output_path, "feature_count": len(result), "filtered_out": len(gdf) - len(result)}


def select_by_location(input_path, by_path, relation, output_path):
    gdf = gpd.read_file(input_path)
    by_gdf = gpd.read_file(by_path)
    if gdf.crs != by_gdf.crs:
        gdf = gdf.to_crs(by_gdf.crs)

    relation_map = {
        "intersects": "intersects",
        "contains": "contains",
        "within": "within",
        "crosses": "crosses",
        "touches": "touches",
        "overlaps": "overlaps",
    }
    op = relation_map.get(relation, "intersects")

    if len(by_gdf) > 1:
        by_geom = by_gdf.unary_union
    else:
        by_geom = by_gdf.geometry.iloc[0]

    mask = gdf.geometry.apply(getattr, args=(op, by_geom))
    result = gdf[mask]
    result.to_file(output_path)
    return {"output": output_path, "feature_count": len(result), "filtered_in": len(result)}


def export_geojson(input_path, output_path):
    gdf = gpd.read_file(input_path)
    gdf.to_file(output_path, driver="GeoJSON")
    return {"output": output_path, "feature_count": len(gdf)}


def get_vector_info(input_path):
    gdf = gpd.read_file(input_path)
    info = {
        "feature_count": len(gdf),
        "crs": str(gdf.crs) if gdf.crs else None,
        "geometry_types": gdf.geometry.geom_type.unique().tolist(),
        "fields": list(gdf.columns),
        "bounds": {
            "minx": gdf.total_bounds[0],
            "miny": gdf.total_bounds[1],
            "maxx": gdf.total_bounds[2],
            "maxy": gdf.total_bounds[3],
        },
        "file_path": input_path,
    }
    try:
        info["sample"] = gdf.head(3).to_dict("records")
    except Exception:
        info["sample"] = None
    return info


def area_calculate(input_path, output_path, field_name="area", unit="sq_m"):
    gdf = gpd.read_file(input_path)
    if gdf.crs is not None and gdf.crs.is_geographic:
        gdf_proj = gdf.to_crs(epsg=3857)
    else:
        gdf_proj = gdf

    areas = gdf_proj.geometry.area

    unit_factors = {"sq_m": 1.0, "sq_km": 1e-6, "ha": 1e-4}
    factor = unit_factors.get(unit, 1.0)

    gdf[field_name] = areas * factor
    gdf.to_file(output_path)
    return {"output": output_path, "feature_count": len(gdf)}


def distance_calculate(input_path, target_path, output_path, field_name="nearest_dist"):
    gdf_src = gpd.read_file(input_path)
    gdf_tgt = gpd.read_file(target_path)

    if gdf_src.crs != gdf_tgt.crs:
        gdf_tgt = gdf_tgt.to_crs(gdf_src.crs)

    tgt_geoms = gdf_tgt.geometry.unary_union

    distances = gdf_src.geometry.apply(
        lambda g: g.distance(tgt_geoms) if g is not None else None
    )
    gdf_src[field_name] = distances
    gdf_src.to_file(output_path)
    return {"output": output_path, "feature_count": len(gdf_src)}
