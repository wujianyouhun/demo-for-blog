import os
import numpy as np
import rasterio
from rasterio import mask, warp, features, transform
from rasterio.enums import Resampling
import json

RESAMPLING_MAP = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "cubic_spline": Resampling.cubic_spline,
    "lanczos": Resampling.lanczos,
    "average": Resampling.average,
    "mode": Resampling.mode,
    "max": Resampling.max,
    "min": Resampling.min,
    "med": Resampling.med,
    "q1": Resampling.q1,
    "q3": Resampling.q3,
    "sum": Resampling.sum,
}

TOOLS_METADATA = [
    {
        "name": "clip_raster",
        "description": "使用矢量或边界范围裁剪栅格数据",
        "category": "raster",
        "parameters": {
            "input_path": "str: 输入栅格文件路径（tif, img 等）",
            "clip_path": "str: 裁剪范围矢量文件路径（shp, geojson 等）",
            "output_path": "str: 输出栅格路径",
        },
        "returns": {"type": "dict", "description": "包含 output、尺寸、坐标系信息"},
    },
    {
        "name": "reproject_raster",
        "description": "将栅格数据重投影到目标坐标系",
        "category": "raster",
        "parameters": {
            "input_path": "str: 输入栅格文件路径",
            "target_epsg": "int: 目标 EPSG 代码，如 4326(WGS84)、3857(Web Mercator)",
            "output_path": "str: 输出栅格路径",
            "resampling": "str: 重采样方法，默认 nearest，可选 bilinear/cubic/average/mode 等",
        },
        "returns": {"type": "dict", "description": "包含 output、目标坐标系、尺寸信息"},
    },
    {
        "name": "resample_raster",
        "description": "调整栅格分辨率（升采样或降采样）",
        "category": "raster",
        "parameters": {
            "input_path": "str: 输入栅格文件路径",
            "scale_factor": "float: 缩放因子，>1 升采样（更细）、<1 降采样（更粗）",
            "output_path": "str: 输出栅格路径",
            "method": "str: 重采样方法，默认 bilinear",
        },
        "returns": {"type": "dict", "description": "包含 output、新分辨率信息"},
    },
    {
        "name": "get_raster_info",
        "description": "获取栅格数据的元数据信息（分辨率、波段数、坐标系、范围等）",
        "category": "raster",
        "parameters": {
            "input_path": "str: 输入栅格文件路径",
        },
        "returns": {"type": "dict", "description": "包含尺寸、波段数、分辨率、范围、nodata 等元数据"},
    },
    {
        "name": "raster_calculator",
        "description": "对单波段或多波段栅格进行代数运算，支持 + - * / 及 numpy 函数",
        "category": "raster",
        "parameters": {
            "input_path": "str: 输入栅格文件路径",
            "expression": "str: 运算表达式，使用 band_1, band_2 引用波段，如 'band_1 * 2 + band_2'",
            "output_path": "str: 输出栅格路径",
            "nodata": "float: 输出 NoData 值，默认使用输入的 nodata 值",
        },
        "returns": {"type": "dict", "description": "包含 output 和统计信息"},
    },
    {
        "name": "raster_to_vector",
        "description": "将栅格数据多边形化，转换为矢量图层",
        "category": "raster",
        "parameters": {
            "input_path": "str: 输入栅格文件路径",
            "output_path": "str: 输出矢量文件路径",
            "band": "int: 使用的波段索引（1-based），默认 1",
        },
        "returns": {"type": "dict", "description": "包含 output 和面要素数量"},
    },
    {
        "name": "merge_rasters",
        "description": "合并（镶嵌）多幅栅格图像为单幅",
        "category": "raster",
        "parameters": {
            "input_dir": "str: 包含待合并栅格文件的目录路径",
            "output_path": "str: 输出栅格路径",
            "resampling": "str: 重采样方法，默认 nearest",
        },
        "returns": {"type": "dict", "description": "包含 output 和合并后尺寸"},
    },
    {
        "name": "raster_statistics",
        "description": "计算栅格波段的基础统计信息（最小值、最大值、均值、标准差等）",
        "category": "raster",
        "parameters": {
            "input_path": "str: 输入栅格文件路径",
            "band": "int: 波段索引（1-based），默认计算所有波段",
        },
        "returns": {"type": "dict", "description": "包含各波段 min/max/mean/std/histogram 等"},
    },
    {
        "name": "calculate_slope",
        "description": "从 DEM 栅格计算坡度（单位：度）",
        "category": "raster",
        "parameters": {
            "input_path": "str: DEM 栅格文件路径",
            "output_path": "str: 输出坡度栅格路径",
        },
        "returns": {"type": "dict", "description": "包含 output 和坡度统计"},
    },
    {
        "name": "calculate_hillshade",
        "description": "从 DEM 栅格生成山体阴影效果图",
        "category": "raster",
        "parameters": {
            "input_path": "str: DEM 栅格文件路径",
            "output_path": "str: 输出山体阴影栅格路径",
            "azimuth": "float: 光源方位角（度），默认 315",
            "altitude": "float: 光源高度角（度），默认 45",
        },
        "returns": {"type": "dict", "description": "包含 output 路径"},
    },
    {
        "name": "extract_raster_values",
        "description": "提取栅格在指定点坐标处的像元值",
        "category": "raster",
        "parameters": {
            "input_path": "str: 输入栅格文件路径",
            "points_path": "str: 点矢量文件路径",
            "output_path": "str: 输出带提取值的矢量路径",
            "field_name": "str: 存储提取值的字段名，默认 raster_val",
        },
        "returns": {"type": "dict", "description": "包含 output 和要素数量"},
    },
    {
        "name": "reclassify_raster",
        "description": "按区间规则重分类栅格值，如将高程分为低/中/高",
        "category": "raster",
        "parameters": {
            "input_path": "str: 输入栅格文件路径",
            "rules": "str: JSON 格式的重分类规则，如 [[0,100,1],[100,200,2],[200,9999,3]]，每项为 [min,max,new_value]",
            "output_path": "str: 输出栅格路径",
            "band": "int: 波段索引（1-based），默认 1",
        },
        "returns": {"type": "dict", "description": "包含 output 和重分类后的唯一值列表"},
    },
]


def _validate_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")


def clip_raster(input_path, clip_path, output_path):
    _validate_file(input_path)
    _validate_file(clip_path)

    import geopandas as gpd
    clip_gdf = gpd.read_file(clip_path)

    with rasterio.open(input_path) as src:
        clip_gdf = clip_gdf.to_crs(src.crs)
        geometries = clip_gdf.geometry.values
        out_image, out_transform = mask.mask(src, geometries, crop=True, nodata=src.nodata)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
        })
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)

    return {
        "output": output_path,
        "width": out_meta["width"],
        "height": out_meta["height"],
        "crs": str(src.crs),
    }


def reproject_raster(input_path, target_epsg, output_path, resampling="nearest"):
    _validate_file(input_path)

    resamp = RESAMPLING_MAP.get(resampling, Resampling.nearest)

    with rasterio.open(input_path) as src:
        dst_crs = f"EPSG:{target_epsg}"
        transform_dst, width_dst, height_dst = warp.calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": dst_crs,
            "transform": transform_dst,
            "width": width_dst,
            "height": height_dst,
            "driver": "GTiff",
        })

        with rasterio.open(output_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                warp.reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform_dst,
                    dst_crs=dst_crs,
                    resampling=resamp,
                )

    return {
        "output": output_path,
        "target_crs": dst_crs,
        "width": width_dst,
        "height": height_dst,
    }


def resample_raster(input_path, scale_factor, output_path, method="bilinear"):
    _validate_file(input_path)

    resamp = RESAMPLING_MAP.get(method, Resampling.bilinear)

    with rasterio.open(input_path) as src:
        new_height = int(src.height * scale_factor)
        new_width = int(src.width * scale_factor)
        new_transform = src.transform * src.transform.scale(
            1.0 / scale_factor, 1.0 / scale_factor
        )

        data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=resamp,
        )

        kwargs = src.meta.copy()
        kwargs.update({
            "height": new_height,
            "width": new_width,
            "transform": new_transform,
            "driver": "GTiff",
        })

        with rasterio.open(output_path, "w", **kwargs) as dst:
            dst.write(data)

    return {
        "output": output_path,
        "new_resolution": (abs(new_transform[0]), abs(new_transform[4])),
        "width": new_width,
        "height": new_height,
    }


def get_raster_info(input_path):
    _validate_file(input_path)

    with rasterio.open(input_path) as src:
        info = {
            "width": src.width,
            "height": src.height,
            "band_count": src.count,
            "crs": str(src.crs) if src.crs else None,
            "transform": list(src.transform),
            "resolution": (abs(src.transform[0]), abs(src.transform[4])),
            "bounds": {
                "left": src.bounds.left,
                "bottom": src.bounds.bottom,
                "right": src.bounds.right,
                "top": src.bounds.top,
            },
            "nodata": src.nodata,
            "dtype": str(src.dtypes[0]),
            "file_path": input_path,
        }

        info["band_stats"] = {}
        for i in range(1, src.count + 1):
            band = src.read(i, masked=True)
            info["band_stats"][f"band_{i}"] = {
                "min": float(band.min()) if band.count() > 0 else None,
                "max": float(band.max()) if band.count() > 0 else None,
                "mean": float(band.mean()) if band.count() > 0 else None,
                "std": float(band.std()) if band.count() > 0 else None,
            }

    return info


def raster_calculator(input_path, expression, output_path, nodata=None):
    _validate_file(input_path)

    with rasterio.open(input_path) as src:
        bands = {}
        for i in range(1, src.count + 1):
            bands[f"band_{i}"] = src.read(i).astype(np.float64)

        try:
            result = eval(expression, {"__builtins__": {}}, {"np": np, **bands})
        except Exception as e:
            raise ValueError(f"Expression evaluation failed: {e}")

        if nodata is not None:
            result = np.where(np.isnan(result), nodata, result)
        else:
            result = np.where(np.isnan(result), src.nodata or -9999, result)

        kwargs = src.meta.copy()
        kwargs.update({
            "count": 1,
            "dtype": result.dtype.name,
            "driver": "GTiff",
            "nodata": float(nodata) if nodata is not None else (src.nodata or -9999),
        })

        with rasterio.open(output_path, "w", **kwargs) as dst:
            dst.write(result.astype(kwargs["dtype"]), 1)

    return {
        "output": output_path,
        "min": float(result.min()),
        "max": float(result.max()),
        "mean": float(result.mean()),
    }


def raster_to_vector(input_path, output_path, band=1):
    _validate_file(input_path)

    import geopandas as gpd
    from shapely.geometry import shape

    with rasterio.open(input_path) as src:
        image = src.read(band)
        mask_img = image != src.nodata if src.nodata is not None else image != 0
        shapes = list(features.shapes(image, mask=mask_img, transform=src.transform))

    records = []
    for geom_dict, value in shapes:
        records.append({
            "geometry": shape(geom_dict),
            "value": value,
        })

    gdf = gpd.GeoDataFrame(records, crs=src.crs)
    gdf.to_file(output_path)

    return {
        "output": output_path,
        "feature_count": len(gdf),
    }


def merge_rasters(input_dir, output_path, resampling="nearest"):
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Directory not found: {input_dir}")

    tif_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith((".tif", ".tiff", ".img"))
    ]
    if not tif_files:
        raise ValueError(f"No raster files found in {input_dir}")

    resamp = RESAMPLING_MAP.get(resampling, Resampling.nearest)

    src_files = [rasterio.open(f) for f in tif_files]
    try:
        merged, out_transform = warp.merge(src_files, resampling=resamp)
        out_meta = src_files[0].meta.copy()
        out_meta.update({
            "height": merged.shape[1],
            "width": merged.shape[2],
            "transform": out_transform,
            "driver": "GTiff",
        })

        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(merged)
    finally:
        for f in src_files:
            f.close()

    return {
        "output": output_path,
        "width": out_meta["width"],
        "height": out_meta["height"],
    }


def raster_statistics(input_path, band=None):
    _validate_file(input_path)

    with rasterio.open(input_path) as src:
        band_indices = [band] if band is not None else list(range(1, src.count + 1))
        stats = {}
        for bidx in band_indices:
            data = src.read(bidx, masked=True)
            valid = data.compressed()
            hist, bins = np.histogram(valid, bins=20) if len(valid) > 0 else ([], [])
            stats[f"band_{bidx}"] = {
                "min": float(valid.min()) if len(valid) > 0 else None,
                "max": float(valid.max()) if len(valid) > 0 else None,
                "mean": float(valid.mean()) if len(valid) > 0 else None,
                "std": float(valid.std()) if len(valid) > 0 else None,
                "median": float(np.median(valid)) if len(valid) > 0 else None,
                "valid_pixels": len(valid),
                "total_pixels": data.count(),
                "nodata_pixels": data.count() - len(valid),
            }

    return stats


def calculate_slope(input_path, output_path):
    _validate_file(input_path)

    res_x = None
    res_y = None

    with rasterio.open(input_path) as src:
        dem = src.read(1).astype(np.float64)
        res_x = abs(src.transform[0])
        res_y = abs(src.transform[4])
        nodata = src.nodata
        meta = src.meta.copy()

    if nodata is not None:
        dem = np.where(dem == nodata, np.nan, dem)

    dz_dx = np.gradient(dem, axis=1) / res_x
    dz_dy = np.gradient(dem, axis=0) / res_y

    slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    slope_deg = np.degrees(slope_rad)

    slope_deg = np.where(np.isnan(dem), nodata or -9999, slope_deg).astype(np.float32)

    meta.update({
        "dtype": "float32",
        "nodata": float(nodata or -9999),
    })

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(slope_deg, 1)

    valid = slope_deg[~np.isnan(dem)] if nodata is not None else slope_deg[slope_deg != meta["nodata"]]
    return {
        "output": output_path,
        "slope_min": float(np.min(valid)) if len(valid) > 0 else None,
        "slope_max": float(np.max(valid)) if len(valid) > 0 else None,
        "slope_mean": float(np.mean(valid)) if len(valid) > 0 else None,
    }


def calculate_hillshade(input_path, output_path, azimuth=315.0, altitude=45.0):
    _validate_file(input_path)

    res_x = None
    res_y = None

    with rasterio.open(input_path) as src:
        dem = src.read(1).astype(np.float64)
        res_x = abs(src.transform[0])
        res_y = abs(src.transform[4])
        nodata = src.nodata
        meta = src.meta.copy()

    if nodata is not None:
        dem = np.where(dem == nodata, np.nan, dem)

    azimuth_rad = np.deg2rad(360.0 - azimuth + 90.0)
    altitude_rad = np.deg2rad(altitude)

    dz_dx = np.gradient(dem, axis=1) / res_x
    dz_dy = np.gradient(dem, axis=0) / res_y

    slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    aspect_rad = np.arctan2(-dz_dy, dz_dx)

    hillshade = (
        np.cos(altitude_rad) * np.cos(slope_rad)
        + np.sin(altitude_rad) * np.sin(slope_rad)
        * np.cos(azimuth_rad - aspect_rad)
    )
    hillshade = np.clip(hillshade * 255, 0, 255)

    hillshade = np.where(np.isnan(dem), 0, hillshade).astype(np.uint8)

    meta.update({
        "dtype": "uint8",
        "nodata": 0,
    })

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(hillshade, 1)

    return {"output": output_path}


def extract_raster_values(input_path, points_path, output_path, field_name="raster_val"):
    _validate_file(input_path)
    _validate_file(points_path)

    import geopandas as gpd

    gdf = gpd.read_file(points_path)

    with rasterio.open(input_path) as src:
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        values = []
        for geom in gdf.geometry:
            try:
                row, col = src.index(geom.x, geom.y)
                val = src.read(1)[row, col]
                values.append(float(val))
            except (IndexError, Exception):
                values.append(None)

    gdf[field_name] = values
    gdf.to_file(output_path)

    return {
        "output": output_path,
        "feature_count": len(gdf),
        "valid_count": sum(1 for v in values if v is not None),
    }


def reclassify_raster(input_path, rules, output_path, band=1):
    _validate_file(input_path)

    try:
        rules_list = json.loads(rules)
    except json.JSONDecodeError:
        raise ValueError("rules must be a valid JSON array of [min, max, new_value]")

    with rasterio.open(input_path) as src:
        data = src.read(band).astype(np.float64)
        nodata = src.nodata
        meta = src.meta.copy()

        result = np.full_like(data, nodata if nodata is not None else -9999, dtype=np.int32)
        for rule in rules_list:
            if len(rule) < 3:
                continue
            rmin, rmax, new_val = float(rule[0]), float(rule[1]), int(rule[2])
            if nodata is not None:
                mask_valid = (data != nodata) & (data >= rmin) & (data < rmax)
            else:
                mask_valid = (data >= rmin) & (data < rmax)
            result[mask_valid] = new_val

    meta.update({
        "dtype": "int32",
        "nodata": int(nodata or -9999),
        "count": 1,
    })

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(result, 1)

    unique_vals = [int(v) for v in np.unique(result) if v != meta["nodata"]]
    return {
        "output": output_path,
        "unique_values": unique_vals,
        "rule_count": len(rules_list),
    }
