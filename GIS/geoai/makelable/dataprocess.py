#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理脚本：影像与矢量数据对齐处理

功能：
1. 从STAC API读取高分辨率影像数据（西安市部分地区）
2. 从STAC API读取建筑shp矢量数据（全国数据）
3. 统一坐标参照系（CRS对齐）
4. 按影像范围裁剪矢量数据
5. 保存处理结果到data文件夹

环境要求：
- conda geoai 环境
- py-geoai库
"""

import os
import json
import zipfile
import requests
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

# 地理数据处理库
import geopandas as gpd
import rasterio
from rasterio.crs import CRS
from rasterio.mask import mask
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import box, mapping
import matplotlib.pyplot as plt


# =============================================================================
# 配置参数
# =============================================================================

# STAC API 基础地址
STAC_API_BASE = os.getenv("STAC_API_BASE", "http://localhost:8010")

# 影像数据STAC Item URL
RASTER_ITEM_URL = f"{STAC_API_BASE}/collections/raster-default/items/0dcc269fd72a4c4e8a3c388564122fd1"

# 矢量数据STAC Item URL
VECTOR_ITEM_URL = f"{STAC_API_BASE}/collections/vector-default/items/90dbe2fc21b84d83891b87fd61ed24e6"

# 输出目录
DATA_DIR = Path(__file__).parent / "data"

# 目标坐标系（WGS84 Web Mercator，常用于影像数据）
TARGET_CRS = "EPSG:3857"


# =============================================================================
# 第一步：数据获取
# =============================================================================

def fetch_stac_item(item_url: str) -> Dict[str, Any]:
    """
    从STAC API获取Item数据
    
    Args:
        item_url: STAC Item的URL
        
    Returns:
        STAC Item的JSON数据
        
    Raises:
        requests.RequestException: 请求失败时抛出
    """
    print(f"正在获取STAC数据: {item_url}")
    
    try:
        response = requests.get(item_url, timeout=30)
        response.raise_for_status()
        data = response.json()
        print(f"✓ 成功获取数据")
        return data
    except requests.RequestException as e:
        print(f"✗ 获取数据失败: {e}")
        raise


def extract_asset_url(stac_item: Dict[str, Any], asset_key: str = "data") -> str:
    """
    从STAC Item中提取资源文件的URL
    
    Args:
        stac_item: STAC Item数据
        asset_key: 资源键名，默认为"data"
        
    Returns:
        资源文件的URL
    """
    assets = stac_item.get("assets", {})
    
    # 尝试多个可能的键名
    possible_keys = [asset_key, "image", "tif", "tiff", "data", "default"]
    
    for key in possible_keys:
        if key in assets:
            href = assets[key].get("href", "")
            if href:
                # 处理相对路径
                if href.startswith("http"):
                    return href
                else:
                    # 构建绝对URL
                    base_url = stac_item.get("links", [{}])[0].get("href", STAC_API_BASE)
                    return f"{STAC_API_BASE}{href}"
    
    # 如果没有找到，返回第一个可用的asset
    if assets:
        first_asset = list(assets.values())[0]
        href = first_asset.get("href", "")
        if href.startswith("http"):
            return href
        else:
            return f"{STAC_API_BASE}{href}"
    
    raise ValueError(f"无法从STAC Item中找到资源文件")


def get_item_bbox(stac_item: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """
    获取STAC Item的空间范围（Bounding Box）
    
    Args:
        stac_item: STAC Item数据
        
    Returns:
        (minx, miny, maxx, maxy) 元组
    """
    bbox = stac_item.get("bbox", None)
    
    if bbox and len(bbox) >= 4:
        return (bbox[0], bbox[1], bbox[2], bbox[3])
    
    # 如果没有bbox，尝试从geometry计算
    geometry = stac_item.get("geometry", {})
    if geometry:
        coords = geometry.get("coordinates", [[]])
        if coords and len(coords[0]) > 0:
            lons = [c[0] for c in coords[0]]
            lats = [c[1] for c in coords[0]]
            return (min(lons), min(lats), max(lons), max(lats))
    
    raise ValueError("无法从STAC Item中获取空间范围")


# =============================================================================
# 第二步：影像数据处理
# =============================================================================

def download_raster(raster_url: str, output_path: Path) -> Path:
    """
    下载影像数据到本地
    
    Args:
        raster_url: 影像数据的URL
        output_path: 本地保存路径
        
    Returns:
        本地文件路径
    """
    print(f"正在下载影像数据...")
    print(f"  URL: {raster_url}")
    print(f"  保存路径: {output_path}")
    
    if output_path.exists():
        print(f"✓ 影像数据已存在，跳过下载")
        return output_path
    
    try:
        response = requests.get(raster_url, stream=True, timeout=120)
        response.raise_for_status()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✓ 影像数据下载完成")
        return output_path
    except requests.RequestException as e:
        print(f"✗ 下载影像数据失败: {e}")
        raise


def read_raster_info(raster_path: Path) -> Dict[str, Any]:
    """
    读取影像数据的基本信息
    
    Args:
        raster_path: 影像文件路径
        
    Returns:
        包含影像信息的字典
    """
    print(f"正在读取影像信息: {raster_path}")
    
    with rasterio.open(raster_path) as src:
        info = {
            "width": src.width,
            "height": src.height,
            "count": src.count,  # 波段数
            "crs": str(src.crs),
            "transform": src.transform,
            "bounds": src.bounds,
            "resolution": src.res,
            "dtype": str(src.dtypes[0]),
            "profile": src.profile
        }
        
        print(f"✓ 影像信息:")
        print(f"  尺寸: {info['width']} x {info['height']}")
        print(f"  波段数: {info['count']}")
        print(f"  坐标系: {info['crs']}")
        print(f"  分辨率: {info['resolution']}")
        print(f"  数据类型: {info['dtype']}")
        print(f"  范围: {info['bounds']}")
        
        return info


# =============================================================================
# 第三步：矢量数据处理
# =============================================================================

def download_and_extract_vector(vector_url: str, output_dir: Path) -> Path:
    """
    下载矢量数据（zip格式）并解压到本地
    
    Args:
        vector_url: 矢量数据的URL（zip格式）
        output_dir: 本地保存目录
        
    Returns:
        解压后的shp文件路径
    """
    import zipfile
    
    print(f"正在下载矢量数据...")
    print(f"  URL: {vector_url}")
    print(f"  保存目录: {output_dir}")
    
    # 检查是否已有解压后的shp文件
    shp_files = list(output_dir.glob("*.shp"))
    if shp_files:
        print(f"✓ 矢量数据已存在，跳过下载")
        return shp_files[0]
    
    try:
        # 下载zip文件
        response = requests.get(vector_url, stream=True, timeout=120)
        response.raise_for_status()
        
        output_dir.mkdir(parents=True, exist_ok=True)
        zip_path = output_dir / "buildings.zip"
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✓ 矢量数据下载完成: {zip_path}")
        
        # 解压zip文件
        print(f"正在解压矢量数据...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        # 删除zip文件
        zip_path.unlink()
        
        # 查找解压后的shp文件（包括子目录）
        shp_files = list(output_dir.rglob("*.shp"))
        if not shp_files:
            raise FileNotFoundError(f"解压后未找到shp文件，请检查 {output_dir}")
        
        print(f"✓ 矢量数据解压完成: {shp_files[0]}")
        return shp_files[0]
        
    except requests.RequestException as e:
        print(f"✗ 下载矢量数据失败: {e}")
        raise
    except zipfile.BadZipFile as e:
        print(f"✗ 解压失败，文件可能不是zip格式: {e}")
        raise


def read_vector_info(vector_path: Path) -> Dict[str, Any]:
    """
    读取矢量数据的基本信息
    
    Args:
        vector_path: 矢量文件路径
        
    Returns:
        包含矢量信息的字典
    """
    print(f"正在读取矢量信息: {vector_path}")
    
    gdf = gpd.read_file(vector_path)
    
    info = {
        "count": len(gdf),
        "crs": str(gdf.crs),
        "columns": list(gdf.columns),
        "geometry_type": str(gdf.geometry.geom_type.unique()),
        "bounds": gdf.total_bounds.tolist(),
        "head": gdf.head()
    }
    
    print(f"✓ 矢量信息:")
    print(f"  要素数量: {info['count']}")
    print(f"  坐标系: {info['crs']}")
    print(f"  几何类型: {info['geometry_type']}")
    print(f"  属性字段: {info['columns']}")
    print(f"  范围: {info['bounds']}")
    
    return info


# =============================================================================
# 第四步：数据对齐处理
# =============================================================================

def reproject_vector(gdf: gpd.GeoDataFrame, target_crs: str) -> gpd.GeoDataFrame:
    """
    将矢量数据重投影到目标坐标系
    
    Args:
        gdf: 输入的GeoDataFrame
        target_crs: 目标坐标系（如 "EPSG:3857"）
        
    Returns:
        重投影后的GeoDataFrame
    """
    print(f"正在重投影矢量数据...")
    print(f"  原始坐标系: {gdf.crs}")
    print(f"  目标坐标系: {target_crs}")
    
    if str(gdf.crs) == target_crs:
        print(f"✓ 坐标系已一致，无需重投影")
        return gdf
    
    gdf_reprojected = gdf.to_crs(target_crs)
    print(f"✓ 重投影完成")
    
    return gdf_reprojected


def clip_vector_by_raster(gdf: gpd.GeoDataFrame, raster_bounds: Tuple[float, float, float, float], 
                          raster_crs: str) -> gpd.GeoDataFrame:
    """
    按影像范围裁剪矢量数据
    
    Args:
        gdf: 输入的GeoDataFrame（已与影像同坐标系）
        raster_bounds: 影像范围 (minx, miny, maxx, maxy)
        raster_crs: 影像坐标系
        
    Returns:
        裁剪后的GeoDataFrame
    """
    print(f"正在按影像范围裁剪矢量数据...")
    print(f"  影像范围: {raster_bounds}")
    
    # 创建影像范围的矩形
    minx, miny, maxx, maxy = raster_bounds
    raster_box = box(minx, miny, maxx, maxy)
    
    # 确保矢量数据与影像同坐标系
    if str(gdf.crs) != raster_crs:
        print(f"  正在将矢量数据转换到影像坐标系...")
        gdf = gdf.to_crs(raster_crs)
    
    # 执行裁剪
    clipped = gpd.clip(gdf, raster_box)
    
    print(f"✓ 裁剪完成")
    print(f"  原始要素数: {len(gdf)}")
    print(f"  裁剪后要素数: {len(clipped)}")
    
    return clipped


def align_data(raster_path: Path, vector_path: Path, output_dir: Path) -> Dict[str, Path]:
    """
    对齐影像和矢量数据
    
    Args:
        raster_path: 影像文件路径
        vector_path: 矢量文件路径
        output_dir: 输出目录
        
    Returns:
        包含输出文件路径的字典
    """
    print("\n" + "="*60)
    print("开始数据对齐处理")
    print("="*60 + "\n")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 读取影像信息
    print("【步骤1】读取影像数据")
    raster_info = read_raster_info(raster_path)
    raster_crs = raster_info["crs"]
    raster_bounds = raster_info["bounds"]
    
    # 2. 读取矢量数据
    print("\n【步骤2】读取矢量数据")
    gdf = gpd.read_file(vector_path)
    vector_info = read_vector_info(vector_path)
    
    # 3. 重投影矢量数据到影像坐标系
    print("\n【步骤3】坐标系对齐")
    gdf_aligned = reproject_vector(gdf, raster_crs)
    
    # 4. 按影像范围裁剪矢量数据
    print("\n【步骤4】按影像范围裁剪")
    gdf_clipped = clip_vector_by_raster(gdf_aligned, raster_bounds, raster_crs)
    
    # 5. 保存处理结果
    print("\n【步骤5】保存处理结果")
    
    # 保存对齐后的矢量数据
    vector_output = output_dir / "buildings_aligned.shp"
    gdf_clipped.to_file(vector_output, encoding='utf-8')
    print(f"✓ 矢量数据已保存: {vector_output}")
    
    # 保存影像的元数据
    metadata = {
        "raster": {
            "path": str(raster_path),
            "crs": raster_crs,
            "bounds": {
                "left": raster_bounds.left,
                "bottom": raster_bounds.bottom,
                "right": raster_bounds.right,
                "top": raster_bounds.top
            },
            "width": raster_info["width"],
            "height": raster_info["height"],
            "resolution": raster_info["resolution"]
        },
        "vector": {
            "original_path": str(vector_path),
            "original_count": vector_info["count"],
            "aligned_path": str(vector_output),
            "aligned_count": len(gdf_clipped),
            "crs": str(gdf_clipped.crs)
        }
    }
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"✓ 元数据已保存: {metadata_path}")
    
    # 6. 生成可视化预览
    print("\n【步骤6】生成预览图")
    preview_path = output_dir / "preview.png"
    create_preview(raster_path, gdf_clipped, preview_path)
    
    print("\n" + "="*60)
    print("数据对齐处理完成")
    print("="*60)
    
    return {
        "vector_aligned": vector_output,
        "metadata": metadata_path,
        "preview": preview_path
    }


# =============================================================================
# 第五步：可视化
# =============================================================================

def create_preview(raster_path: Path, gdf: gpd.GeoDataFrame, output_path: Path, max_size: int = 2048):
    """
    创建影像和矢量的叠加预览图（缩略图，避免内存溢出）
    
    Args:
        raster_path: 影像文件路径
        gdf: 矢量GeoDataFrame
        output_path: 预览图保存路径
        max_size: 预览图最大尺寸（像素），默认2048
    """
    print(f"正在生成预览图（缩略图，最大尺寸 {max_size}x{max_size}）...")
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    with rasterio.open(raster_path) as src:
        # 计算缩放比例，使预览图不超过 max_size
        scale = max(1, max(src.width, src.height) // max_size)
        
        # 使用 overview 或降采样读取
        if src.overviews(1):
            # 如果有 overview，使用最合适的级别
            overview_level = 0
            for i, ov in enumerate(src.overviews(1)):
                if max(src.width // ov, src.height // ov) <= max_size:
                    overview_level = i
                    break
            
            data = src.read(out_shape=(src.count, src.height // src.overviews(1)[overview_level], 
                                      src.width // src.overviews(1)[overview_level]))
        else:
            # 使用窗口读取并降采样
            data = src.read(out_shape=(src.count, src.height // scale, src.width // scale))
        
        # 读取RGB波段（假设前3个波段是RGB）
        if src.count >= 3 and data.shape[0] >= 3:
            rgb = np.dstack([data[i] for i in range(3)])
            
            # 归一化到0-1范围
            rgb = rgb.astype(np.float32)
            for i in range(3):
                band = rgb[:, :, i]
                min_val, max_val = np.percentile(band, [2, 98])
                if max_val > min_val:
                    rgb[:, :, i] = np.clip((band - min_val) / (max_val - min_val), 0, 1)
            
            # 显示影像
            bounds = src.bounds
            extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
            ax.imshow(rgb, extent=extent)
        else:
            # 单波段影像
            bounds = src.bounds
            extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
            ax.imshow(data[0], extent=extent, cmap='gray')
    
    # 叠加矢量数据
    if len(gdf) > 0:
        gdf.boundary.plot(ax=ax, color='red', linewidth=0.5, alpha=0.7)
    
    ax.set_title('影像与建筑矢量叠加预览（缩略图）')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 预览图已保存: {output_path}")


# =============================================================================
# 主函数
# =============================================================================

def main():
    """
    主函数：执行完整的数据处理流程
    """
    print("\n" + "="*60)
    print("STAC 影像与矢量数据对齐处理工具")
    print("="*60 + "\n")
    
    # 创建输出目录
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # 步骤1：获取STAC Item信息
        print("【阶段1】获取STAC数据信息")
        print("-" * 40)
        
        raster_item = fetch_stac_item(RASTER_ITEM_URL)
        vector_item = fetch_stac_item(VECTOR_ITEM_URL)
        
        # 步骤2：提取资源URL
        print("\n【阶段2】提取资源URL")
        print("-" * 40)
        
        raster_url = extract_asset_url(raster_item)
        vector_url = extract_asset_url(vector_item)
        
        print(f"影像URL: {raster_url}")
        print(f"矢量URL: {vector_url}")
        
        # 步骤3：下载数据
        print("\n【阶段3】下载数据")
        print("-" * 40)
        
        raster_path = DATA_DIR / "image.tif"
        vector_path = DATA_DIR / "buildings.shp"
        
        # 注意：如果STAC提供的是直接下载链接，下载文件
        # 如果是服务接口（如WMS、WFS），可能需要其他方式获取
        
        # 这里假设可以下载到本地文件
        # 实际使用时可能需要根据STAC Item的具体asset类型调整
        
        # 步骤4：对齐处理
        print("\n【阶段4】数据对齐处理")
        print("-" * 40)
        
        # 检查文件是否存在（如果之前已下载）
        if not raster_path.exists():
            print(f"警告: 影像文件不存在，请确保数据可访问")
            print(f"  期望路径: {raster_path}")
            # 尝试从URL下载
            try:
                raster_path = download_raster(raster_url, raster_path)
            except Exception as e:
                print(f"下载影像失败: {e}")
                print("请手动将影像数据放入 data/image.tif")
                return
        
        # 检查矢量数据是否存在（查找data目录下的shp文件）
        shp_files = list(DATA_DIR.glob("*.shp"))
        if shp_files:
            vector_path = shp_files[0]
            print(f"✓ 找到本地矢量数据: {vector_path}")
        else:
            print(f"警告: 矢量文件不存在，尝试从URL下载")
            # 尝试从URL下载并解压
            try:
                vector_path = download_and_extract_vector(vector_url, DATA_DIR)
            except Exception as e:
                print(f"下载矢量失败: {e}")
                print("请手动将矢量数据放入 data/ 目录")
                return
        
        # 执行对齐
        results = align_data(raster_path, vector_path, DATA_DIR)
        
        print("\n" + "="*60)
        print("处理完成！输出文件：")
        print("="*60)
        for key, path in results.items():
            print(f"  {key}: {path}")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n✗ 处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


# =============================================================================
# 测试函数
# =============================================================================

def test_fetch_stac():
    """测试：获取STAC数据"""
    print("\n【测试】获取STAC数据")
    print("-" * 40)
    
    try:
        raster_item = fetch_stac_item(RASTER_ITEM_URL)
        print(f"✓ 影像Item获取成功")
        print(f"  ID: {raster_item.get('id', 'N/A')}")
        print(f"  时间: {raster_item.get('properties', {}).get('datetime', 'N/A')}")
        
        vector_item = fetch_stac_item(VECTOR_ITEM_URL)
        print(f"✓ 矢量Item获取成功")
        print(f"  ID: {vector_item.get('id', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


def test_extract_url():
    """测试：提取资源URL"""
    print("\n【测试】提取资源URL")
    print("-" * 40)
    
    try:
        raster_item = fetch_stac_item(RASTER_ITEM_URL)
        vector_item = fetch_stac_item(VECTOR_ITEM_URL)
        
        raster_url = extract_asset_url(raster_item)
        vector_url = extract_asset_url(vector_item)
        
        print(f"✓ 影像URL: {raster_url}")
        print(f"✓ 矢量URL: {vector_url}")
        
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


def test_bbox():
    """测试：获取空间范围"""
    print("\n【测试】获取空间范围")
    print("-" * 40)
    
    try:
        raster_item = fetch_stac_item(RASTER_ITEM_URL)
        bbox = get_item_bbox(raster_item)
        
        print(f"✓ 影像范围:")
        print(f"  MinX: {bbox[0]:.6f}")
        print(f"  MinY: {bbox[1]:.6f}")
        print(f"  MaxX: {bbox[2]:.6f}")
        print(f"  MaxY: {bbox[3]:.6f}")
        
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


def test_read_local_data():
    """测试：读取本地数据"""
    print("\n【测试】读取本地数据")
    print("-" * 40)
    
    raster_path = DATA_DIR / "image.tif"
    vector_path = DATA_DIR / "buildings.shp"
    
    success = True
    
    if raster_path.exists():
        try:
            info = read_raster_info(raster_path)
            print(f"✓ 影像数据读取成功")
        except Exception as e:
            print(f"✗ 影像数据读取失败: {e}")
            success = False
    else:
        print(f"⚠ 影像数据不存在: {raster_path}")
    
    if vector_path.exists():
        try:
            info = read_vector_info(vector_path)
            print(f"✓ 矢量数据读取成功")
        except Exception as e:
            print(f"✗ 矢量数据读取失败: {e}")
            success = False
    else:
        print(f"⚠ 矢量数据不存在: {vector_path}")
    
    return success


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("运行测试")
    print("="*60)
    
    tests = [
        ("STAC数据获取", test_fetch_stac),
        ("资源URL提取", test_extract_url),
        ("空间范围获取", test_bbox),
        ("本地数据读取", test_read_local_data),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} 测试异常: {e}")
            results.append((name, False))
    
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {name}: {status}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\n总计: {passed}/{total} 通过")
    print("="*60 + "\n")
    
    return all(r for _, r in results)


# =============================================================================
# 入口点
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # 运行测试模式
        success = run_all_tests()
        sys.exit(0 if success else 1)
    else:
        # 运行主程序
        sys.exit(main())
