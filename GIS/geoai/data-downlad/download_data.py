#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GeoAI 数据下载模块

该模块提供了以下功能：
1. NAIP 影像下载：从微软 Planetary Computer 下载美国国家农业影像计划 (NAIP) 数据
2. Overture Maps 数据下载：从 Overture Maps 下载建筑物、地址等开放地理空间数据
3. Planetary Computer STAC 数据处理：搜索、下载和处理 STAC (SpatioTemporal Asset Catalog) 项目
4. 带进度条的文件下载功能

数据保存位置：所有下载的文件都保存在项目根目录的 data 文件夹中
"""

import os
import sys
from typing import Tuple, List, Optional, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import geopandas as gpd
import requests
import pystac
from pystac_client import Client
from shapely.geometry import box
import planetary_computer as pc
import rioxarray as rxr
import xarray as xr
from tqdm import tqdm


# 设置工作目录为 data-downlad 的上级目录的 data 文件夹
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "./data")
# 创建数据目录（如果不存在）
os.makedirs(DATA_DIR, exist_ok=True)


def download_naip(
    bbox: Tuple[float, float, float, float],
    output_dir: str,
    year: Optional[int] = None,
    max_items: int = 10,
    overwrite: bool = False,
    preview: bool = False,
    **kwargs: Any,
) -> List[str]:
    """从 Planetary Computer 下载 NAIP 影像

    Args:
        bbox: 边界框坐标 (min_lon, min_lat, max_lon, max_lat)
        output_dir: 输出目录（相对于 DATA_DIR）
        year: 指定年份（可选）
        max_items: 最大下载数量
        overwrite: 是否覆盖现有文件
        preview: 是否预览下载的影像

    Returns:
        下载的文件路径列表
    """
    # 构建完整的输出目录路径
    output_dir = os.path.join(DATA_DIR, output_dir)
    print(out_dir)
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 创建边界框几何对象
    geometry = box(*bbox)

    # 连接到 Planetary Computer STAC API
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    # 构建搜索参数
    search_params = {
        "collections": ["naip"],  # NAIP 集合
        "intersects": geometry,    # 空间相交
        "limit": max_items,        # 每页返回的项目数
    }

    # 添加年份过滤器（如果指定）
    if year:
        search_params["query"] = {"naip:year": {"eq": year}}

    # 执行搜索
    search_results = catalog.search(**search_params)
    # 获取所有匹配的项目
    items = list(search_results.items())

    # 存储下载的文件路径
    downloaded_files = []
    
    # 遍历每个项目
    for i, item in enumerate(items):
        # 对项目进行签名（Planetary Computer 需要）
        signed_item = pc.sign(item)

        # 获取 RGB 资产（真彩色影像）
        rgb_asset = signed_item.assets.get("image")
        if not rgb_asset:
            print(f"警告: 项目 {i+1} 没有找到 RGB 资产")
            continue

        # 构建输出文件名
        original_filename = os.path.basename(rgb_asset.href.split("?")[0])
        output_path = os.path.join(output_dir, original_filename)

        # 下载文件
        if rgb_asset.href.startswith("http"):
            # 通过 HTTP 下载
            download_with_progress(rgb_asset.href, output_path)
        else:
            # 直接读取本地文件
            data = rxr.open_rasterio(rgb_asset.href)
            data.rio.to_raster(output_path)

        # 添加到下载列表
        downloaded_files.append(output_path)

    return downloaded_files


def download_overture_buildings(
    bbox: Tuple[float, float, float, float],
    output: str,
    overture_type: str = "building",
    **kwargs: Any,
) -> str:
    """从 Overture Maps 下载建筑物数据

    Args:
        bbox: 边界框坐标 (min_lon, min_lat, max_lon, max_lat)
        output: 输出文件路径（相对于 DATA_DIR）
        overture_type: Overture 数据类型（默认为 "building"）

    Returns:
        输出文件路径
    """
    # 构建完整的输出路径
    output_path = os.path.join(DATA_DIR, output)
    # 确保输出目录存在
    out_dir = os.path.dirname(output_path)
    os.makedirs(out_dir, exist_ok=True)

    try:
        # 尝试导入 overturemaps 模块
        from overturemaps import core
    except ImportError:
        # 导入失败时提示安装
        raise ImportError("需要安装 overturemaps 包: pip install overturemaps")

    # 从 Overture Maps 获取数据
    gdf = core.geodataframe(overture_type, bbox=bbox)
    # 设置坐标系为 WGS84
    gdf.crs = "EPSG:4326"
    # 保存数据到文件
    gdf.to_file(output_path, **kwargs)

    return output_path


def pc_stac_search(
    collection: str,
    bbox: Optional[List[float]] = None,
    time_range: Optional[str] = None,
    query: Optional[Dict[str, Any]] = None,
    limit: int = 10,
    max_items: Optional[int] = None,
    quiet: bool = False,
    endpoint: str = "https://planetarycomputer.microsoft.com/api/stac/v1",
) -> List["pystac.Item"]:
    """在 Planetary Computer 中搜索 STAC 项目

    Args:
        collection: STAC 集合 ID
        bbox: 边界框坐标 [西, 南, 东, 北]
        time_range: 时间范围，格式为 "start/end"
        query: 额外的查询参数
        limit: 每页返回的项目数
        max_items: 最大返回项目数
        quiet: 是否静默模式
        endpoint: STAC API 端点

    Returns:
        STAC 项目列表
    """
    # 初始化 STAC 客户端
    catalog = Client.open(endpoint)

    # 处理时间范围参数
    if time_range:
        if isinstance(time_range, tuple) and len(time_range) == 2:
            # 处理时间范围为元组的情况
            start, end = time_range
            if isinstance(start, datetime):
                start = start.isoformat()
            if isinstance(end, datetime):
                end = end.isoformat()
            time_str = f"{start}/{end}"
        elif isinstance(time_range, str):
            # 直接使用字符串时间范围
            time_str = time_range
        else:
            # 时间范围格式错误
            raise ValueError("time_range 必须是 'start/end' 字符串或 (start, end) 元组")
    else:
        time_str = None

    # 创建搜索对象
    search = catalog.search(
        collections=[collection],  # 要搜索的集合
        bbox=bbox,                # 边界框
        datetime=time_str,         # 时间范围
        query=query,               # 额外查询参数
        limit=limit                # 每页返回数量
    )

    # 收集搜索结果
    items = []
    if max_items:
        # 限制返回数量
        items_gen = search.get_items()
        for item in items_gen:
            items.append(item)
            if len(items) >= max_items:
                break
    else:
        # 返回所有结果
        items = list(search.get_items())

    return items


def pc_stac_download(
    items: Union["pystac.Item", List["pystac.Item"]],
    output_dir: str = ".",
    assets: Optional[List[str]] = None,
    max_workers: int = 1,
    skip_existing: bool = True,
) -> Dict[str, Dict[str, str]]:
    """下载 STAC 项目的资产

    Args:
        items: STAC 项目或项目列表
        output_dir: 输出目录（相对于 DATA_DIR）
        assets: 要下载的资产列表
        max_workers: 最大并发下载线程数
        skip_existing: 是否跳过已存在的文件

    Returns:
        资产路径映射字典，格式为 {item_id: {asset_key: path}}
    """
    # 构建完整的输出目录路径
    output_dir = os.path.join(DATA_DIR, output_dir)
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 处理单个项目的情况
    if isinstance(items, pystac.Item) or isinstance(items, str):
        items = [items]

    # 内部函数：下载单个资产
    def download_asset(item, asset_key, asset):
        # 对项目进行签名（Planetary Computer 需要）
        item = pc.sign(item)
        item_id = item.id

        # 获取资产 URL
        asset_url = item.assets[asset_key].href

        # 确定文件扩展名
        if asset.media_type:
            if "tiff" in str(asset.media_type) or "geotiff" in str(asset.media_type):
                ext = ".tif"
            elif "jpeg" in str(asset.media_type):
                ext = ".jpg"
            elif "png" in str(asset.media_type):
                ext = ".png"
            elif "json" in str(asset.media_type):
                ext = ".json"
            else:
                # 从 URL 中提取扩展名
                ext = os.path.splitext(asset_url.split("?")[0])[1] or ".data"
        else:
            # 从 URL 中提取扩展名
            ext = os.path.splitext(asset_url.split("?")[0])[1] or ".data"

        # 构建输出路径
        output_path = os.path.join(output_dir, f"{item_id}_{asset_key}{ext}")

        # 跳过已存在的文件
        if skip_existing and os.path.exists(output_path):
            return asset_key, output_path

        # 下载资产
        with requests.get(asset_url, stream=True) as r:
            # 检查请求是否成功
            r.raise_for_status()
            # 获取文件大小
            total_size = int(r.headers.get("content-length", 0))
            # 写入文件并显示进度条
            with open(output_path, "wb") as f:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=f"下载 {item_id}_{asset_key}",
                ) as pbar:
                    # 分块下载
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))

        return asset_key, output_path

    # 存储下载结果
    results = {}
    
    # 遍历每个项目
    for item in items:
        item_assets = {}
        # 处理项目 ID 为字符串的情况
        if isinstance(item, str):
            item = pystac.Item.from_file(item)
        item_id = item.id

        # 确定要下载的资产
        if assets:
            # 只下载指定的资产
            assets_to_download = {k: v for k, v in item.assets.items() if k in assets}
        else:
            # 下载所有资产
            assets_to_download = item.assets

        # 使用线程池并发下载资产
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交下载任务
            future_to_asset = {
                executor.submit(download_asset, item, asset_key, asset): (asset_key, asset)
                for asset_key, asset in assets_to_download.items()
            }

            # 处理完成的任务
            for future in as_completed(future_to_asset):
                asset_key, asset = future_to_asset[future]
                try:
                    key, path = future.result()
                    if path:
                        item_assets[key] = path
                except Exception as e:
                    print(f"错误: 处理资产 {asset_key} 时出错: {e}")

        # 保存该项目的下载结果
        results[item_id] = item_assets

    return results


def download_with_progress(
    url: str, output_path: str, max_size: Optional[int] = None
) -> None:
    """带进度条的文件下载

    Args:
        url: 文件 URL
        output_path: 输出路径
        max_size: 最大文件大小（可选）
    """
    # 发送 GET 请求，使用流式下载
    response = requests.get(url, stream=True)
    # 检查请求是否成功
    response.raise_for_status()
    # 获取文件大小
    total_size = int(response.headers.get("content-length", 0))

    # 检查文件大小是否超过限制
    if max_size is not None and total_size > max_size:
        response.close()
        raise ValueError(
            f"文件大小 ({total_size} bytes) 超过最大允许大小 ({max_size} bytes)."
        )

    # 分块大小（1KB）
    block_size = 1024

    # 打开文件并显示下载进度
    with (
        open(output_path, "wb") as file,
        tqdm(
            desc=os.path.basename(output_path),  # 进度条描述
            total=total_size,                   # 总大小
            unit="iB",                         # 单位
            unit_scale=True,                    # 自动缩放单位
            unit_divisor=1024,                  # 单位除数
        ) as bar,
    ):
        # 分块下载并写入文件
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)  # 更新进度条


def main():
    """主函数，支持选择下载内容"""
    # 显示欢迎信息和菜单
    print("=" * 60)
    print("GeoAI 数据下载工具")
    print("=" * 60)
    print("\n请选择要下载的数据类型:")
    print("1. NAIP 影像 (美国国家农业影像计划)")
    print("2. Overture 建筑物数据")
    print("3. Sentinel-2 数据 (STAC)")
    print("4. Landsat-8 数据 (STAC)")
    print("5. 退出")
    print("-" * 60)

    # 获取用户选择
    choice = input("请输入选项 (1-5): ").strip()

    # 处理用户选择
    if choice == "1":
        # 下载 NAIP 影像
        print("\n正在下载 NAIP 影像...")
        # 旧金山区域边界框
        bbox = (-122.51, 37.71, -122.41, 37.81)
        try:
            files = download_naip(
                bbox=bbox,
                output_dir="naip_data",  # 输出目录
                year=2020,               # 2020 年数据
                max_items=5,             # 最多下载 5 个项目
                overwrite=False,          # 不覆盖现有文件
                preview=True              # 预览下载的影像
            )
            print(f"\n下载完成！共 {len(files)} 个文件:")
            for f in files:
                print(f"  - {f}")
        except Exception as e:
            print(f"下载失败: {e}")

    elif choice == "2":
        # 下载 Overture 建筑物数据
        print("\n正在下载 Overture 建筑物数据...")
        # 纽约曼哈顿区域边界框
        bbox = (-74.01, 40.70, -73.99, 40.72)
        try:
            output_file = download_overture_buildings(
                bbox=bbox,
                output="overture_buildings/manhattan_buildings.geojson"  # 输出文件
            )
            print(f"\n下载完成！文件保存至: {output_file}")
        except ImportError as e:
            print(f"需要安装 overturemaps 包: {e}")
        except Exception as e:
            print(f"下载失败: {e}")

    elif choice == "3":
        # 下载 Sentinel-2 数据
        print("\n正在搜索和下载 Sentinel-2 数据...")
        # 旧金山区域边界框
        bbox = [-122.4, 37.7, -122.3, 37.8]
        try:
            # 搜索 Sentinel-2 数据
            items = pc_stac_search(
                collection="sentinel-2-l2a",        # Sentinel-2 L2A 集合
                bbox=bbox,                         # 边界框
                time_range="2023-01-01/2023-01-31", # 2023 年 1 月
                max_items=2                         # 最多 2 个项目
            )
            print(f"找到 {len(items)} 个 Sentinel-2 项目")

            # 下载数据
            if items:
                downloaded = pc_stac_download(
                    items=items,
                    output_dir="sentinel2_data",  # 输出目录
                    assets=["B02", "B03", "B04"],  # 蓝、绿、红波段
                    max_workers=3                # 3 个并发线程
                )
                print("\n下载完成:")
                for item_id, assets in downloaded.items():
                    print(f"  项目 {item_id}:")
                    for asset, path in assets.items():
                        print(f"    - {asset}: {path}")
        except Exception as e:
            print(f"下载失败: {e}")

    elif choice == "4":
        # 下载 Landsat-8 数据
        print("\n正在搜索和下载 Landsat-8 数据...")
        # 旧金山区域边界框
        bbox = [-122.4, 37.7, -122.3, 37.8]
        try:
            # 搜索 Landsat-8 数据
            items = pc_stac_search(
                collection="landsat-c2-l2",         # Landsat Collection 2 Level-2
                bbox=bbox,                         # 边界框
                time_range="2023-01-01/2023-01-31", # 2023 年 1 月
                max_items=1                         # 最多 1 个项目
            )
            print(f"找到 {len(items)} 个 Landsat-8 项目")

            # 下载数据
            if items:
                downloaded = pc_stac_download(
                    items=items,
                    output_dir="landsat8_data",  # 输出目录
                    assets=["SR_B4", "SR_B5"],    # 红、近红外波段
                    max_workers=2                # 2 个并发线程
                )
                print("\n下载完成:")
                for item_id, assets in downloaded.items():
                    print(f"  项目 {item_id}:")
                    for asset, path in assets.items():
                        print(f"    - {asset}: {path}")
        except Exception as e:
            print(f"下载失败: {e}")

    elif choice == "5":
        # 退出程序
        print("\n退出程序")
        sys.exit(0)

    else:
        # 无效选项
        print("\n无效的选项，请重新运行程序并选择 1-5")


if __name__ == "__main__":
    # 运行主函数
    main()
