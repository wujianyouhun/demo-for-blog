import rasterio
import geoai

import geopandas as gpd
import os

def get_raster_metadata_raw(raster_path: str) -> dict:
    """
    直接使用 rasterio 提取栅格元数据
    
    参数:
        raster_path: 栅格文件路径
    
    返回:
        包含元数据的字典
    """
    # 打开栅格文件（不加载像素数据）
    with rasterio.open(raster_path) as src:
        metadata = {
            "file_path": raster_path,
            "driver": src.driver,           # 文件格式（如 GTiff）
            "width": src.width,             # 宽度（像素）
            "height": src.height,           # 高度（像素）
            "count": src.count,             # 波段数
            "dtype": src.dtypes[0],         # 数据类型
            "nodata": src.nodata,           # 无数据值
            "crs": str(src.crs),            # 坐标参考系统
            "transform": src.transform,     # 地理变换矩阵
            "bounds": src.bounds,           # 地理边界
            "resolution": (src.transform[0], abs(src.transform[4])),  # 分辨率
        }
    print(f"驱动: {metadata['driver']}")
    print(f"尺寸: {metadata['width']} x {metadata['height']}")
    print(f"波段数: {metadata['count']}")
    print(f"坐标系统: {metadata['crs']}")
    print(f"分辨率: {metadata['resolution']}")    
    return metadata

def get_metadata_geoai(raster_path):
    # 方式 1：使用 read_raster_metadata 获取结构化元数据
    metadata = geoai.read_raster_metadata(raster_path)

    print(f"驱动: {metadata.driver}")
    print(f"尺寸: {metadata.width} x {metadata.height}")
    print(f"波段数: {metadata.count}")
    print(f"坐标系统: {metadata.crs}")
    print(f"数据类型: {metadata.dtype}")
    print(f"无数据值: {metadata.nodata}")
    print(f"地理边界: {metadata.bounds}")  
    # 方式 2：使用 get_raster_info 获取详细信息（包含统计）
    info = geoai.get_raster_info(raster_path)

    print(f"\n详细信息:")
    print(f"分辨率: {info['resolution']}")
    print(f"波段统计: {info['band_stats']}")

    # 方式 3：使用 print_raster_info 打印并可视化
    geoai.print_raster_info(raster_path, show_preview=True)  




def get_vector_metadata_raw(vector_path: str) -> dict:
    """
    直接使用 geopandas 提取矢量元数据
    
    参数:
        vector_path: 矢量文件路径
    
    返回:
        包含元数据的字典
    """
    # 判断文件格式
    if vector_path.endswith(".parquet"):
        gdf = gpd.read_parquet(vector_path)
        driver = "PARQUET"
    else:
        gdf = gpd.read_file(vector_path)
        driver = os.path.splitext(vector_path)[1][1:].upper()
    
    # 基本元数据
    metadata = {
        "file_path": vector_path,
        "driver": driver,
        "feature_count": len(gdf),
        "crs": str(gdf.crs),
        "geometry_type": str(gdf.geom_type.value_counts().to_dict()),
        "attribute_count": len(gdf.columns) - 1,
        "attribute_names": list(gdf.columns[gdf.columns != "geometry"]),
        "bounds": gdf.total_bounds.tolist(),
    }
    
    # 数值属性统计
    numeric_columns = gdf.select_dtypes(include=["number"]).columns
    attribute_stats = {}
    for col in numeric_columns:
        if col != "geometry":
            attribute_stats[col] = {
                "min": gdf[col].min(),
                "max": gdf[col].max(),
                "mean": gdf[col].mean(),
                "std": gdf[col].std(),
                "null_count": gdf[col].isna().sum(),
            }
    
    metadata["attribute_stats"] = attribute_stats

    print(f"驱动: {metadata['driver']}")
    print(f"要素数量: {metadata['feature_count']}")
    print(f"坐标系统: {metadata['crs']}")
    print(f"几何类型: {metadata['geometry_type']}")
    print(f"属性字段: {metadata['attribute_names']}")
    print(f"地理边界: {metadata['bounds']}")

    # 如果有数值属性，打印统计
    if metadata["attribute_stats"]:
        print("\n属性统计:")
        for attr, stats in metadata["attribute_stats"].items():
            print(f"  {attr}: min={stats['min']}, max={stats['max']}, mean={stats['mean']:.2f}")

    return metadata


def get_vector_metadata_geoai(vector_path: str) -> dict:
    """
    使用 geoai 提取矢量元数据
    
    参数:
        vector_path: 矢量文件路径
    
    返回:
        包含元数据的字典
    """
    # 方式 1：使用 get_vector_info 获取元数据
    info = geoai.get_vector_info(vector_path)

    print(f"驱动: {info['driver']}")
    print(f"要素数量: {info['feature_count']}")
    print(f"坐标系统: {info['crs']}")
    print(f"几何类型: {info['geometry_type']}")
    print(f"属性字段: {info['attribute_names']}")
    print(f"地理边界: {info['bounds']}")

    # 如果有数值属性，打印统计
    if info["attribute_stats"]:
        print("\n属性统计:")
        for attr, stats in info["attribute_stats"].items():
            print(f"  {attr}: min={stats['min']}, max={stats['max']}, mean={stats['mean']:.2f}")

    # 方式 2：使用 print_vector_info 打印并可视化
    geoai.print_vector_info(vector_path, show_preview=True)

    # 方式 3：使用 analyze_vector_attributes 分析特定属性
    analysis = geoai.analyze_vector_attributes(
        vector_path=vector_path,
        attribute_name="confidence"
    )
    print(f"\n高度属性分析:")
    print(f"  类型: {analysis['type']}")
    print(f"  平均值: {analysis['mean']:.2f}")
    print(f"  标准差: {analysis['std']:.2f}")    



def main():
    """主函数，支持选择下载内容"""
    # 显示欢迎信息和菜单
    print("=" * 60)
    print("GeoAI 元数据提取")
    print("=" * 60)
    print("\n请选择处理方式:")
    print("1. 源码及调用提取栅格数据元数据")
    print("2. geoai提取栅格元数据")
    print("3. 源码调用提取矢量元数据")
    print("4. geoai调用提取矢量元数据")
    print("5. 退出")
    print("-" * 60)

    raster_url=r"E:\testdata\image\GF7\GF7_subset-BWDPAN.tiff"
    raster_url1=r"E:\testdata\image\zg\ZY3_01a_synbavp_010147_20141003_113211_0008_SASMAC_CHN_sec_rel_001_1410107143.tif"
    dem_url=r"E:\testdata\dem\ASTGTMV003_N30E110_dem.tif"
    shp_url=r"E:\data\topic\river\river_line.shp"
    geojson_url=r"E:\data\geoai下载\overture_buildings\xian_place.geojson"

    # 获取用户选择
    choice = input("请输入选项 (1-5): ").strip()
    if choice == "1":
        get_raster_metadata_raw(dem_url)
    elif choice == "2":
        get_metadata_geoai(dem_url)
    elif choice == "3":
        get_vector_metadata_raw(shp_url)
    elif choice == "4":
        get_vector_metadata_geoai(geojson_url)
    elif choice == "5":
        print("谢谢使用！")
        sys.exit(0)
    else:
        print("无效选项，请输入 1-5 之间的数字")




if __name__ == "__main__":
    main()























