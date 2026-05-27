import shutil
import zipfile
from pathlib import Path

import geoai
import geopandas as gpd
import rasterio
import requests
from shapely.geometry import box


# =========================================================
# STAC 配置
# =========================================================

RASTER_ITEM_URL = (
    "http://localhost:8010/collections/raster-default/items/"
    "0dcc269fd72a4c4e8a3c388564122fd1"
)

VECTOR_ITEM_URL = (
    "http://localhost:8010/collections/vector-default/items/"
    "90dbe2fc21b84d83891b87fd61ed24e6"
)

BASE_DIR = Path("./data")

RAW_RASTER_DIR = BASE_DIR / "raw/raster"
RAW_VECTOR_DIR = BASE_DIR / "raw/vector"

ALIGN_RASTER_DIR = BASE_DIR / "aligned/raster"
ALIGN_VECTOR_DIR = BASE_DIR / "aligned/vector"

OUTPUT_DIR = BASE_DIR / "output"


# =========================================================
# 初始化目录
# =========================================================

def init_dirs():

    dirs = [
        RAW_RASTER_DIR,
        RAW_VECTOR_DIR,
        ALIGN_RASTER_DIR,
        ALIGN_VECTOR_DIR,
        OUTPUT_DIR,
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    print("目录初始化完成")


# =========================================================
# 获取 STAC Asset
# =========================================================

def get_asset_href(item_url):

    r = requests.get(item_url)
    r.raise_for_status()

    item = r.json()

    assets = item.get("assets", {})

    for _, asset in assets.items():

        href = asset.get("href")

        if href:
            return href

    raise Exception("未找到资产地址")


# =========================================================
# 下载文件
# =========================================================

def download_file(url, save_path):

    if save_path.exists():

        print(f"文件已存在: {save_path}")
        return

    print(f"下载: {url}")

    r = requests.get(url, stream=True)
    r.raise_for_status()

    with open(save_path, "wb") as f:

        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"保存: {save_path}")


# =========================================================
# 解压 shp zip
# =========================================================

def unzip_vector(zip_path, extract_dir):

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    shp_files = list(extract_dir.glob("*.shp"))

    if not shp_files:
        raise Exception("未找到 shp 文件")

    return shp_files[0]


# =========================================================
# 矢量与影像对齐
# =========================================================

def align_vector_to_raster(raster_path, shp_path):

    print("开始对齐矢量与影像")

    with rasterio.open(raster_path) as src:

        raster_crs = src.crs
        raster_bounds = src.bounds

        raster_geom = box(*raster_bounds)

        print(f"影像 CRS: {raster_crs}")

    gdf = gpd.read_file(shp_path)

    print(f"矢量 CRS: {gdf.crs}")

    # 坐标统一
    if gdf.crs != raster_crs:

        print("执行坐标转换")

        gdf = gdf.to_crs(raster_crs)

    # 修复 geometry
    gdf = gdf[gdf.is_valid]

    # 裁剪
    clip_gdf = gdf.clip(raster_geom)

    print(f"裁剪后数量: {len(clip_gdf)}")

    aligned_shp = ALIGN_VECTOR_DIR / "buildings_clip.geojson"

    clip_gdf.to_file(
        aligned_shp,
        driver="GeoJSON"
    )

    print(f"保存裁剪结果: {aligned_shp}")

    return aligned_shp


# =========================================================
# 准备影像
# =========================================================

def prepare_raster(raster_path):

    out_raster = ALIGN_RASTER_DIR / "image_align.tif"

    if not out_raster.exists():

        shutil.copy(raster_path, out_raster)

    return out_raster


# =========================================================
# 使用 geoai-py 生成训练样本
# =========================================================

def generate_training_data(
        raster_path,
        vector_path
):

    print("=" * 60)
    print("开始生成训练样本")
    print("=" * 60)

    geoai.export_geotiff_tiles(

        in_raster=str(raster_path),

        out_folder=str(OUTPUT_DIR),

        in_class_data=str(vector_path),

        # 切片大小
        tile_size=512,

        # 滑窗步长
        stride=256,

        # 只保留有目标的切片
        skip_empty_tiles=True,

        # polygon buffer
        buffer_radius=0,

        # 包含接触边界像素
        all_touched=True,

        # 创建 overview
        create_overview=True,

        # 数据增强
        apply_augmentation=True,

        # 每个 tile 增强数量
        augmentation_count=2,

        quiet=False,
    )

    print("训练样本生成完成")


# =========================================================
# 主函数
# =========================================================

def main():

    print("=" * 60)
    print("GeoAI 样本生产")
    print("=" * 60)

    # 初始化
    init_dirs()

    # 获取资产地址
    raster_href = get_asset_href(RASTER_ITEM_URL)
    vector_href = get_asset_href(VECTOR_ITEM_URL)

    # 下载
    raster_path = RAW_RASTER_DIR / "image.tif"
    vector_zip = RAW_VECTOR_DIR / "buildings.zip"

    download_file(raster_href, raster_path)
    download_file(vector_href, vector_zip)

    # 解压
    vector_extract_dir = RAW_VECTOR_DIR / "buildings"

    shp_path = unzip_vector(
        vector_zip,
        vector_extract_dir
    )

    # 对齐
    aligned_vector = align_vector_to_raster(
        raster_path,
        shp_path
    )

    aligned_raster = prepare_raster(
        raster_path
    )

    # 生成训练数据
    generate_training_data(
        aligned_raster,
        aligned_vector
    )

    print("\n")
    print("=" * 60)
    print("全部完成")
    print("=" * 60)

    print(f"输出目录:\n{OUTPUT_DIR}")


# =========================================================
# 入口
# =========================================================

if __name__ == "__main__":

    main()