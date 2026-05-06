import pystac
import planetary_computer
import rioxarray
import os

# 输入参数
item_url = "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/S2A_MSIL2A_20260426T015641_R117_T51KYB_20260426T064255"
output_dir = "sentinel2_data"  # 输出目录
assets_to_download = ["B04", "B08"]  # 要下载的波段：B04-红光, B08-近红外

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 加载项目元数据并签名资产
item = pystac.Item.from_file(item_url)
signed_item = planetary_computer.sign(item)

print(f"项目 ID: {item.id}")
print(f"可用资产: {list(item.assets.keys())}")
print(f"时间: {item.datetime}")

# 下载指定的资产
for asset_key in assets_to_download:
    if asset_key in signed_item.assets:
        print(f"\n正在下载 {asset_key}...")
        asset_href = signed_item.assets[asset_key].href
        
        # 打开数据
        ds = rioxarray.open_rasterio(asset_href)
        
        # 构建输出文件名
        output_path = os.path.join(output_dir, f"{item.id}_{asset_key}.tif")
        
        # 保存数据到本地文件
        ds.rio.to_raster(output_path)
        
        print(f"已保存: {output_path}")
        print(f"数据形状: {ds.shape}")
        print(f"数据类型: {ds.dtype}")
    else:
        print(f"\n警告: 资产 {asset_key} 不存在")

print("\n下载完成！")
