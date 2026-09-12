# 西安与 Sentinel-2 的 U-Net 批量推理

六个脚本分别使用经典 U-Net、ResUNet、Attention U-Net、U-Net++、TransUNet、Swin-UNet 对 `Unet/data` 的全部 GeoTIFF 进行像素级建筑/植被提取；脚本不包含训练。

将西安高分影像、Sentinel-2 影像或其子目录直接放在 `Unet/data`。程序递归读取 `.tif`、`.tiff`，并跳过 `mask`、`label` 目录。

```text
Unet/data/xian/xian_rgb.tif
Unet/data/sentinel2/s2_rgb.tif

Unet/out/resunet/building/xian/xian_rgb_mask.tif
Unet/out/resunet/building/xian/xian_rgb_mask_probability.tif
Unet/out/resunet/vegetation/sentinel2/s2_rgb_mask.tif
```

输出掩膜为 `0=背景、1=目标`，同时产生 0–1 概率 GeoTIFF；两者继承输入影像的坐标系、空间范围和尺寸。模型位置为 `models/Unet/<模型>_building.pth` 与 `models/Unet/<模型>_vegetation.pth`。模型不存在时，仅会在指定对应下载地址后下载，避免不匹配权重。

```powershell
# 批量建筑、植被提取，读取 Unet/data 并输出到 Unet/out/resunet
python Unet/unet_example.py
python Unet/resunet_example.py

# 只提取建筑
python Unet/attention_unet_example.py --task building

# 本地模型不存在时下载，然后推理
python Unet/swin_unet_example.py --building-weights-url https://your-server/swin_unet_building.pth --vegetation-weights-url https://your-server/swin_unet_vegetation.pth
```

其余入口为 `unetpp_example.py`、`transunet_example.py`。Sentinel-2 使用的波段由 checkpoint 中的 `bands` 字段确定。依赖：`torch`、`rasterio`、`numpy`。
