# 祁连遥感影像 U-Net 五分类推理

本目录包含一个 GeoAI 统一入口和经典 U-Net、ResUNet、Attention U-Net、U-Net++、TransUNet、Swin-UNet 六个独立推理入口。流程复用 `DATA/裁剪.tif`、`DATA/shp/polygon_rg.shp` 制作的训练样本及本项目已训练权重，读取 `DATA/Qilian.tif`，输出道路、居民地、耕地、裸地、光伏场分类结果，并自动转为 GIS 矢量面数据。

## 快速运行

```powershell
conda activate python12
python Unet/geoai_unet.py --model resunet --task both --device cpu
```

一次比较六种模型：

```powershell
python Unet/geoai_unet.py --model all --task both --device auto --min-vector-area 100
```

其他模型入口：

```powershell
python Unet/unet_example.py --device cpu
python Unet/attention_unet_example.py --device cpu
python Unet/unetpp_example.py --device cpu
python Unet/transunet_example.py --device cpu
python Unet/swin_unet_example.py --device cpu
```

权重默认读取：

```text
models/Unet/qilian_multiclass_5class_refined/<模型>/best.pth
```

五分类结果默认输出：

```text
Unet/out/qilian_multiclass_5class_refined/<模型>/classification.tif
```

分类编码为：`0=道路`、`1=居民地`、`2=耕地`、`3=裸地`、`4=光伏场`。

## 建筑与植被/耕地提取

建筑提取：

```powershell
python Unet/resunet_example.py --task building --device cpu
```

植被/耕地提取：

```powershell
python Unet/resunet_example.py --task vegetation --device cpu
```

输出为二值掩膜和概率图：

```text
building/building_mask.tif
building/building_probability.tif
vegetation/vegetation_mask.tif
vegetation/vegetation_probability.tif
```

同时会写出可直接加载到 QGIS/ArcGIS 的 GeoPackage 面图层：

```text
classification_vector.gpkg   # 全部五分类面，含 class_id、class_name
building_vector.gpkg         # 居民地 / 建筑面
vegetation_vector.gpkg       # 耕地（当前植被代理）面
```

可用 `--min-vector-area 100` 丢弃小于 100 平方米的碎面；如只需要栅格结果，则使用 `--no-vectors`。

统一入口默认写入 `Unet/out/geoai_results/<模型>/`，不会覆盖原有模型输出。使用 `--model all --output-dir Unet/out/compare` 时，会自动写入 `compare/<模型>/`。

当前样本中没有单独的自然植被类，`vegetation` 目前对应 `cropland / 耕地`。如需真正植被提取，需要补充植被样本并重新训练。
