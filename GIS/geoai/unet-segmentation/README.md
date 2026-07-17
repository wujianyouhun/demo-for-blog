# U-Net GeoAI 多类地物分割

独立、可离线运行的 U-Net 教学与工程项目。它使用同一数据划分和训练预算对比经典 U-Net、去掉跳跃连接的编码解码器、U-Net++ 和轻量 DeepLabV3+，重点展示跳跃连接对道路、建筑边缘等空间细节的恢复作用。

## 快速开始

```powershell
conda activate geoai
python -m unet_geoai.cli generate-data --profile quick
python -m unet_geoai.cli train --model unet --profile quick
python -m unet_geoai.cli compare --profile quick
python start.py --no-install
```

Web 默认地址为 `http://127.0.0.1:5188`，API 文档为 `http://127.0.0.1:8028/docs`。

## 数据模式

- `quick`：160 张 128×128 离线六类样本，3 epoch，用于全链路冒烟。
- `full`：800 张 256×256 样本，最多 25 epoch，用于正式结构对比。
- `--real-buildings`：读取相邻 `makelable/data/output` 中的四组真实影像/建筑标签，并保留第五张未标注影像用于推理。

六类为其他背景、建筑、道路、水体、植被、裸地。生成器包含纹理、阴影、旋转目标、弯曲道路、水体和类别遮挡。

## GIS 推理

```powershell
python -m unet_geoai.cli predict --input image.tif --checkpoint path/to/best.pth
```

输出包括类别 GeoTIFF、各类概率 GeoTIFF、彩色预览、GeoJSON、GPKG 和按类别打包的 Shapefile。CRS、仿射变换和栅格尺寸从源影像继承。

模型权重统一保存在 `../models/unet-segmentation/checkpoints`，可通过 `GEOAI_MODELS_DIR` 覆盖。
