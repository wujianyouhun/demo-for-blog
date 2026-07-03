# GeoAI 建筑物轮廓提取

本项目使用 GeoAI 预训练建筑物模型和可选的 SAM 模型，从本地 GeoTIFF 影像中自动提取建筑物轮廓，输出 Shapefile、GeoPackage、统计表和预览图，适合做遥感影像建筑物提取实验、GIS 数据生产演示和公众号文章配图。

## 项目结构

```text
building-extraction/
  data/                  # 放原始 tif/tiff 影像，例如 data/西安19级.tif
  models/                # 自动下载或手动放置模型权重
  outputs/               # 运行结果、瓦片缓存、矢量成果和预览图
  extract_buildings.py   # 推荐入口：GeoAI/SAM 提取、切片、合并、正则化
  building_extractor.py  # 备用入口：DeepLabV3+ 语义分割流程
  requirements.txt
```

原始影像默认放在项目本地 `data/` 目录下。运行 `extract_buildings.py` 时，如果不传 `--source-tif`，脚本会自动扫描 `data/*.tif` 和 `data/*.tiff`。如果 `data/` 中有多张影像，使用 `--source-tif data/影像名.tif` 明确指定。

## 环境安装

建议使用独立 Conda 环境：

```powershell
conda create -n geoai python=3.11 -y
conda activate geoai
pip install -r requirements.txt
```

检查依赖：

```powershell
python extract_buildings.py --check-env
```

## 快速测试

把原始 GeoTIFF 放入 `data/` 后执行：

```powershell
python extract_buildings.py --test-crop --models geoai --device cuda
```

如果只是验证流程，也可以使用 CPU：

```powershell
python extract_buildings.py --test-crop --models geoai --device cpu
```

## 全图提取

大幅影像默认采用切片处理，再合并建筑物结果：

```powershell
python extract_buildings.py --models geoai --tile-size 2048 --overlap 256 --device cuda
```

指定某一张影像：

```powershell
python extract_buildings.py --source-tif data/西安19级.tif --models geoai --tile-size 2048 --overlap 256 --device cuda
```

缺少模型时，脚本默认会下载到项目 `models/` 目录，后续运行会复用本地模型。离线运行可加：

```powershell
python extract_buildings.py --models geoai --no-download-models
```

## 输出成果

主要结果位于 `outputs/`：

- `outputs/geoai_tiled/raw.shp`：合并后的原始建筑物轮廓
- `outputs/geoai_tiled/regularized.shp`：正则化后的建筑物轮廓
- `outputs/geoai_tiled/regularized.gpkg`：GeoPackage 格式成果
- `outputs/compare/summary.csv`：不同模型或参数的统计对比
- `outputs/compare/best_regularized.shp`：脚本自动挑选的较优结果
- `outputs/preview/*.png`：公众号或报告可用的预览图

## Git 地址

项目代码地址：

https://github.com/wujianyouhun/demo-for-blog/tree/master/GIS/geoai/building-extraction

如果模型下载不下来，可以在公众号后台私信我，我可以提供模型文件或离线放置方式。
