# GeoAI 示例项目集

本目录按一条完整的 GeoAI 数据链路组织：数据下载、数据处理、样本制作、模型训练、推理提取、正则化与质量检查。每个项目均保留 CLI，并提供独立 Web 入口；统一项目清单见 `projects.json`。

## 推荐学习顺序

1. `data-downlad` / `Planetary Computer`：获取 STAC、Sentinel-2 和矢量数据。
2. `data-process`：检查栅格、矢量的 CRS、范围、分辨率和属性。
3. `makelable` / `sam`：对齐数据、栅格化标签、切片和交互式标注。
4. `Classification` / `geoai_segmentation` / `unet-segmentation`：训练分类、基线分割和 U-Net 模型。
5. `feature-extraction` / `building2shp` / `building-extraction` / `change-detection`：执行推理和 GIS 成果生产。
6. `building-regularize` / `vector-quality-check`：处理轮廓并检查矢量质量。

## 环境与模型缓存

```powershell
conda activate geoai
$env:GEOAI_MODELS_DIR = "F:\study\demo-for-blog\GIS\geoai\models"
```

未设置 `GEOAI_MODELS_DIR` 时，各项目默认使用本目录下的 `models/`。下载数据、模型权重、训练日志和推理成果均由 `.gitignore` 排除。

## 统一运行约定

新建或整理后的启动器均支持：

```text
--host --backend-port --frontend-port --no-install --no-browser
```

每个后端至少提供 `GET /api/health`、`GET /api/config`、`GET /api/tasks/{task_id}` 和成果下载接口。默认端口及项目状态由 `projects.json` 管理，`geoai-demo` 提供可视化门户。

重点项目：`unet-segmentation` 使用离线六类数据和现有真实建筑样本，对比经典 U-Net、无跳连编码解码器、U-Net++ 与 DeepLabV3+，并输出 GeoTIFF、GeoJSON、GPKG 和 Shapefile。
