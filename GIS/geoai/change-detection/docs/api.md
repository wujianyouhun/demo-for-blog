# API 文档

启动后端后访问 http://localhost:8000/docs 查看交互式 API。

## 端点总览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| GET | `/api/config` | 配置、训练参数、GeoAI 模型列表 |
| GET | `/api/files` | data 目录文件列表 |
| POST | `/api/data/download-pair` | 使用 GeoAI/STAC 下载双时相 Sentinel-2 影像 |
| GET | `/api/data/download-pair/{id}` | 查询下载任务 |
| GET | `/api/data/pairs` | 已下载影像对 |
| GET | `/api/data/regions` | 预设区域 |
| GET | `/api/data/preview-file?path=...` | 预览 data 目录内 GeoTIFF，包括 output 结果 |
| GET | `/api/detect/samples` | 查询训练样本数量 |
| POST | `/api/detect/samples` | 制作训练样本 |
| POST | `/api/detect/train` | 训练自定义模型 |
| POST | `/api/detect/run` | 执行变化检测 |
| GET | `/api/detect/status/{id}` | 查询训练/制样/检测任务 |
| GET | `/api/detect/models` | 自训练模型和 GeoAI ChangeStar 模型列表 |
| GET | `/api/detect/results` | 检测结果列表 |
| POST | `/api/compare/visualize` | 生成对比图 |
| POST | `/api/compare/overlay` | 生成变化叠加图 |
| POST | `/api/compare/stats` | 变化统计 |

## 下载影像对

```json
POST /api/data/download-pair
{
  "region": "beijing",
  "date_a": "2022-06-01",
  "date_b": "2023-06-01",
  "max_cloud_cover": 20.0
}
```

也可以传自定义 bbox：

```json
{
  "bbox": [116.2, 39.75, 116.6, 40.05],
  "date_a": "2022-06-01",
  "date_b": "2023-06-01"
}
```

## 制作样本

模拟样本：

```json
POST /api/detect/samples
{
  "mode": "synthetic",
  "num_samples": 100,
  "tile_size": 256,
  "overwrite": true
}
```

GeoAI 弱标签样本：

```json
POST /api/detect/samples
{
  "mode": "weak-label",
  "image_a": "data/raw/time_a/xxx_A.tif",
  "image_b": "data/raw/time_b/xxx_B_aligned.tif",
  "geoai_model": "s1_s1c1_vitb",
  "threshold": 0.5,
  "tile_size": 256,
  "stride": 256
}
```

真实矢量标签样本：

```json
POST /api/detect/samples
{
  "mode": "vector-label",
  "image_a": "data/raw/time_a/xxx_A.tif",
  "image_b": "data/raw/time_b/xxx_B_aligned.tif",
  "vector_label": "labels/change.geojson",
  "tile_size": 256,
  "stride": 256
}
```

## 训练模型

```json
POST /api/detect/train
{
  "model_name": "siamese_unet",
  "epochs": 30,
  "batch_size": 4,
  "learning_rate": 0.0001
}
```

训练前会检查 `data/samples/time_a`、`time_b`、`labels` 数量是否一致。

## 执行变化检测

GeoAI 默认引擎，不需要 `model_path`：

```json
POST /api/detect/run
{
  "engine": "geoai",
  "image_a": "data/raw/time_a/xxx_A.tif",
  "image_b": "data/raw/time_b/xxx_B_aligned.tif",
  "model_name": "s1_s1c1_vitb",
  "tile_size": 1024,
  "overlap": 64,
  "threshold": 0.5
}
```

自训练引擎需要 `model_path`：

```json
POST /api/detect/run
{
  "engine": "cdd",
  "image_a": "data/raw/time_a/xxx_A.tif",
  "image_b": "data/raw/time_b/xxx_B_aligned.tif",
  "model_path": "data/models/best_model.pth",
  "model_name": "siamese_unet",
  "tile_size": 256,
  "overlap": 32,
  "threshold": 0.5
}
```

结果字段：

```json
{
  "mask": "data/output/..._change.tif",
  "vectors": "data/output/..._change.gpkg",
  "t1_semantic": "data/output/..._t1_semantic.tif",
  "t2_semantic": "data/output/..._t2_semantic.tif",
  "previews": {
    "overlay": "data/output/..._overlay.png",
    "side_by_side": "data/output/..._side_by_side.png"
  }
}
```
