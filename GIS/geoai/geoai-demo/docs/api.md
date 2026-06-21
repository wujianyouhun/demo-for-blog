# API 参考文档

基础 URL: `http://localhost:8000`

## 数据管理

### POST /api/data/download

下载遥感影像数据。

**请求体:**
```json
{
  "region": "beijing",
  "bbox": null,
  "date_start": "2023-06-01",
  "date_end": "2023-08-31",
  "cloud_cover_max": 20
}
```

**响应:**
```json
{
  "task_id": "abc123",
  "status": "started"
}
```

### GET /api/data/download/status/{task_id}

查询下载进度。

**响应:**
```json
{
  "status": "running",
  "progress": 0.65
}
```

状态: `started`, `running`, `completed`, `failed`

### GET /api/data/files

列出已下载的数据文件。

**参数:**
- `type` (可选): `image` | `label` | `all`

**响应:**
```json
{
  "files": [
    { "name": "sentinel2_北京_2023-06-01.npy", "size": "12.5MB", "date": "2023-06-01", "path": "..." }
  ]
}
```

## 模型训练

### POST /api/train/prepare-samples

准备训练样本 (切片 + 增强)。

**请求体:**
```json
{
  "model": "deeplabv3p_resnet50"
}
```

### POST /api/train/start

启动模型训练。

**请求体:**
```json
{
  "model": "deeplabv3p_resnet50",
  "epochs": 10,
  "batch_size": 4,
  "learning_rate": 0.001
}
```

### GET /api/train/status/{task_id}

查询训练进度。

**响应:**
```json
{
  "status": "running",
  "epoch": 5,
  "loss": 0.3421,
  "miou": 0.5623
}
```

### GET /api/train/models

列出可用的已训练模型。

**响应:**
```json
{
  "models": [
    { "name": "model_checkpoint.json", "path": "output/models/model_checkpoint.json" }
  ]
}
```

## 要素提取

### POST /api/extract/predict

运行语义分割推理。

**请求体:**
```json
{
  "input_image": "data/beijing/images/sentinel2.npy",
  "model_path": "output/models/model_checkpoint.json",
  "tile_size": 256,
  "overlap": 32,
  "threshold": 0.5
}
```

### GET /api/extract/status/{task_id}

查询推理进度。

**响应:**
```json
{
  "status": "completed",
  "progress": 1.0,
  "total_features": 42,
  "elapsed": 3.5,
  "output_file": "output/results/prediction_vectors.geojson",
  "class_stats": [
    { "name": "building", "count": 15, "area": 12500.0, "ratio": 0.12 }
  ],
  "geojson": { "type": "FeatureCollection", "features": [...] }
}
```

## 要素正则化

### POST /api/extract/regularize

对提取结果进行几何正则化。

**请求体:**
```json
{
  "simplify_tolerance": 1.0,
  "smooth_iterations": 2,
  "min_area": 50,
  "orthogonalize": true,
  "ortho_angle": 10
}
```

### GET /api/extract/regularize/status/{task_id}

查询正则化进度。

**响应:**
```json
{
  "status": "completed",
  "progress": 1.0,
  "stats_before": { "count": 42, "total_area": 50000, "avg_area": 1190.5, "vertices": 850 },
  "stats_after": { "count": 35, "total_area": 48000, "avg_area": 1371.4, "vertices": 280 },
  "geojson": { "type": "FeatureCollection", "features": [...] }
}
```

## 健康检查

### GET /api/health

服务健康检查。

**响应:**
```json
{
  "status": "ok",
  "version": "1.0.0"
}
```
