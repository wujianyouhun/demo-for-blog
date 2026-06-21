# API 文档

启动后端后访问 http://localhost:8000/docs 查看交互式 API 文档。

## 端点总览

### 系统

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 系统信息 |
| GET | `/api/health` | 健康检查 |
| GET | `/api/config` | 当前配置 |
| GET | `/api/files` | 数据文件列表 |

### 数据管理

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/data/download-pair` | 下载双时相影像对 |
| GET | `/api/data/download-pair/{id}` | 查询下载状态 |
| GET | `/api/data/pairs` | 已有影像对列表 |
| GET | `/api/data/regions` | 预定义区域 |

### 变化检测

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/detect/run` | 执行变化检测 |
| POST | `/api/detect/train` | 训练模型 |
| GET | `/api/detect/status/{id}` | 查询任务状态 |
| GET | `/api/detect/models` | 模型列表 |
| GET | `/api/detect/results` | 检测成果列表 |

### 对比分析

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/compare/visualize` | 生成对比可视化 |
| POST | `/api/compare/overlay` | 生成变化叠加图 |
| POST | `/api/compare/stats` | 变化统计信息 |
| GET | `/api/compare/preview/{name}` | 预览矢量变化 |

## 请求示例

### 下载影像对

```json
POST /api/data/download-pair
{
    "region": "beijing",
    "date_a": "2022-06-01",
    "date_b": "2023-06-01",
    "max_cloud_cover": 20.0
}
```

### 执行变化检测

```json
POST /api/detect/run
{
    "image_a": "data/raw/time_a/xxx.tif",
    "image_b": "data/raw/time_b/xxx_aligned.tif",
    "model_path": "data/models/best_model.pth",
    "model_name": "siamese_unet",
    "threshold": 0.5,
    "smoothing_sigma": 1.0
}
```

### 生成对比图

```json
POST /api/compare/visualize
{
    "image_a": "data/raw/time_a/xxx.tif",
    "image_b": "data/raw/time_b/xxx_aligned.tif",
    "change_map": "data/output/xxx_change.tif",
    "mode": "heatmap",
    "opacity": 0.5,
    "change_color": "#FF0000"
}
```
