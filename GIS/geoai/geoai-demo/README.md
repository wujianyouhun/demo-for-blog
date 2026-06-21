# GeoAI Demo - 土地覆盖分类系统

基于深度学习的遥感影像语义分割系统，用于土地覆盖分类与要素提取。

## 功能特性

- **数据管理** - 下载 Sentinel-2 卫星影像，支持多区域选择
- **模型训练** - DeepLabV3+ 语义分割模型训练，支持多种骨干网络
- **要素提取** - 基于训练模型的遥感影像推理与矢量化
- **要素正则化** - 几何简化、平滑、正交化等后处理
- **地图可视化** - Leaflet 交互式地图，支持图层切换与要素着色

## 技术栈

| 层次 | 技术 |
|------|------|
| 后端 | FastAPI + Uvicorn |
| 前端 | Vue 3 + Element Plus + Leaflet |
| 深度学习 | PyTorch + segmentation-models-pytorch |
| 数据处理 | rasterio + albumentations + shapely |
| 构建工具 | Vite |

## 快速开始

```bash
# 一键启动（自动检查环境、安装依赖、启动服务）
python start.py
```

启动后访问:
- 前端界面: http://localhost:5173
- API 文档: http://localhost:8000/docs

## 项目结构

```
geoai-demo/
├── backend/                # FastAPI 后端
│   └── main.py            # 应用入口
├── frontend/              # Vue 3 前端
│   ├── src/
│   │   ├── App.vue        # 主组件
│   │   ├── main.js        # 入口文件
│   │   └── components/    # 页面组件
│   │       ├── MapView.vue          # 地图组件
│   │       ├── DataPanel.vue        # 数据管理
│   │       ├── TrainPanel.vue       # 模型训练
│   │       ├── ExtractPanel.vue     # 要素提取
│   │       └── RegularizePanel.vue  # 要素正则化
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── scripts/               # CLI 脚本
│   ├── prepare_data.py    # 数据准备
│   ├── train.py           # 模型训练
│   └── predict.py         # 推理预测
├── docs/                  # 文档
│   ├── environment.md     # 环境配置
│   ├── data-download.md   # 数据下载
│   ├── training.md        # 训练指南
│   └── api.md             # API 参考
├── start.py               # 统一启动脚本
├── .gitignore
└── README.md
```

## CLI 工具

### 数据准备

```bash
python scripts/prepare_data.py --region beijing --date 2023-06-01
```

### 模型训练

```bash
python scripts/train.py --model deeplabv3p_resnet50 --epochs 10
```

### 推理预测

```bash
python scripts/predict.py --input image.tif --model model.pth --regularize
```

## 分类类别

| ID | 类别 | 颜色 |
|----|------|------|
| 0 | 背景 (background) | #808080 |
| 1 | 建筑 (building) | #e6194b |
| 2 | 道路 (road) | #ffe119 |
| 3 | 水体 (water) | #3cb44b |
| 4 | 植被 (vegetation) | #42d4f4 |
| 5 | 裸地 (barren) | #f58231 |

## 支持模型

- DeepLabV3+ ResNet-50 (~40M params)
- DeepLabV3+ ResNet-101 (~60M params)
- DeepLabV3+ Xception (~45M params)
- DeepLabV3+ MobileNetV2 (~8M params)

## 环境要求

- Python >= 3.10
- Node.js >= 18
- CUDA 11.8+ (可选，GPU 加速)

详细配置参见 [docs/environment.md](docs/environment.md)。
