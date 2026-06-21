# ChangeDetection - 遥感影像时序变化检测平台

基于深度学习的双时相遥感影像变化检测与对比分析平台。

## 功能概览

- **数据下载**：从 Planetary Computer 下载双时相 Sentinel-2 影像对，支持自动配对
- **变化检测**：基于 Siamese U-Net / BiT 等深度学习模型的二值变化检测
- **影像对比**：支持卷帘对比、并列叠加、差异热力图等多种可视化模式
- **模型训练**：支持 LEVIR-CD / WHU-CD 等公开数据集训练
- **自动化流水线**：下载 → 预处理 → 变化检测 → 后处理 → 矢量化 一键完成

## 项目结构

```
change-detection/
├── cdd/                    # 核心变化检测库
│   ├── downloader.py       # 双时相数据下载与配对
│   ├── dataset.py          # 双时相数据集定义
│   ├── models.py           # Siamese U-Net / BiT 模型
│   ├── trainer.py          # 训练引擎
│   ├── inference.py        # 推理引擎
│   ├── metrics.py          # 变化检测评价指标
│   └── visualize.py        # 差异图与可视化工具
├── backend/                # FastAPI 后端
│   ├── main.py
│   ├── routers/            # API 路由
│   └── utils/              # 工具函数
├── frontend/               # Vue 3 + OpenLayers 前端
│   └── src/
│       └── components/     # 地图对比/变化检测/结果面板
├── scripts/                # 独立脚本
├── tests/                  # 单元测试
├── data/                   # 数据目录
├── docs/                   # 文档
├── config.py               # 全局配置
├── requirements.txt        # Python 依赖
└── environment.yml         # Conda 环境
```

## 快速开始

### 1. 环境安装

```bash
conda env create -f environment.yml
conda activate change-detection
# 或
pip install -r requirements.txt
```

### 2. 生成模拟数据（快速体验）

```bash
python scripts/generate_sample.py
```

### 3. 下载真实双时相数据

```bash
python scripts/download_data.py --region beijing --date-a 2022-06-01 --date-b 2023-06-01
```

### 4. 训练

```bash
python scripts/train.py --model siamese_unet --epochs 50 --batch-size 8
```

### 5. 推理

```bash
python scripts/predict.py \
    --image-a data/raw/time_a/xxx.tif \
    --image-b data/raw/time_b/xxx_aligned.tif \
    --model data/models/best_model.pth \
    --visualize
```

### 6. Web 服务

```bash
cd backend && uvicorn main:app --reload --port 8000
cd frontend && npm install && npm run dev
```

访问 http://localhost:5173

## 技术栈

| 层级 | 技术 |
|------|------|
| 深度学习 | PyTorch 2.x, torchvision |
| 地理空间 | rasterio, geopandas, shapely, planetary-computer |
| 后端 | FastAPI, uvicorn |
| 前端 | Vue 3, Vite, OpenLayers, Element Plus |
| 测试 | pytest, httpx |

## 文档

- [快速上手](docs/quickstart.md)
- [变化检测算法](docs/change-detection.md)
- [API 文档](docs/api.md)
