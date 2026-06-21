# 环境配置指南

## 系统要求

- Python >= 3.10
- Node.js >= 18
- CUDA 11.8+ (可选, GPU 加速)

## 创建 Conda 环境

```bash
conda create -n geoai-demo python=3.10 -y
conda activate geoai-demo
```

## 安装 Python 依赖

```bash
pip install -r requirements.txt
```

核心依赖:
- `fastapi` + `uvicorn` - Web 后端
- `torch` + `torchvision` - 深度学习框架
- `segmentation-models-pytorch` - DeepLabV3+ 模型
- `rasterio` - 遥感影像读写
- `albumentations` - 数据增强
- `shapely` - 几何运算
- `numpy`, `Pillow` - 基础依赖

## 安装前端依赖

```bash
cd frontend
npm install
```

前端依赖:
- `vue` 3.4+ - 前端框架
- `element-plus` 2.9+ - UI 组件库
- `leaflet` 1.9+ - 地图引擎
- `axios` - HTTP 客户端
- `vite` 5.4+ - 构建工具

## 验证安装

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import fastapi; print(f'FastAPI {fastapi.__version__}')"
node -v
npm -v
```

## 快速启动

```bash
python start.py
```

或手动启动:

```bash
# 终端 1: 后端
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# 终端 2: 前端
cd frontend && npm run dev
```
