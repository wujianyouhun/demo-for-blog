# GeoAI Change Detection - 遥感建筑变化检测示例

这是一个以 `geoai` 为主线的双时相遥感变化检测教学项目，覆盖数据下载、样本制作、模型训练、变化检测和结果展示。默认推理引擎使用 GeoAI 自带的 ChangeStar 建筑变化检测模型，因此下载完双时相影像后可以直接检测，不必先训练模型。

## 功能

- **GeoAI 数据下载**：使用 `geoai.pc_stac_search` 搜索 Planetary Computer Sentinel-2 L2A 数据，并按 bbox 裁剪下载双时相影像。
- **样本制作**：支持模拟样本、GeoAI 弱标签样本、真实变化矢量标签样本。
- **变化检测**：默认使用 `geoai.changestar_detect` 输出变化掩膜、建筑语义图和变化矢量。
- **自训练实验**：保留 Siamese U-Net / BiT 训练与推理流程，适合学习监督变化检测。
- **前端展示**：Vue + OpenLayers 展示双时相影像、卷帘对比、变化图和矢量结果。

## 快速流程

```bash
conda activate geoai
python -m pip install -r requirements.txt
```

本项目按 `geoai-py==0.40.0` 测试；如果需要新建隔离环境，也可以使用 `environment.yml`。

下载真实双时相数据：

```bash
python scripts/download_data.py --region beijing --date-a 2022-06-01 --date-b 2023-06-01
```

使用 GeoAI ChangeStar 直接检测：

```bash
python scripts/predict.py ^
  --engine geoai ^
  --image-a data/raw/time_a/xxx_A.tif ^
  --image-b data/raw/time_b/xxx_B_aligned.tif ^
  --visualize
```

制作教学样本并训练自定义模型：

```bash
python scripts/generate_sample.py --mode synthetic --num-samples 100
python scripts/train.py --model siamese_unet --epochs 30 --batch-size 4
```

启动 Web：

```bash
python start.py
```

或分别启动：

```bash
python -m uvicorn backend.main:app --reload --port 8000
cd frontend
npm run dev
```

访问 http://localhost:5173。

## 项目结构

```text
change-detection/
├── cdd/                    # 核心库：下载、样本、训练、GeoAI 推理封装
├── backend/                # FastAPI 后端
├── frontend/               # Vue 3 + OpenLayers 前端
├── scripts/                # CLI：下载、制样、训练、预测
├── docs/                   # 学习与 API 文档
├── data/                   # raw/samples/models/output 数据目录
├── config.py               # 全局配置
└── environment.yml         # Conda 环境
```

## 关键命令

```bash
# 查看 GeoAI 默认变化检测模型
python -c "import geoai; print(geoai.list_changestar_models())"

# GeoAI 弱标签制样：先用 ChangeStar 生成变化标签，再切片训练
python scripts/generate_sample.py --mode weak-label --image-a data/raw/time_a/xxx_A.tif --image-b data/raw/time_b/xxx_B_aligned.tif

# 真实矢量标签制样
python scripts/generate_sample.py --mode vector-label --image-a data/raw/time_a/xxx_A.tif --image-b data/raw/time_b/xxx_B_aligned.tif --vector-label labels/change.geojson

# 自训练模型推理
python scripts/predict.py --engine cdd --image-a data/raw/time_a/xxx_A.tif --image-b data/raw/time_b/xxx_B_aligned.tif --model data/models/best_model.pth --model-name siamese_unet
```

## 文档

- [快速上手](docs/quickstart.md)
- [GeoAI 变化检测学习指南](docs/geoai-change-detection-guide.md)
- [API 文档](docs/api.md)
- [变化检测算法](docs/change-detection.md)
