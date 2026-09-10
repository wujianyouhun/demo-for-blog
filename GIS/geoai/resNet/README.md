# 🛰️ GeoAI 遥感图像分类系统

基于深度学习的遥感卫星图像地物分类项目，使用 **EuroSAT** 数据集，支持 ResNet50 / ResNet101 / ViT-Base 骨干网络，提供完整的训练、评估、推理和前后端服务。

## ResNet 推理示例

仓库提供了一个独立的 ResNet 推理示例：`scripts/resnet_inference_example.py`。
它会优先加载 `models/Classification/checkpoints/best_model.pth`；如果该文件不存在，则使用 torchvision 的 ImageNet 预训练 ResNet50。

```bash
pip install torch torchvision pillow

# 单张图片
python scripts/resnet_inference_example.py --image path/to/image.jpg

# 批量推理，并将结果保存为 JSON
python scripts/resnet_inference_example.py --dir path/to/images --topk 3 --output results.json
```

### 可以使用什么数据？

- **直接体验预训练模型**：使用普通 RGB 图片（JPG、JPEG、PNG、BMP、TIFF）即可。ImageNet 模型适合识别日常物体，类别名称是 ImageNet 的 1000 个类别。
- **遥感分类**：推荐使用本项目的 **EuroSAT** 数据集。它包含 10 类 Sentinel-2 遥感影像：农田、森林、草本植被、公路、工业区、牧场、永久作物、居民区、河流、海湖。使用本项目训练得到的 checkpoint 时，输入应尽量与训练数据的裁剪大小和波段形式一致。
- **自定义数据**：可以使用自己的 GeoTIFF/JPG/PNG 图像，但必须先用相同类别定义训练或微调 ResNet；未经训练的 ImageNet 模型不会自动理解自定义的遥感类别。

注意：ImageNet 预训练 ResNet 通常只接受 3 通道 RGB 输入。如果是 Sentinel-2 的多光谱影像，应先选择 RGB 波段或修改模型第一层和预处理流程。

## Landsat 8 ZIP 到分类格网

`landsat8_pipeline.py` 将多个已经下载完成的 Landsat 8 Collection 2 Level-2 ZIP 文件按以下顺序处理：**解压 → 同波段镶嵌 → 裁剪 → 云/云影掩膜 → 反射率缩放 → B4/B3/B2 RGB 合成 → 滑窗切片 → ResNet 推理 → 空间分类格网**。

先安装依赖：

```bash
pip install -r resNet/requirements_landsat8.txt
```

将所有 ZIP 放入同一个目录后运行。默认研究区是你给出的 `97.38, 36.48, 103.77, 39.73`（经度、纬度，WGS84）：

```bash
python resNet/landsat8_pipeline.py --input-zips D:/landsat8_zips --output-dir D:/landsat8_result --tile-size 64 --batch-size 32
```

输出结果：

- `02_mosaic_clipped/`：先镶嵌、后裁剪的 B2/B3/B4 和 `QA_PIXEL`。
- `03_rgb_reflectance.tif`：应用云/云影掩膜后的 B4/B3/B2 三波段地表反射率。
- `04_inference/classification_grid.tif`：每个像元代表一个切片的预测类别；`-1` 表示云或无效像元过多而跳过。
- `04_inference/confidence_grid.tif`：每个分类格网的最大类别概率。
- `04_inference/tile_predictions.json`：每个格网的类别、置信度和行列号。

`64 × 64` 的 Landsat 30 m 像元约为 `1.92 km × 1.92 km`。该研究区很大，建议先用较小范围试运行；正式处理时优先使用 CUDA，并根据显存调整 `--batch-size`。

---

## 📁 项目结构

```
geoai/
├── environment.yml          # conda 环境配置
├── .env                     # 项目参数配置
├── README.md
│
├── data/
│   ├── raw/                 # 原始下载数据
│   ├── processed/
│   │   └── EuroSAT/
│   │       ├── train/       # 训练集 (70%)
│   │       ├── val/         # 验证集 (15%)
│   │       └── test/        # 测试集 (15%)
│   └── samples/             # 样例图像
│
├── scripts/
│   ├── download_dataset.py  # 数据集下载 & 拆分
│   ├── dataset.py           # Dataset / DataLoader 模块
│   ├── train.py             # 模型训练脚本
│   ├── evaluate.py          # 模型评估脚本
│   └── infer.py             # 命令行推理脚本
│
├── backend/
│   ├── main.py              # FastAPI 后端服务
│   └── predictor.py         # 推理引擎（单例）
│
├── frontend/
│   ├── index.html           # Web 前端主页
│   ├── style.css            # 样式
│   └── app.js               # 前端交互逻辑
│
├── checkpoints/             # 模型权重保存
│   ├── best_model.pth       # 最优模型
│   └── last_model.pth       # 最新模型
│
├── logs/                    # TensorBoard 日志 & 评估报告
└── notebooks/               # Jupyter 分析笔记本
```

---

## 🌍 数据集：EuroSAT

| 类别 | 中文名 | 样本数 |
|------|--------|--------|
| AnnualCrop | 农田（一年生作物） | 3000 |
| Forest | 森林 | 3000 |
| HerbaceousVegetation | 草本植被 | 3000 |
| Highway | 公路 | 2500 |
| Industrial | 工业区 | 2500 |
| Pasture | 牧场 | 2000 |
| PermanentCrop | 永久作物 | 2500 |
| Residential | 居民区 | 3000 |
| River | 河流 | 2500 |
| SeaLake | 海湖 | 3000 |

- **总计**: 27,000 张 64×64 Sentinel-2 卫星图像
- **来源**: [EuroSAT Dataset](https://github.com/phelber/EuroSAT)

---

## ⚙️ 环境安装

```bash
# 方式 1：更新已有 geoai 环境（推荐）
conda activate geoai
pip install torch==2.2.0 torchvision==0.17.0 timm==0.9.16
pip install fastapi==0.110.0 uvicorn[standard]==0.29.0 python-multipart==0.0.9
pip install rasterio geopandas albumentations tensorboard rich python-dotenv gradio

# 方式 2：从 environment.yml 创建新环境
conda env create -f environment.yml
conda activate geoai
```

---

## 🚀 快速开始

### 第一步：下载数据集

```bash
conda activate geoai
cd F:/geoai
python scripts/download_dataset.py
```

### 第二步：训练模型

```bash
# 默认：ResNet50，30 epochs，GPU
python scripts/train.py

# 使用 ViT
python scripts/train.py --model vit_base_patch16_224 --epochs 20 --lr 5e-5

# CPU 训练（较慢）
python scripts/train.py --device cpu --batch_size 16

# 断点续训
python scripts/train.py --resume checkpoints/last_model.pth
```

### 第三步：评估模型

```bash
python scripts/evaluate.py
# 输出: logs/eval_results.json, logs/confusion_matrix.png
```

### 第四步：启动后端 API

```bash
cd F:/geoai
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 第五步：打开前端界面

直接用浏览器打开 `frontend/index.html`，或访问：
```
http://localhost:8000
```

---

## 🔌 API 接口说明

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 服务健康检查 |
| GET | `/classes` | 获取类别列表 |
| POST | `/predict` | 上传图像分类（multipart） |
| POST | `/predict/base64` | Base64 图像分类 |
| POST | `/predict/batch` | 批量图像分类（≤16张） |
| GET | `/docs` | Swagger 接口文档 |

### 示例请求

```python
import requests

# 单张预测
with open("image.jpg", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/predict",
        files={"file": ("image.jpg", f, "image/jpeg")}
    )
print(resp.json())
```

```json
{
  "class_name": "Forest",
  "class_name_zh": "森林",
  "class_id": 1,
  "confidence": 0.9831,
  "description": "自然或人工林地，树木覆盖密集",
  "top5": [
    {"class": "Forest", "class_zh": "森林", "prob": 0.9831},
    {"class": "HerbaceousVegetation", "class_zh": "草本植被", "prob": 0.0102},
    ...
  ],
  "infer_time_ms": 12.4
}
```

---

## 🖥️ 前端功能

- 🖼️ **拖拽上传** / 点击选择图像（JPG / PNG / TIFF）
- 🔍 **单张分类** — 显示 Top-5 概率条形图
- 📦 **批量分类** — 最多 16 张同时处理
- 🟢 **实时状态** — 后端服务连接状态指示
- 🗺️ **类别说明** — 展示所有支持的地物类别

---

## 📊 模型性能参考

| 模型 | 验证集准确率 | 参数量 | 推理时间(CPU) |
|------|------------|--------|--------------|
| ResNet50 | ~97-98% | 25M | ~50ms |
| ResNet101 | ~98% | 45M | ~80ms |
| ViT-Base | ~98-99% | 86M | ~120ms |

---

## 🔧 参数调整

编辑 `.env` 文件修改配置：

```env
MODEL_NAME=resnet50          # 选择模型
EPOCHS=30                    # 训练轮数
LR=0.0001                    # 学习率
BATCH_SIZE=32                # 批大小（GPU内存不足时减小）
DEVICE=cuda                  # cuda / cpu
API_PORT=8000                # API 服务端口
```

---

## 📈 TensorBoard 可视化

```bash
tensorboard --logdir logs/
# 访问 http://localhost:6006
```

---

## 🛠️ 常见问题

**Q: CUDA 内存不足？**
```bash
python scripts/train.py --batch_size 16 --num_workers 2
```

**Q: 数据集下载失败？**
- 手动下载: https://zenodo.org/records/7711810
- 解压后放入 `data/raw/EuroSAT_raw/`

**Q: 模型文件不存在？**
- 必须先运行 `python scripts/train.py` 完成训练
- 或使用预训练权重替换 `checkpoints/best_model.pth`

---

## 📦 技术栈

| 模块 | 技术 |
|------|------|
| 深度学习框架 | PyTorch 2.2 + torchvision + timm |
| 数据增强 | albumentations + torchvision.transforms |
| 后端 API | FastAPI + Uvicorn |
| 前端 | 原生 HTML5 + CSS3 + JavaScript (ES2022) |
| 可视化 | TensorBoard + Matplotlib |
| 地理处理 | rasterio + geopandas |

---

*GeoAI 遥感图像分类系统 v1.0.0*
