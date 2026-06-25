# GeoAI 遥感图像分类项目使用手册

本手册旨在指导用户如何在该项目中使用本地数据集进行模型加载、选择、训练、影像分类推理，以及如何将分类影像在地图上进行展示。

---

## 目录
1. [环境准备](#1-环境准备)
2. [本地数据集加载](#2-本地数据集加载)
3. [模型选择](#3-模型选择)
4. [模型训练](#4-模型训练)
5. [模型使用（影像分类）](#5-模型使用影像分类)
6. [影像加载与地图展示方案](#6-影像加载与地图展示方案)

---

## 1. 环境准备

项目使用 Anaconda/Miniconda 进行环境管理。项目根目录下已提供了 `environment.yml` 配置文件，包含运行所需的深度学习（PyTorch）、遥感/地理信息处理（Rasterio, Geopandas）以及 Web 服务（FastAPI, Gradio）等依赖库。

### 激活或创建环境
在终端中执行以下命令：
```bash
# 基于 environment.yml 创建 conda 环境
conda env create -f environment.yml

# 激活环境
conda activate geoai
```

---

## 2. 本地数据集加载

项目默认使用的是 **EuroSAT** 遥感图像分类数据集（包含 10 个地物类别：农田、森林、草本植被、公路、工业区、牧场、永久作物、居民区、河流、海湖）。

如果您已经将数据集手动下载到了本地，不需要运行 `python scripts/download_dataset.py`，只需按照以下步骤配置和组织目录。

### 2.1 数据集目录结构
请将下载好的数据集解压，并严格按照以下结构存放在项目根目录下的 `data/processed/EuroSAT/` 目录中：

```text
data/processed/EuroSAT/
├── train/
│   ├── AnnualCrop/          # 存放该类别的图片（.jpg/.png/.tif）
│   ├── Forest/
│   ├── HerbaceousVegetation/
│   └── ... (共10个类别文件夹)
├── val/
│   ├── AnnualCrop/
│   ├── Forest/
│   └── ...
└── test/
    ├── AnnualCrop/
    ├── Forest/
    └── ...
```

### 2.2 修改路径配置
可以通过修改项目根目录下的 `.env` 环境变量文件来指定数据集路径：
```ini
# .env 文件
DATA_DIR=./data
```
在代码中，数据集加载器 `scripts/dataset.py` 会自动寻找 `DATA_DIR` 路径下的 `processed/EuroSAT/{train,val,test}` 子目录，并自动扫描各子目录下的图片进行加载与数据增强（如随机裁剪、翻转、颜色抖动等）。

---

## 3. 模型选择

本项目内置了三种主流的深度学习图像分类骨干网络（Backbone），您可以根据硬件配置和任务复杂度自由选择：

| 模型名称 | `.env` 配置值 | 命令行参数值 | 特点 |
| :--- | :--- | :--- | :--- |
| **ResNet-50** | `resnet50` | `resnet50` | 默认模型。经典残差网络，速度快，显存占用低，准确率高。 |
| **ResNet-101** | `resnet101` | `resnet101` | 更深的残差网络，特征提取能力更强，适合更复杂的遥感地物。 |
| **ViT Base** | `vit_base_patch16_224` | `vit_base_patch16_224` | Vision Transformer 架构。适合捕捉大范围的全局上下文纹理，需要较好显卡支持。 |

### 如何配置模型：
1. **全局默认配置**：修改 `.env` 文件中的 `MODEL_NAME` 项：
   ```ini
   MODEL_NAME=resnet50          # 可选 resnet50 | resnet101 | vit_base_patch16_224
   ```
2. **命令行动态指定**：在训练或评估脚本中直接通过 `--model` 参数覆盖默认配置。

---

## 4. 模型训练

在配置好本地数据集和选择模型后，即可开始进行训练。

### 4.1 开始训练
运行 `scripts/train.py` 脚本：
```bash
# 使用默认的 ResNet-50 在 GPU/CPU 上启动训练
python scripts/train.py

# 显式指定 ViT 模型、训练轮数(Epochs)和学习率(LR)
python scripts/train.py --model vit_base_patch16_224 --epochs 20 --lr 5e-5 --batch_size 16
```

### 4.2 常用训练参数
- `--data_root`: 数据集根目录（默认 `data/processed/EuroSAT`）
- `--epochs`: 训练总轮数（默认 `30`）
- `--batch_size`: 每批次样本数（默认 `32`）
- `--lr`: 初始学习率（默认 `1e-4`）
- `--device`: 运算设备，自动检测 `cuda` 或 `cpu`
- `--resume`: 断点续训，指定权重路径（如 `--resume checkpoints/last_model.pth`）

### 4.3 训练可视化
训练过程中会产生 TensorBoard 日志，保存在 `./logs` 目录下。您可以通过以下命令实时查看损失（Loss）和准确率（Accuracy）曲线：
```bash
tensorboard --logdir ./logs
```
然后在浏览器中打开 `http://localhost:6006` 即可。

训练完成后，最佳模型权重会自动保存在 `./checkpoints/best_model.pth`。

---

## 5. 模型使用（影像分类）

本项目提供了两种推理方式：**命令行脚本推理** 和 **Web 交互界面推理**。

### 5.1 命令行单张/批量影像推理
运行 `scripts/infer.py` 脚本，可以对本地影像进行快速地物分类：
```bash
# 自动加载最佳权重对单张影像进行分类
python scripts/infer.py --image data/processed/EuroSAT/test/Forest/Forest_1.jpg

# 显式指定模型与权重路径
python scripts/infer.py --image path/to/your/image.tif --model resnet50 --checkpoint checkpoints/best_model.pth
```
运行后终端会输出图像所属地物类别、置信度以及 Top-5 的概率预测。

### 5.2 Web 交互界面推理 (推荐)
项目提供了一个轻量级的 FastAPI 后端和前端 Web 交互界面。
1. **启动后端服务**：
   在根目录下运行启动脚本：
   ```bash
   python start.py
   # 或者执行 shell 脚本
   ./start_server.sh
   ```
   服务将运行在 `http://localhost:8000`。
2. **访问 Web 页面**：
   在浏览器中打开 `http://localhost:8000`。
3. **分类操作**：
   - **单张分类**：在“图像上传”区，拖拽或点击上传单张遥感图片（支持 JPG, PNG, TIFF 等），点击“**开始分类**”按钮，右侧将实时渲染分类地物结果、Top-5 概率分布条和推理耗时。
   - **批量分类**：在下方“批量分类”面板中，可以一次性选择多张图片（最多支持 16 张），点击“**批量分类**”，系统会并行进行分类并以卡片网格的形式直观展示每张图的结果。

---

## 6. 影像加载与地图展示方案

在实际的 GIS/遥感应用中，单纯的单张图片分类是不够的，通常需要将分类结果精准叠加在地理底图上展示。

项目环境已内置了 `rasterio`（地理栅格处理库）和 `geopandas`（空间矢量处理库）。若要实现**加载影像进行分类并使用地图展示**，可以采用以下标准的 GIS 地图可视化扩展方案：

### 6.1 前后端地图集成的核心思路
1. **后端读取空间参考（GeoTIFF）**：
   标准的遥感影像（`.tif`）通常带有投影信息（如 Web Mercator 或 WGS84）。利用 `rasterio`，后端可以获取影像的**经纬度范围（Bounding Box）**：
   ```python
   import rasterio
   
   with rasterio.open("your_image.tif") as src:
       # 获取影像边界经纬度 (左, 下, 右, 上)
       bounds = src.bounds  
       # 获取投影
       crs = src.crs  
   ```
2. **分类结果栅格化/矢量化**：
   - **方案 A（像素级分割分类/图层叠加）**：如果影像已经过像素级分类（例如每个像素属于某地物），可利用 `matplotlib` 或 `PIL` 将分类后的类别矩阵（1-10的ID）映射为对应的彩色调色板，输出为一张透明背景的 `.png` 图像。
   - **方案 B（块级分类/矢量化）**：如果是整张影像分类，或者将大图切块（Tile）进行分类，可以为每个网格创建多边形边界，并利用 `geopandas` 导出为标准地理格式（`GeoJSON`）。
3. **前端地图底图渲染（Leaflet.js / OpenLayers）**：
   在 Web 端集成开源地图库（如 **Leaflet** 或 **Mapbox GL JS**）。

### 6.2 极简的前端 Leaflet 展示实现示例
在 `frontend/index.html` 中引入 Leaflet CSS 和 JS：
```html
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<div id="map" style="height: 500px; width: 100%;"></div>
```

在 `frontend/app.js` 中添加地图初始化与影像叠加逻辑：
```javascript
// 1. 初始化地图，定位到遥感影像大致区域
const map = L.map('map').setView([37.8, -96], 4); 

// 2. 添加无偏卫星底图或 OpenStreetMap 底图
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 19,
  attribution: '© OpenStreetMap contributors'
}).addTo(map);

// 3. 将后端返回的分类透明 PNG 结果，根据经纬度边界精准覆盖到地图上
// 假设后端返回的 bounding box 为: [[min_lat, min_lon], [max_lat, max_lon]]
const imageUrl = 'http://localhost:8000/static/outputs/classified_overlay.png';
const imageBounds = [[37.7, -122.5], [37.85, -122.35]]; // 后端 rasterio 提取的 bounds

// 4. 将分类图层覆盖在地图上，并设置透明度以透出底图
const classifiedLayer = L.imageOverlay(imageUrl, imageBounds, {
  opacity: 0.6,
  interactive: true
}).addTo(map);

// 5. 点击图层时弹出 Popup 气泡展示地物分类结果
classifiedLayer.bindPopup("<b>地物类别: 森林 (Forest)</b><br>置信度: 98.4%");
map.fitBounds(imageBounds); // 自动缩放地图到影像范围
```

通过这一套方案，用户即可在前端交互地图中，既看到高清晰度的遥感底图，又能在其上方半透明叠加您的 GeoAI 模型的地物分类渲染层。
