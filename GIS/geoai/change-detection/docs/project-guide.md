# ChangeDetection — 遥感影像时序变化检测平台

## 项目概述

ChangeDetection 是一个端到端的遥感影像双时相变化检测系统，覆盖从卫星影像下载、训练样本准备、深度学习模型训练到变化检测推理与矢量化的完整工作流。系统采用前后端分离架构，后端提供 RESTful API，前端提供可视化交互界面，核心算法库可独立调用。

**技术栈**：PyTorch 2.x + FastAPI + Vue 3 + Element Plus + OpenLayers 9

**核心能力**：

- 通过 Planetary Computer STAC API 下载 Sentinel-2 L2A 双时相影像
- Siamese U-Net / BiT Transformer 双架构变化检测模型
- 滑窗推理 + 重叠区域概率融合 + 高斯平滑 + 自适应阈值分割
- 检测结果矢量化输出 GeoPackage/GeoJSON
- 卷帘对比、并列对比、变化图叠加三种可视化模式

---

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│  前端 (Vue 3 + Element Plus + OpenLayers)               │
│  ┌──────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │ DataPanel │  │DetectionPanel│  │  ComparePanel   │   │
│  │ 数据下载  │  │ 训练 & 检测   │  │  统计 & 可视化  │   │
│  └──────────┘  └──────────────┘  └─────────────────┘   │
│                    ↕ axios HTTP                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  MapCompare (OpenLayers 卷帘 / 并列 / 变化图)     │   │
│  └──────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────┘
                            │ /api/*
┌───────────────────────────┴─────────────────────────────┐
│  后端 (FastAPI + Uvicorn)                               │
│  ┌──────────┐  ┌────────────┐  ┌───────────────────┐   │
│  │ data.py  │  │detection.py│  │   compare.py      │   │
│  │ 数据管理  │  │ 训练 & 推理 │  │   对比 & 统计     │   │
│  └──────────┘  └────────────┘  └───────────────────┘   │
└───────────────────────────┬─────────────────────────────┘
                            │ import
┌───────────────────────────┴─────────────────────────────┐
│  核心库 cdd (Change Detection Deep-learning)             │
│  ┌────────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐  │
│  │downloader  │ │ dataset  │ │  models  │ │ trainer │  │
│  │ 影像下载    │ │ 数据加载  │ │ 模型定义  │ │ 训练引擎│  │
│  ├────────────┤ ├──────────┤ ├──────────┤ ├─────────┤  │
│  │inference   │ │ metrics  │ │visualize │ │         │  │
│  │ 推理引擎    │ │ 评价指标  │ │ 可视化工具│ │         │  │
│  └────────────┘ └──────────┘ └──────────┘ └─────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 项目文件说明

### 根目录

| 文件 | 说明 |
|------|------|
| `config.py` | 全局配置中心。定义项目根路径、数据目录结构（raw/samples/models/output）、预设区域（北京/上海/深圳/迪拜/旧金山的 bbox 和 EPSG）、Sentinel-2 波段列表、模型配置（siamese_unet/bit 的编码器与解码器参数）、训练超参数（epochs/batch_size/lr/数据增强策略）、推理配置（滑窗大小/重叠/阈值）和 API 端口等 |
| `start.py` | 一键启动脚本。自动检测 Python >= 3.10 和 Node.js，探测关键依赖包是否缺失并自动 `pip install`，检测 `node_modules` 并自动 `npm install`，创建数据目录，然后以子进程方式同时启动后端（uvicorn --reload）和前端（vite dev），注册 SIGINT/SIGTERM/SIGBREAK 信号实现 Ctrl+C 优雅退出 |
| `requirements.txt` | Python 依赖清单，涵盖 PyTorch、rasterio、geopandas、pystac-client、planetary-computer、albumentations、FastAPI、uvicorn 等 |
| `environment.yml` | Conda 环境定义文件，可一键创建 conda 虚拟环境 |
| `.gitignore` | Git 忽略规则，排除 Python 字节码、GeoTIFF/GPKG/SHP 地理数据文件、模型权重（pth/onnx/safetensors）、node_modules、IDE 配置、测试缓存等 |

### cdd/ — 核心算法库

`cdd` 是 **Change Detection Deep-learning** 的缩写，封装了全部深度学习与地理数据处理逻辑。

| 文件 | 说明 |
|------|------|
| `__init__.py` | 统一导出 7 个核心类：BiTemporalDownloader、BiTemporalDataset、create_dataloaders、build_model、Trainer、ChangeDetector、ChangeMetrics、ChangeVisualizer |
| `downloader.py` | **双时相影像下载器**。通过 `pystac_client` 连接 Planetary Computer STAC API，搜索 Sentinel-2 L2A 数据，使用 `planetary_computer.sign()` 签名获取 Azure SAS 令牌，然后通过 `rasterio` 窗口裁剪读取（`from_bounds` + `transform_bounds`）仅下载 bbox 对应区域而非全幅 10980x10980 影像，大幅减少数据传输量。内置 GDAL HTTP 超时配置（`GDAL_HTTP_TIMEOUT=60`），支持按云量排序选取最优影像，自动执行双时相空间对齐（`reproject`） |
| `dataset.py` | **数据集与数据加载**。`BiTemporalDataset` 继承 `torch.utils.data.Dataset`，从三个目录（time_a/time_b/labels）读取配对的 GeoTIFF 文件，支持 albumentations 数据增强（水平翻转、垂直翻转、随机旋转90°、平移缩放旋转、亮度对比度扰动），归一化到 [0,1]，标签裁剪到 {0,1}。`create_dataloaders()` 按 85%/15% 比例切分训练/验证集，训练集启用 shuffle 和 drop_last，使用固定随机种子 42 保证可复现 |
| `models.py` | **模型定义**。实现两种变化检测网络架构：**Siamese U-Net** — 共享权重的 ResNet34/50/MobileNetV2 编码器分别提取双时相特征，解码器通过 `ConvTranspose2d` 上采样并与双路 skip connection 拼接（通道数翻倍），最终输出 2 通道 logits（变化/未变化）；**BiT (Bitemporal Image Transformer)** — ResNet50 编码器提取最深层特征后拼接，经 1x1 卷积投影到嵌入空间，通过 4 层 Transformer Block（MultiheadAttention + FFN）建模全局上下文关系，再通过 3 层 ConvTranspose2d 逐级上采样恢复原始分辨率。`build_model()` 工厂函数按名称构建模型，`load_model()` 加载预训练权重 |
| `trainer.py` | **训练引擎**。`Trainer` 类封装完整的训练循环：AdamW 优化器 + CosineAnnealing 学习率调度器，支持 CUDA 混合精度训练（`torch.amp.GradScaler`），每 epoch 计算训练/验证的 loss 和 F1 分数，基于验证 F1 的 Early Stopping（默认 patience=10），自动保存 `best_model.pth`（最优验证 F1）和 `final_model.pth`（最后一个 epoch） |
| `inference.py` | **推理引擎**（详见下文核心流程）。`ChangeDetector` 实现滑窗推理、概率融合、高斯平滑、阈值分割和矢量化的完整推理管线 |
| `metrics.py` | **评价指标**。`ChangeMetrics.compute()` 从预测和真值二值掩膜计算混淆矩阵、总体精度（OA）、精确率（Precision）、召回率（Recall）、F1 分数、IoU、Kappa 系数、变化比例等指标，并提供 `print_report()` 格式化输出 |
| `visualize.py` | **可视化工具**。`ChangeVisualizer` 提供三种对比图生成：并列对比（side_by_side，左右拼接）、变化叠加（overlay，红色半透明蒙版覆盖在底图上）、差异热力图（heatmap，蓝→青→绿→黄→红渐变色带表示差异强度）。内置 `max_size` 限制输出分辨率，自动处理多波段→RGB 转换 |

### backend/ — FastAPI 后端

| 文件 | 说明 |
|------|------|
| `main.py` | FastAPI 应用入口。注册三个路由模块（data/detection/compare），配置 CORS 中间件（允许前端 localhost:5173 跨域），挂载 `/output` 静态文件目录供前端直接访问检测结果图片，提供 `/api/health`（健康检查）、`/api/config`（获取预设配置）、`/api/files`（递归列出数据目录下所有文件）等全局端点 |
| `routers/data.py` | **数据管理 API**（前缀 `/api/data`）。`POST /download-pair` 启动后台下载任务，支持按预设区域或自定义 bbox 下载，返回 task_id 后通过 `GET /download-pair/{task_id}` 轮询进度（5 个阶段：searching → downloading_a → downloading_b → aligning → done）。`GET /pairs` 列出已有影像对，`GET /preview/{subdir}/{filename}` 将 GeoTIFF 转为 PNG 预览（2% 线性拉伸增强），响应头携带 `X-Image-Bounds` 和 `X-Image-CRS` 供前端地图定位 |
| `routers/detection.py` | **变化检测 API**（前缀 `/api/detect`）。`POST /train` 启动训练任务（启动前预检查样本目录是否为空或数量不匹配，返回 400 错误），`POST /run` 启动推理任务（加载模型 → ChangeDetector → detect_and_vectorize），`GET /status/{task_id}` 轮询状态，`GET /models` 列出可用模型权重文件，`GET /results` 列出检测结果（tif 掩膜 + gpkg 矢量） |
| `routers/compare.py` | **对比分析 API**（前缀 `/api/compare`）。`POST /visualize` 生成对比可视化图（并列/叠加/热力图），`POST /overlay` 创建变化叠加图，`POST /stats` 读取变化图统计像素信息（总像素/变化像素/变化比例/平均概率），`GET /preview/{filename}` 将矢量文件转为 GeoJSON 返回（自动投影到 EPSG:4326） |

### frontend/ — Vue 3 前端

| 文件 | 说明 |
|------|------|
| `vite.config.js` | Vite 构建配置，开发服务器监听 5173 端口，将 `/api` 和 `/output` 路径代理到后端 `localhost:8000` |
| `package.json` | 前端依赖：Vue 3.4、Element Plus 2.9、OpenLayers 9.2、Axios 1.7 |
| `src/main.js` | Vue 应用入口，注册 Element Plus 组件库和图标 |
| `src/App.vue` | **主布局**。顶部渐变导航栏显示 API 连接状态，左侧 400px 侧边栏包含三个标签页（数据下载/变化检测/对比分析），右侧主区域为 OpenLayers 地图。核心数据流：`onPairLoaded` → 构建预览 URL → 获取影像 bounds → 传给地图组件；`onDetectResult` → 构建变化图 URL → 传给地图；`onShowGeoJson` → 矢量数据传给地图叠加 |
| `src/api/client.js` | Axios 实例，baseURL 为空（走 Vite 代理），超时 300 秒 |
| `src/components/DataPanel.vue` | **数据下载面板**。表单选择预设区域或自定义 bbox，配置双时相日期和最大云量，点击"下载双时相影像"触发后台任务，通过 2 秒间隔轮询展示进度条（el-progress，striped + striped-flow 动画），按阶段（搜索→下载A→下载B→对齐→完成）更新标签文字，底部列出已有影像对并支持点击选中 |
| `src/components/DetectionPanel.vue` | **变化检测面板**。分为模型训练区（选择模型架构、epoch/batch_size/学习率）和推理检测区（选择影像对、模型权重、阈值/平滑/最小面积参数），均通过后台任务异步执行，5 秒轮询状态，训练完成自动刷新模型列表 |
| `src/components/ComparePanel.vue` | **对比分析面板**。自动获取检测结果的像素统计（变化比例/变化像素数），支持生成三种对比可视化图，列出矢量结果文件并支持点击预览 GeoJSON（emit 到地图叠加显示） |
| `src/components/MapCompare.vue` | **地图组件**。基于 OpenLayers 9 实现三种对比模式：**卷帘模式**（swipe）通过 layer 级 `prerender`/`postrender` 事件对 canvas 上下文进行矩形裁剪，拖拽手柄控制分割线位置；**并列模式**（side）时相 B 以 50% 透明度叠加在时相 A 之上；**变化图模式**（change）隐藏原始影像显示变化检测结果。支持 GeoJSON 矢量叠加（红色半透明填充 + 红色边框），自动缩放到数据范围 |
| `src/assets/main.css` | 全局样式 |

### scripts/ — 命令行工具

| 文件 | 说明 |
|------|------|
| `download_data.py` | CLI 数据下载脚本，接受 region/bbox/date 等参数，调用 BiTemporalDownloader 下载影像对 |
| `generate_sample.py` | **模拟样本生成器**。生成 100 对 256x256 合成双时相样本：时相 A 为随机 RGB 地物 + 矩形结构（模拟建筑/道路），时相 B 在 A 基础上添加 1-5 处矩形变化区域（新建/拆除），标签为精确的二值变化掩膜，添加高斯噪声模拟真实传感器差异 |
| `train.py` | CLI 训练脚本，支持 `--model`/`--epochs`/`--batch-size`/`--lr` 等参数，调用 `build_model` → `create_dataloaders` → `Trainer.fit` 完整训练流程 |
| `predict.py` | CLI 推理脚本，加载模型权重，创建 `ChangeDetector`，执行 `detect_and_vectorize`，可选 `--visualize` 生成对比图 |

### tests/ — 单元测试

| 文件 | 说明 |
|------|------|
| `test_models.py` | 测试编码器输出特征图维度、SiameseUNet/BiT 前向传播输出形状、数据集读取与增强、数据加载器创建（共 20 个测试用例） |
| `test_inference.py` | 测试 ChangeDetector 滑窗推理输出、矢量化输出、高斯平滑参数 |
| `test_api.py` | 测试 FastAPI 端点响应（health/config/pairs/regions/train 预检查） |

### data/ — 数据目录（运行时生成，不入 Git）

```
data/
├── raw/
│   ├── time_a/    # 时相 A 原始 GeoTIFF 影像
│   └── time_b/    # 时相 B 原始 GeoTIFF 影像（含 _aligned.tif 对齐版本）
├── samples/
│   ├── time_a/    # 训练样本 - 时相 A patch (256x256)
│   ├── time_b/    # 训练样本 - 时相 B patch (256x256)
│   └── labels/    # 训练样本 - 二值变化标签 (0=未变, 1=变化)
├── models/        # 训练产出的模型权重 (.pth)
└── output/        # 推理产出的变化图 (.tif) 和矢量文件 (.gpkg)
```

---

## 训练核心流程

训练流程由 `scripts/train.py` 或 `POST /api/detect/train` 触发，完整链路如下：

### 1. 数据加载 (`cdd/dataset.py`)

```
BiTemporalDataset
├── 扫描 time_a/*.tif, time_b/*.tif, labels/*.tif
├── 校验三个目录文件数量一致
└── __getitem__(idx):
    ├── rasterio 读取 image_a → float32, 归一化到 [0,1]
    ├── rasterio 读取 image_b → float32, 归一化到 [0,1]
    ├── rasterio 读取 label → int64, 裁剪到 {0,1}
    ├── albumentations 增强 (拼接 A+B 通道后联合增强):
    │   ├── HorizontalFlip (p=0.5)
    │   ├── VerticalFlip (p=0.3)
    │   ├── RandomRotate90 (p=0.5)
    │   ├── ShiftScaleRotate (p=0.3)
    │   └── RandomBrightnessContrast (p=0.3)
    └── 转为 torch.Tensor (CHW float32 + long)
```

`create_dataloaders()` 按 85:15 比例切分，训练集 `shuffle=True, drop_last=True`，验证集 `shuffle=False`。

### 2. 模型构建 (`cdd/models.py`)

**Siamese U-Net (resnet34)**:

```
输入: image_a (B,3,256,256)    输入: image_b (B,3,256,256)
        ↓                              ↓
   ┌────┴─────────────────────────┐
   │  共享编码器 ResNet34          │
   │  conv1 → [64, 128, 128]     │
   │  layer1 → [64, 64, 64]      │
   │  layer2 → [128, 32, 32]     │
   │  layer3 → [256, 16, 16]     │
   │  layer4 → [512, 8, 8]       │
   └──────────────────────────────┘
        ↓ feats_a                    ↓ feats_b
   ┌────┴────────────────────────────┐
   │  解码器 (每层拼接 siamese 双路)  │
   │  concat(fa[-1], fb[-1])        │  → [1024, 8, 8]
   │  ConvTranspose2d → [256, 16,16]│
   │  cat(skip_a, skip_b) → [512]   │
   │  Conv2d(256+512, 256) → [256]  │
   │  ConvTranspose2d → [128, 32,32]│
   │  cat(skip) → [256]             │
   │  Conv2d(128+256, 128) → [128]  │
   │  ConvTranspose2d → [64, 64,64] │
   │  cat(skip) → [128]             │
   │  Conv2d(64+128, 64) → [64]     │
   │  ConvTranspose2d → [32,128,128]│
   │  cat(skip) → [128]             │
   │  Conv2d(32+128, 32) → [32]     │
   │  ConvTranspose2d → [32,256,256]│
   │  Conv2d(32, 2) → [2, 256, 256]│
   └─────────────────────────────────┘
        ↓
   输出 logits (B, 2, 256, 256)
   通道 0 = 未变化概率, 通道 1 = 变化概率
```

总参数量约 2488 万（ResNet34 编码器 + 解码器）。

### 3. 训练循环 (`cdd/trainer.py`)

```
Trainer.fit(train_loader, val_loader, epochs, save_dir):
  for epoch in 1..epochs:
    ┌── train_epoch():
    │   for batch in train_loader:
    │     ├── 前向传播: model(image_a, image_b) → logits
    │     ├── 损失计算: CrossEntropyLoss(logits, label)
    │     ├── [混合精度] autocast + GradScaler
    │     ├── 反向传播: loss.backward()
    │     ├── 参数更新: optimizer.step()
    │     └── 计算 batch F1
    │
    ├── validate():
    │   for batch in val_loader:
    │     ├── 前向传播 (no_grad)
    │     ├── 损失计算
    │     └── 计算 batch F1
    │
    ├── CosineAnnealingLR.step()  — 学习率余弦退火
    ├── 记录 history (loss/f1/lr)
    ├── if val_f1 > best_val_f1:
    │     保存 best_model.pth, 重置 patience
    └── else: patience++ → 达到阈值则早停
```

**混合精度训练**：在 CUDA 设备上自动启用 `torch.amp.autocast("cuda")`，前向传播使用 FP16 加速，`GradScaler` 处理梯度缩放防止下溢，可减少约 40% 显存占用并提升训练速度。

**F1 评分**：以像素为单位计算 TP（真变化）、FP（伪变化）、FN（漏检），F1 = 2 * Precision * Recall / (Precision + Recall)，综合衡量检测精度与完整性。

---

## 变化检测核心流程

推理流程由 `scripts/predict.py` 或 `POST /api/detect/run` 触发：

### 1. 滑窗推理 (`cdd/inference.py → ChangeDetector.detect`)

```
输入: image_a (W×H), image_b (W×H), model

Step 1: 计算滑窗位置
  stride = tile_size - overlap = 256 - 32 = 224
  windows = {(y, x) | y ∈ [0, H-256] step 224, x ∈ [0, W-256] step 224}
  边缘位置 clamp 到 max(0, H-256) / max(0, W-256) 确保覆盖完整

Step 2: 批量推理
  for window in windows (每 batch_size=4 个):
    ├── rasterio 窗口读取: Window(x, y, 256, 256)
    ├── 归一化到 [0,1], 确保 3 通道
    ├── Stack → tensor (B, 3, 256, 256)
    ├── model(tile_a, tile_b) → logits (B, 2, 256, 256)
    └── softmax(dim=1)[:, 1] → probs (B, 256, 256)

Step 3: 概率融合
  prob_sum[y:y+256, x:x+256] += probs[i]
  count_map[y:y+256, x:x+256] += 1
  ────────────────────────────────────
  prob_avg = prob_sum / count_map
  (重叠区域取平均，消除拼接痕迹)
```

### 2. 后处理

```
Step 4: 高斯平滑
  prob_avg = gaussian_filter(prob_avg, sigma=smoothing_sigma)
  (消除噪声孤立点，使变化区域边界更平滑)

Step 5: 阈值分割
  change_map = (prob_avg > threshold).astype(uint8)
  (默认 threshold=0.5，概率 > 50% 判定为变化)

Step 6: 写入 GeoTIFF
  2 波段输出:
    Band 1: 二值变化掩膜 (0/1, float32)
    Band 2: 变化概率图 (0~1, float32)
  保留原始 CRS 和 transform，LZW 压缩
```

### 3. 矢量化 (`ChangeDetector.vectorize`)

```
Step 7: 栅格→矢量转换
  rasterio.features.shapes(label, mask=label==1, transform=transform)
  → 提取所有连通变化区域的几何轮廓

Step 8: 过滤与属性
  for each polygon:
    ├── shapely_shape(geom) → Polygon
    ├── 过滤: area < min_area_pixels (默认 30)
    └── 属性: change_type="changed", area_px=pixel面积

Step 9: 输出矢量文件
  GeoDataFrame → GeoPackage (.gpkg) 或 GeoJSON (.geojson)
  保留原始坐标参考系
```

---

## 数据下载流程

`cdd/downloader.py → BiTemporalDownloader` 实现从 Planetary Computer 获取 Sentinel-2 数据：

```
1. STAC 搜索
   pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
   client.search(
     collections=["sentinel-2-l2a"],
     bbox=[left, bottom, right, top],
     datetime="YYYY-MM-DD/YYYY-MM-DD",
     query={"eo:cloud_cover": {"lt": max_cloud}},
     max_items=10
   )
   → 按云量排序取最优

2. 签名
   planetary_computer.sign(best_item)
   → 为 Azure Blob 存储的资产添加 SAS 令牌

3. 窗口裁剪读取
   rasterio.warp.transform_bounds(EPSG:4326 → 影像 CRS)
   rasterio.windows.from_bounds(bbox_in_crs, transform)
   → 仅读取 bbox 对应的像素区域 (而非全幅 10980x10980)
   → 配置 GDAL_HTTP_TIMEOUT=60 防止无限等待

4. 波段合成
   读取 B04(R) + B03(G) + B02(B) 三个 10m 波段
   归一化: clip(data / 3000 * 255, 0, 255) → uint8
   写入 3 波段 GeoTIFF (LZW 压缩)

5. 空间对齐
   rasterio.warp.reproject(时相B → 时相A 的 CRS/transform/shape)
   → 确保双时相影像像素对齐，消除配准误差
```

---

## API 端点汇总

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| GET | `/api/config` | 获取预设配置（区域、模型、训练参数） |
| GET | `/api/files` | 列出数据目录下所有文件 |
| POST | `/api/data/download-pair` | 启动双时相影像下载任务 |
| GET | `/api/data/download-pair/{task_id}` | 查询下载进度 |
| GET | `/api/data/pairs` | 列出已有影像对 |
| GET | `/api/data/regions` | 获取预设区域列表 |
| GET | `/api/data/preview/{subdir}/{filename}` | GeoTIFF → PNG 预览 |
| POST | `/api/detect/train` | 启动模型训练 |
| POST | `/api/detect/run` | 启动变化检测推理 |
| GET | `/api/detect/status/{task_id}` | 查询训练/检测任务状态 |
| GET | `/api/detect/models` | 列出可用模型权重 |
| GET | `/api/detect/results` | 列出检测结果文件 |
| POST | `/api/compare/visualize` | 生成对比可视化图 |
| POST | `/api/compare/overlay` | 生成变化叠加图 |
| POST | `/api/compare/stats` | 变化像素统计 |
| GET | `/api/compare/preview/{filename}` | 矢量文件 → GeoJSON |

---

## 快速启动

```bash
# 方式一：一键启动（自动检测并安装依赖）
python start.py

# 方式二：手动启动
pip install -r requirements.txt
cd frontend && npm install && cd ..
python -m uvicorn backend.main:app --reload --port 8000 &
cd frontend && npm run dev

# 生成模拟训练样本（无需下载真实数据）
python scripts/generate_sample.py

# 训练模型
python scripts/train.py --model siamese_unet --epochs 30 --batch-size 4

# 执行变化检测
python scripts/predict.py \
  --image-a data/raw/time_a/s2_beijing_A.tif \
  --image-b data/raw/time_b/s2_beijing_B.tif \
  --model data/models/best_model.pth \
  --threshold 0.5 --smoothing 1.0 --min-area 30
```

启动后访问 `http://localhost:5173` 进入前端界面，`http://localhost:8000/docs` 查看 API 文档。
