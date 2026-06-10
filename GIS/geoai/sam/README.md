# GeoAI SAM — 基于 SAM 的遥感半自动标注项目

基于 Segment Anything Model (SAM) 系列模型的遥感影像半自动标注工具。通过点提示、框提示、文本提示等多种交互式方式驱动 SAM 进行目标分割，再经后处理和矢量化流程，快速生成 GeoAI 训练标签，大幅降低人工矢量化成本。

> 详细文档请查看 [`docs/`](./docs/) 目录。

---

## 目录

- [项目简介](#项目简介)
- [集成模型](#集成模型)
- [项目结构](#项目结构)
- [环境准备与安装](#环境准备与安装)
- [快速开始](#快速开始)
- [标注模式](#标注模式)
- [后处理与导出](#后处理与导出)
- [调试与运行](#调试与运行)
- [常见问题](#常见问题)
- [文档索引](#文档索引)

---

## 项目简介

传统遥感标签制作流程依赖人工逐像素勾画 Polygon，存在人工成本高、一致性差、大面积区域效率低等问题。本项目引入 SAM 模型后，工作流变为"模型自动分割 + 人工修正 + 标签导出"，通常可减少 70%~90% 的人工标注时间。

完整工作流如下:

```
原始影像 → 加载SAM模型 → 交互提示(点/框/文本) → SAM生成Mask
    → 后处理(去噪/平滑/孔洞填充/面积过滤) → 矢量化(Mask→Polygon)
    → 质量评估(IoU/Dice/边界误差) → 训练标签(GeoJSON/Shapefile/GeoPackage)
```

## 集成模型

| 模型 | 说明 | 在本项目中的角色 |
|------|------|------------------|
| **SAM** (vit_b / vit_l / vit_h) | Segment Anything Model | 核心分割引擎，接受点/框提示生成分割 Mask |
| **SAM2** | SAM 第二代 | 增强分割能力，支持时序数据 |
| **SAM3** | SAM 第三代 | 进一步提升分割精度 |
| **GroundingDINO** | 开放词汇目标检测 | 接收文本提示，输出检测框，驱动 SAM 分割 |
| **CLIP** | 对比语言-图像模型 | 对候选 Mask 做语义排序与筛选 |
| **PyTorch** | 深度学习计算框架 | 所有模型的底层运行引擎 |

## 项目结构

```
sam/
│
├── geoai_sam/                        # 核心 Python 包
│   ├── __init__.py                   # 包入口，统一导出 5 个核心类
│   ├── core.py                       # SAMWrapper — SAM/SAM2/SAM3 统一封装
│   ├── grounded_sam.py               # GroundedSAMWrapper — GroundingDINO + SAM + CLIP
│   ├── postprocess.py                # MaskPostProcessor — Mask 后处理(链式调用)
│   ├── vectorize.py                  # MaskVectorizer — Mask 转 Polygon 矢量化
│   └── quality.py                    # QualityMetrics — IoU/Dice/F1 等质量评估
│
├── demo_point_prompt.py              # 演示脚本: 点提示标注
├── demo_box_prompt.py                # 演示脚本: 框提示标注
├── demo_text_prompt.py               # 演示脚本: 文本提示标注 (Grounded-SAM)
├── demo_postprocess.py               # 演示脚本: Mask 后处理全流程
│
├── interactive_annotate.py           # 交互式标注工具 (命令行 CLI)
├── quick_start.py                    # 快速入门 (最短可运行示例)
│
├── config.py                         # 全局配置 (支持 .env 文件覆盖)
├── requirements.txt                  # Python 依赖清单
├── .env.example                      # 环境变量模板
├── .gitignore                        # Git 忽略规则
│
├── docs/                             # 详细文档
│   ├── 01-项目介绍.md
│   ├── 02-环境搭建.md
│   ├── 03-项目结构详解.md
│   ├── 04-标注模式详解.md
│   ├── 05-后处理与导出.md
│   ├── 06-调试与排错.md
│   └── 07-API参考.md
│
├── 西安19级.tif                       # 测试遥感影像 (19级瓦片)
├── models/                           # 模型权重目录 (首次运行自动下载)
└── output/                           # 运行结果输出目录
```

各模块的职责边界清晰: `core.py` 只负责 SAM 模型的加载与推理，`grounded_sam.py` 负责文本驱动的自动检测+分割，`postprocess.py` 专注 Mask 形态学后处理，`vectorize.py` 处理栅格到矢量的转换，`quality.py` 提供标签质量量化评估。

## 环境准备与安装

### 前置要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows 10/11, Linux, macOS |
| Python | 3.9 ~ 3.11 |
| Conda 环境 | `geoai` (已预装 PyTorch + CUDA) |
| GPU 显存 | vit_b: 2~4GB, vit_l: 4~8GB, vit_h: 8GB+ |
| 磁盘空间 | 模型权重约 1~2GB (首次运行自动下载到项目 `models/` 目录) |

### 安装步骤

```bash
# 1. 激活 conda 环境
conda activate geoai

# 2. 进入项目目录
cd sam/

# 3. 安装 Python 依赖
pip install -r requirements.txt

# 4. (可选) 复制环境变量模板并编辑
cp .env.example .env

# 5. 验证配置
python config.py
```

`config.py` 会打印当前所有配置项（模型类型、设备、阈值等），确认无误即可开始使用。

> 关于环境搭建的详细排查步骤，请参考 [docs/02-环境搭建.md](./docs/02-环境搭建.md)。

## 快速开始

### 最短可运行示例

```bash
python quick_start.py
```

该脚本自动完成: 加载影像 → 框提示分割 → 后处理 → 导出 Polygon。运行后查看 `output/quick_start/` 目录获取结果。

### 交互式标注工具

```bash
# 框提示标注 (推荐首选，默认 vit_l)
python interactive_annotate.py --image 西安19级.tif --mode box

# 点提示标注
python interactive_annotate.py --image 西安19级.tif --mode point

# 文本提示标注 (Grounded-SAM)
python interactive_annotate.py --image 西安19级.tif --mode text --prompt building

# 自动全图分割
python interactive_annotate.py --image 西安19级.tif --mode auto

# 标注 + 自动后处理 + 自动矢量化导出
python interactive_annotate.py --image 西安19级.tif --mode box --postprocess --vectorize

# 切换到 vit_b (更轻量) 或 vit_h (更高精度，需 8GB+ 显存)
python interactive_annotate.py --image 西安19级.tif --mode box --model-type vit_b
```

### 运行独立演示脚本

```bash
python demo_point_prompt.py      # 点提示: 单点/多点/正负联合
python demo_box_prompt.py        # 框提示: 单框/多框/联合提示
python demo_text_prompt.py       # 文本提示: building/water/road
python demo_postprocess.py       # 后处理: 去噪/填充/平滑/对比
```

每个脚本运行后会在 `output/` 下创建对应子目录，包含 Mask 文件、可视化图片和矢量结果。

## 标注模式

### 点提示 — 适合快速探索

```python
from geoai_sam import SAMWrapper

sam = SAMWrapper(model_type="vit_h")
sam.set_image("西安19级.tif")

# 单点标注
masks = sam.generate_masks_by_points([[750, 370]])

# 前景 + 背景联合提示
masks = sam.generate_masks_by_points(
    points=[[750, 370], [1125, 625]],
    point_labels=[1, 0],   # 1=前景, 0=背景
)
```

### 框提示 — 精度最高，推荐首选

```python
# 单框: [xmin, ymin, xmax, ymax]
masks = sam.generate_masks_by_box(box=[100, 120, 500, 480])

# 多框批量
masks = sam.generate_masks_by_boxes(boxes=[
    [100, 120, 500, 480],
    [600, 300, 900, 700],
])
```

### 文本提示 — Grounded-SAM，零样本自动分割

```python
from geoai_sam import GroundedSAMWrapper

gsam = GroundedSAMWrapper(sam_model_type="vit_h")
gsam.set_image("西安19级.tif")

masks = gsam.segment_by_text("building")
masks = gsam.segment_by_text("water. lake. river")
masks = gsam.segment_by_text_with_clip("building", clip_model="ViT-B-32")
```

> 详细的模式对比和参数调优请参考 [docs/04-标注模式详解.md](./docs/04-标注模式详解.md)。

## 后处理与导出

### Mask 后处理（链式调用）

```python
from geoai_sam import MaskPostProcessor

processor = MaskPostProcessor()
clean_mask = (
    processor
    .load(raw_mask)
    .remove_small_objects(min_size=200)
    .fill_holes()
    .opening(radius=2)
    .closing(radius=3)
    .smooth(sigma=1.5)
    .get_result()
)

# 或一行代码
clean_mask = MaskPostProcessor.default_pipeline(raw_mask)
```

### Mask → Polygon 矢量化

```python
from geoai_sam import MaskVectorizer

vectorizer = MaskVectorizer()
gdf = vectorizer.vectorize(clean_mask, reference_image="西安19级.tif", min_area=100)
vectorizer.save("output/buildings.geojson")   # GeoJSON
vectorizer.save("output/buildings.gpkg")      # GeoPackage
vectorizer.save("output/buildings.shp")       # Shapefile
```

### 质量评估

```python
from geoai_sam import QualityMetrics

results = QualityMetrics.evaluate(pred_mask, gt_mask)
QualityMetrics.print_report(results)
# 输出: IoU, Dice, Precision, Recall, F1, 面积误差, Hausdorff 距离
```

> 后处理参数调优与导出格式详细说明请参考 [docs/05-后处理与导出.md](./docs/05-后处理与导出.md)。

## 调试与运行

### 快速检查环境

```bash
# 检查配置
python config.py

# 检查依赖是否安装完整
python -c "from geoai_sam import SAMWrapper; print('核心包导入成功')"

# 检查 GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### 分步调试建议

1. **先用小模型快速验证**: 将 `model_type` 设为 `vit_b`，避免首次下载大模型耗时
2. **先跑 `quick_start.py`**: 验证整条流水线是否通畅
3. **逐步切换演示脚本**: 从 `demo_box_prompt.py` 开始，逐步尝试各模式
4. **使用 `--postprocess` 参数**: 自动执行后处理，免去手动操作
5. **检查 `output/` 目录**: 每次运行后查看图片和 GeoJSON 确认效果

### 常见问题速查

| 问题 | 解决方案 |
|------|----------|
| `ModuleNotFoundError: samgeo` | `pip install samgeo` |
| `CUDA out of memory` | 换用 `vit_l` 或 `vit_b`，或设 `DEVICE=cpu` |
| 模型下载慢/失败 | 手动下载 `.pth` 放入 `models/`，设 `SAM_CHECKPOINT_PATH` |
| 模型下载到了 .cache | 项目已自动重定向，如仍有问题运行 `python config.py` 检查环境变量 |
| `rasterio` 安装失败 | `conda install -c conda-forge rasterio` |
| GroundingDINO 不可用 | 文本模式回退到 LangSAM；或 `pip install groundingdino-py` |

> 完整的排错指南请参考 [docs/06-调试与排错.md](./docs/06-调试与排错.md)。

## 模型选择指南

| 模型 | 参数量 | 显存 | 速度 | 精度 | 推荐场景 |
|------|--------|------|------|------|----------|
| **vit_b** | 91M | 2~4GB | 快 | 中 | 快速验证、GPU 资源有限 |
| **vit_l** | 308M | 4~8GB | 中 | 较高 | **默认推荐，适合 6GB 显存** |
| vit_h | 636M | 8GB+ | 慢 | 高 | 建筑物提取、精细分割 (需大显存) |

## 标注模式推荐

| 模式 | 效率 | 精度 | 推荐度 | 适用场景 |
|------|------|------|--------|----------|
| 纯人工矢量化 | 低 | 高 | ★★ | 精度要求极高的验收场景 |
| 点提示 SAM | 高 | 中 | ★★★ | 快速探索、目标模糊 |
| 框提示 SAM | 高 | 高 | ★★★★★ | 建筑提取（首选） |
| 文本提示 SAM | 极高 | 中 | ★★★★ | 大批量零样本预标注 |
| GroundedSAM + 人工修正 | 极高 | 高 | ★★★★★ | 生产级标注流水线 |

## 遥感常用文本提示词

| 类别 | 提示词 |
|------|--------|
| 建筑物 | `building`, `house`, `roof`, `construction` |
| 水体 | `water`, `lake`, `river`, `pond`, `reservoir` |
| 道路 | `road`, `highway`, `street`, `path` |
| 植被 | `tree`, `forest`, `vegetation`, `grass` |
| 光伏板 | `solar panel`, `solar farm`, `photovoltaic` |
| 车辆 | `car`, `truck`, `vehicle`, `bus` |
| 农田 | `farmland`, `crop field`, `agricultural land` |

多个关键词用 `.` 分隔，如 `"water. lake. river"`。

## 文档索引

| 文档 | 内容 |
|------|------|
| [01-项目介绍](./docs/01-项目介绍.md) | 项目背景、SAM 在遥感中的价值、工作流全景 |
| [02-环境搭建](./docs/02-环境搭建.md) | Conda 环境配置、依赖安装、GPU/CUDA 排查 |
| [03-项目结构详解](./docs/03-项目结构详解.md) | 每个文件的职责、设计思路、模块间关系 |
| [04-标注模式详解](./docs/04-标注模式详解.md) | 点/框/文本/自动四种模式的参数调优与最佳实践 |
| [05-后处理与导出](./docs/05-后处理与导出.md) | 后处理参数选择、矢量格式对比、质量评估 |
| [06-调试与排错](./docs/06-调试与排错.md) | 常见报错、性能调优、调试技巧 |
| [07-API参考](./docs/07-API参考.md) | 全部类和函数的完整 API 文档 |

## 许可

本项目仅供学习参考使用。
