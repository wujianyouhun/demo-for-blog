# 基于 SAM 的遥感影像半自动标注实战：从环境搭建到标签导出的完整指南

## 前言

做过遥感深度学习的人都知道，标注是最耗时的环节。一栋建筑物在高分影像中可能有几百个像素，在 GIS 软件里逐个勾画 Polygon，一个熟练标注员处理一平方公里的高分影像可能需要数小时。

2023 年 Meta 发布的 SAM（Segment Anything Model）改变了这个局面。它是一个通用的图像分割基础模型，只需要一个点或一个框作为"提示"，就能输出精确的分割边界。将它配合 GroundingDINO 目标检测模型使用，甚至可以做到"输入文本描述 → 自动检测 → 精确分割"的全自动工作流。

但 SAM 毕竟是为自然图像设计的，直接用在遥感影像上有不少坑要踩：GeoTIFF 怎么加载？地理坐标怎么保留？模型权重放在哪？显存不够怎么办？

这篇文章分享一个我在实际项目中搭建的完整工具包，覆盖了从影像加载到标签导出的全流程，包含可运行的代码和调试过程中踩过的坑。

## 项目能做什么

整个项目是一个可复用的遥感半自动标注工程，核心封装为 `geoai_sam` 包，提供 5 个主要类：

```
SAMWrapper            → SAM 模型封装（点/框/自动分割）
GroundedSAMWrapper    → 文本驱动自动检测+分割
MaskPostProcessor     → Mask 后处理（链式调用）
MaskVectorizer        → 栅格 Mask → Polygon 矢量
QualityMetrics        → 标签质量量化评估
```

支持的标注模式有四种：**点提示**、**框提示**、**文本提示**和**自动分割**。生产中最推荐的是框提示（精度高）和文本提示（全自动），下面逐一介绍。

## 环境搭建

项目基于 conda 的 `geoai` 环境，Python 3.10/3.11 均可。建议有一块 NVIDIA 显卡（6GB 显存即可），没有 GPU 也能跑（CPU 模式，速度慢 10~50 倍）。准备一份 GeoTIFF 格式的遥感影像用于实践。

核心依赖关系如下：

```
PyTorch (计算引擎)
    ├── SAM / SAM2 / SAM3 (分割模型)
    ├── GroundingDINO (目标检测)
    └── CLIP (语义匹配)
        ↓
samgeo / geoai-py (遥感接口层)
        ↓
geoai_sam (本项目封装层)
```

安装步骤：

```bash
conda activate geoai
cd sam/
pip install -r requirements.txt
```

遥感相关库（rasterio、geopandas）在 Windows 上 pip 安装可能失败，建议用 conda：

```bash
conda install -c conda-forge rasterio geopandas shapely pyproj
```

### 显存适配

SAM 有三个规模的模型可选：

| 模型 | 参数量 | 文件大小 | 显存需求 | 适用场景 |
|------|--------|----------|----------|----------|
| vit_b | 91M | ~375MB | 2~4GB | 快速测试 |
| vit_l | 308M | ~1.2GB | 4~8GB | **日常使用（推荐）** |
| vit_h | 636M | ~2.5GB | 8GB+ | 精细任务 |

如果你的显存只有 6GB，项目默认使用 `vit_l`，精度和速度的平衡最好。在 `.env` 文件或代码中都可以配置：

```python
sam = SAMWrapper(model_type="vit_l")   # 默认，适合 6GB 显存
sam = SAMWrapper(model_type="vit_b")   # 显存更小时使用
```

### 模型缓存管理

SAM 的权重文件有 1.2GB，默认会下载到 `~/.cache` 目录。项目中通过设置环境变量将所有模型缓存重定向到了项目的 `models/` 目录：

```python
# geoai_sam/__init__.py 中的关键配置
os.environ.setdefault("TORCH_HOME", "项目路径/models/torch")
os.environ.setdefault("HF_HOME", "项目路径/models/huggingface")
os.environ.setdefault("HF_HUB_CACHE", "项目路径/models/huggingface/hub")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "项目路径/models/sentence_transformers")
os.environ.setdefault("CLIP_CACHE", "项目路径/models/clip")
```

这样 SAM 权重、GroundingDINO、CLIP、SentenceTransformers 等模型都集中存放在项目目录下，方便管理和迁移。

## 快速上手：50 行代码跑通全流程

项目的 `quick_start.py` 用不到 50 行代码演示了完整的标注流程：

```python
from geoai_sam import SAMWrapper, MaskPostProcessor, MaskVectorizer

# Step 1: 加载模型和影像
sam = SAMWrapper(model_type="vit_l")
sam.set_image("西安19级.tif")

# Step 2: 框提示分割
masks = sam.generate_masks_by_box(box=[100, 120, 500, 480])
sam.save_masks("output/raw_mask.tif")

# Step 3: 后处理（去噪 → 填充 → 平滑）
mask = sam.masks[0]  # 取最佳候选
clean_mask = MaskPostProcessor.default_pipeline(
    mask, min_size=200, fill_holes_flag=True, smooth_sigma=1.5,
)

# Step 4: 矢量化导出为 GeoJSON
vectorizer = MaskVectorizer()
gdf = vectorizer.vectorize(clean_mask, reference_image="西安19级.tif", min_area=50)
vectorizer.save("output/buildings.geojson")
```

> 以上为核心逻辑的精简版（不到 50 行），完整代码（含路径处理和错误处理）见项目中的 `quick_start.py`。

运行命令：

```bash
conda activate geoai
python quick_start.py
```

## 四种标注模式详解

### 点提示：最简单的交互

在目标上点击一个或多个点，SAM 推断分割区域。支持前景点（label=1）和背景点（label=0）的组合：

```python
# 单点标注
masks = sam.generate_masks_by_points([[750, 370]])

# 前景 + 背景联合标注
masks = sam.generate_masks_by_points(
    points=[[750, 370], [1125, 625]],
    point_labels=[1, 0],   # 1=前景(目标), 0=背景(排除)
)
```

**调参技巧：** 点击位置尽量靠近目标中心，避免点在边缘。当 SAM 默认结果包含了不想要的区域时，在那些区域添加背景点可以有效修正。

### 框提示：遥感场景首选

框提示是遥感标注**最推荐**的方式。建筑物轮廓清晰、边界明确，框提示的分割精度显著高于单点。

```python
# 单框
masks = sam.generate_masks_by_box(box=[100, 120, 500, 480])

# 多框批量
masks = sam.generate_masks_by_boxes(boxes=[
    [100, 120, 500, 480],
    [600, 300, 900, 700],
])
```

框的大小应该**略大于**目标范围，每边多出 5~15 像素。太紧会截断边界，太松会包含过多背景。

### 文本提示：最有生产价值的模式

输入一段文本描述，GroundingDINO 自动检测匹配的目标，SAM 自动分割——完全无需人工交互：

```python
from geoai_sam import GroundedSAMWrapper

gsam = GroundedSAMWrapper(sam_model_type="vit_l")
gsam.set_image("西安19级.tif")

# 单关键词
masks = gsam.segment_by_text("building")

# 多关键词（用 . 分隔，提高召回率）
masks = gsam.segment_by_text("building. house. roof")
```

遥感常用提示词参考：

| 目标类型 | 推荐提示词 |
|----------|------------|
| 建筑物 | `building. house. roof` |
| 水体 | `water. lake. river. pond` |
| 道路 | `road. highway` |
| 植被 | `tree. forest. vegetation` |
| 光伏板 | `solar panel. solar farm` |

关键参数是 `box_threshold` 和 `text_threshold`（默认 0.25）：调低到 0.1~0.2 可以多检但可能误检，调高到 0.4~0.5 则精度优先但可能遗漏。大规模场景建议先用文本提示粗筛，再框提示精修。

### 自动分割：全图扫描

不需要任何提示，SAM 对全图网格采样，自动发现所有目标。适合快速了解影像中有哪些可分割目标：

```python
sam = SAMWrapper(model_type="vit_l", automatic=True)
sam.set_image("西安19级.tif")
masks = sam.generate_masks_auto(points_per_side=32)
```

四种模式的对比总结：

| 维度 | 点提示 | 框提示 | 文本提示 | 自动分割 |
|------|--------|--------|----------|----------|
| 人工交互量 | 低 | 中 | 极低 | 无 |
| 精度 | 中 | **高** | 中 | 中 |
| 速度 | 快 | 快 | 极快 | 慢 |
| 推荐度 | ★★★ | **★★★★★** | ★★★★ | ★★ |

## 图形化交互标注

除了代码调用，项目还提供了基于 matplotlib 的图形化交互标注工具 `interactive_annotate.py`。启动后可以在影像窗口上直接操作：

```bash
python interactive_annotate.py --image 西安19级.tif
python interactive_annotate.py --image 西安19级.tif --mode box
python interactive_annotate.py --image 西安19级.tif --mode text --prompt building
```

窗口操作方式：
- **点标注：** 左键放前景点，右键放背景点，中键或按 S 执行分割
- **框标注：** 左键依次点击两个角点绘制矩形，中键执行分割
- **快捷键：** S 分割 / C 清空输入 / Esc 全部清空

大影像（如 29000x26000 像素）会自动降采样到 2048px 显示，鼠标点击的坐标自动映射回原图分辨率再传给 SAM。SAM 模型采用延迟加载策略，首次分割时才初始化，不阻塞窗口启动。

## Mask 后处理：从粗糙到精细

SAM 生成的原始 Mask 通常有毛刺、小碎片、孔洞和边界抖动。后处理是生成高质量训练标签的关键步骤。

推荐的后处理顺序：开运算（去毛刺）→ 去小斑块 → 孔洞填充 → 闭运算（填裂缝）→ 边界平滑。

`MaskPostProcessor` 支持链式调用，灵活组合各操作：

```python
clean_mask = (
    MaskPostProcessor()
    .load(raw_mask)
    .opening(radius=2)          # 去毛刺
    .remove_small_objects(200)  # 去小斑块
    .fill_holes()               # 孔洞填充
    .closing(radius=3)          # 填裂缝
    .smooth(sigma=1.5)          # 平滑边界
    .get_result()
)
```

如果不确定参数，用一键默认流程：

```python
clean_mask = MaskPostProcessor.default_pipeline(
    raw_mask, min_size=200, fill_holes_flag=True, smooth_sigma=1.5,
)
```

不同场景的参数调优参考：

| 场景 | min_size | 孔洞填充 | 开运算 | 闭运算 | 平滑 |
|------|----------|----------|--------|--------|------|
| 建筑物 | 200 | 是 | 2 | 3 | 1.5 |
| 水体 | 500 | 是 | 2 | 5 | 2.0 |
| 道路 | 100 | 否 | 1 | 2 | 1.0 |
| 植被 | 300 | 否 | 2 | 3 | 1.5 |

## 矢量化导出

后处理完成后，用 `MaskVectorizer` 将栅格 Mask 转为 Polygon：

```python
vectorizer = MaskVectorizer()
gdf = vectorizer.vectorize(
    clean_mask,
    reference_image="西安19级.tif",  # 保留地理参考信息
    min_area=100,
)
```

支持导出多种矢量格式，根据扩展名自动选择驱动：

```python
vectorizer.save("output/buildings.geojson")   # Web 可视化
vectorizer.save("output/buildings.gpkg")       # GIS 推荐首选
vectorizer.save("output/buildings.shp")        # 传统 GIS
```

## 质量评估

如果有真值标签（Ground Truth），可以用 `QualityMetrics` 做量化评估：

```python
from geoai_sam import QualityMetrics

# pred_mask: SAM 生成的预测 Mask (二值 numpy 数组)
# gt_mask:   人工标注的真值 Mask (二值 numpy 数组)
results = QualityMetrics.evaluate(
    prediction=pred_mask,
    ground_truth=gt_mask,
    include_boundary=True,
)
QualityMetrics.print_report(results)
```

报告包含 IoU、Dice、Precision、Recall、F1、面积误差和 Hausdorff 距离等指标。一般来说 IoU >= 0.75 就算良好的标签质量。

## 调试过程中的坑

在开发这个项目的过程中，我踩了几个值得记录的坑：

**1. SamGeo 的 checkpoint 参数名陷阱**

这是最隐蔽的 bug。我们的 `SAMWrapper` 把模型下载到 `models/` 目录后，传 `checkpoint_path=本地路径` 给 SamGeo，结果 SamGeo 完全忽略了这个参数，重新从网上下载了一份模型。

查了 SamGeo 源码才发现，它检查的 kwargs key 是 `"checkpoint"` 而不是 `"checkpoint_path"`：

```python
# SamGeo 源码中的逻辑
if "checkpoint" in kwargs:
    checkpoint = kwargs["checkpoint"]
else:
    checkpoint = download_checkpoint(...)  # 重新下载！
```

修复方法：传参时用 `sam_kwargs["checkpoint"] = 本地路径`，而不是作为命名参数传递。

**2. int 坐标导致 predict 走错分支**

SamGeo 的 `predict()` 方法内部通过 `isinstance(boxes[0], float)` 做分支判断。传 int 坐标 `[100, 120, 500, 480]` 时条件为 False，错误地走进了 `predict_torch` 路径，而该路径对 numpy 数组的处理有 bug（没转 torch tensor），直接报 `numpy.ndarray has no attribute 'clone'`。

解决方案是在调用前自动转为 float：

```python
box = [float(v) for v in box]
```

**3. 大影像内存溢出**

29248 x 25984 的 GeoTIFF 加载为 float64 需要 17GB 内存。解决方案是用 rasterio 的 `out_shape` 参数直接在读取时降采样，而不是先全图加载再 resize：

```python
img = src.read(
    list(range(1, bands + 1)),
    out_shape=(bands, target_h, target_w),
    resampling=Resampling.average,
)
```

**4. matplotlib 中文乱码**

Windows 上 matplotlib 默认字体不支持中文，所有中文标题和标签显示为方块。在包初始化时自动配置中文字体即可全局解决：

```python
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
```

## 项目结构一览

```
sam/
├── geoai_sam/                  # 核心包（5 个模块）
│   ├── __init__.py             # 包入口 + 中文字体配置
│   ├── core.py                 # SAMWrapper
│   ├── grounded_sam.py         # GroundedSAMWrapper
│   ├── postprocess.py          # MaskPostProcessor
│   ├── vectorize.py            # MaskVectorizer
│   └── quality.py              # QualityMetrics
├── demo_point_prompt.py        # 点提示演示
├── demo_box_prompt.py          # 框提示演示
├── demo_text_prompt.py         # 文本提示演示
├── demo_postprocess.py         # 后处理演示
├── interactive_annotate.py     # 图形化交互工具
├── quick_start.py              # 快速入门
├── config.py                   # 全局配置
├── requirements.txt            # 依赖清单
├── .env.example                # 环境变量模板
├── models/                     # 模型权重（自动创建）
└── output/                     # 输出结果
```

每个 `demo_*.py` 都是独立可运行的完整流程，适合对照学习。`interactive_annotate.py` 是面向最终用户的图形化交互工具。

## 总结

SAM 在遥感标注场景中的价值在于：它是一个预训练的通用分割模型，不需要针对遥感场景额外训练，只需极少的交互（一个点或一个框）就能输出精确的分割边界。配合 GroundingDINO，甚至只需输入文本描述就能全自动完成。实际测试中，SAM 辅助标注通常可以减少 70%~90% 的人工标注时间。

这个项目的核心价值在于把"调 SAM API"这个零散的过程封装成了一个完整的工程化方案，解决了遥感影像特有的问题：GeoTIFF 地理信息保留、模型缓存管理、显存适配、后处理流水线、矢量导出和质量评估。代码结构清晰，可以直接作为二次开发的基础框架使用。
