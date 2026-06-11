## 07 - API 参考

本文档列出 `geoai_sam` 包中全部公开类和方法的完整签名与说明。

---

### SAMWrapper

SAM / SAM2 / SAM3 模型的统一封装。

```python
from geoai_sam import SAMWrapper
```

#### 构造函数

```python
SAMWrapper(
    model_type: str = "vit_h",
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
    automatic: bool = False,
    sam_version: str = "sam1",
    model_dir: str = "./models",
    **kwargs
)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_type` | str | `"vit_h"` | 模型规模: `"vit_b"`, `"vit_l"`, `"vit_h"` |
| `checkpoint_path` | Optional[str] | None | 权重路径，None 自动下载 |
| `device` | Optional[str] | None | 运行设备，None 自动检测 |
| `automatic` | bool | False | 是否自动分割模式 |
| `sam_version` | str | `"sam1"` | SAM 版本: `"sam1"`, `"sam2"`, `"sam3"` |
| `model_dir` | str | `"./models"` | 模型权重目录 |

#### 方法

**`set_image(image_path: str) -> None`**

加载遥感影像。支持 GeoTIFF、PNG、JPG 等格式。

---

**`generate_masks_by_points(points, point_labels=None, multimask_output=True, **kwargs) -> np.ndarray`**

点提示分割。

| 参数 | 类型 | 说明 |
|------|------|------|
| `points` | List[List[int]] | 坐标列表 `[[x,y], ...]` |
| `point_labels` | Optional[List[int]] | 标签列表，1=前景 0=背景。None 时全前景 |
| `multimask_output` | bool | True 输出 3 个候选 mask |

---

**`generate_masks_by_box(box, multimask_output=True, **kwargs) -> np.ndarray`**

单框提示分割。

| 参数 | 类型 | 说明 |
|------|------|------|
| `box` | List[int] | `[xmin, ymin, xmax, ymax]` |
| `multimask_output` | bool | True 输出 3 个候选 mask |

---

**`generate_masks_by_boxes(boxes, multimask_output=True, **kwargs) -> np.ndarray`**

多框批量分割。

| 参数 | 类型 | 说明 |
|------|------|------|
| `boxes` | List[List[int]] | 框列表 `[[xmin,ymin,xmax,ymax], ...]` |

---

**`generate_masks_auto(points_per_side=32, pred_iou_thresh=0.88, stability_score_thresh=0.95, min_mask_region_area=100, **kwargs) -> np.ndarray`**

全图自动分割（需要 `automatic=True`）。

---

**`save_masks(output_path: str, **kwargs) -> str`**

保存 Mask 到文件。

---

**`show_masks(figsize=(12,8), title="SAM Segmentation Result", save_path=None, cmap="viridis") -> None`**

可视化 Mask 结果。`save_path` 不为 None 时保存为图片。

---

**`get_mask_info() -> Dict[str, Any]`**

返回当前 Mask 的信息字典: 形状、类型、前景像素数、覆盖率等。

#### 属性

**`masks`** — 最近一次生成的 Mask (np.ndarray 或 None)。

---

### GroundedSAMWrapper

GroundingDINO + SAM + CLIP 文本驱动分割。

```python
from geoai_sam import GroundedSAMWrapper
```

#### 构造函数

```python
GroundedSAMWrapper(
    sam_model_type: str = "vit_h",
    groundingdino_model: str = "GroundingDINO_SwinB",
    device: Optional[str] = None,
    box_threshold: float = 0.25,
    text_threshold: float = 0.25,
    model_dir: str = "./models",
    **kwargs
)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sam_model_type` | str | `"vit_h"` | SAM 模型类型 |
| `groundingdino_model` | str | `"GroundingDINO_SwinB"` | DINO 模型: `SwinT` 或 `SwinB` |
| `box_threshold` | float | 0.25 | 检测框置信度阈值 |
| `text_threshold` | float | 0.25 | 文本匹配置信度阈值 |

#### 方法

**`set_image(image_path: str) -> None`**

加载影像。

---

**`segment_by_text(text_prompt, box_threshold=None, text_threshold=None, return_boxes=True, **kwargs) -> np.ndarray`**

文本提示分割。

| 参数 | 类型 | 说明 |
|------|------|------|
| `text_prompt` | str | 文本提示，如 `"building"` |
| `box_threshold` | Optional[float] | 覆盖默认阈值 |
| `text_threshold` | Optional[float] | 覆盖默认阈值 |

---

**`segment_by_text_with_clip(text_prompt, clip_model="ViT-B-32", **kwargs) -> np.ndarray`**

CLIP 增强的文本分割。

---

**`save_results(mask_path="output/grounded_sam_mask.tif", box_path=None) -> Dict[str, str]`**

保存 Mask 和检测框。

---

**`get_common_prompts(category: str) -> List[str]`** (classmethod)

获取遥感常用提示词。可选 category: `"building"`, `"water"`, `"road"`, `"vegetation"`, `"solar"`, `"vehicle"`, `"farmland"`。

#### 属性

**`masks`** — 最近生成的 Mask。
**`boxes`** — 检测到的边界框。
**`labels`** — 检测标签列表。

---

### MaskPostProcessor

Mask 后处理工具，支持链式调用。

```python
from geoai_sam import MaskPostProcessor
```

#### 方法

| 方法 | 参数 | 说明 |
|------|------|------|
| `load(mask)` | `mask: np.ndarray` | 加载 Mask (2D bool/uint8) |
| `remove_small_objects(min_size, connectivity)` | `min_size=200, connectivity=2` | 去除小斑块 |
| `fill_holes()` | 无 | 孔洞填充 |
| `opening(radius)` | `radius=2` | 开运算 (去毛刺) |
| `closing(radius)` | `radius=3` | 闭运算 (填裂缝) |
| `smooth(sigma, threshold)` | `sigma=1.5, threshold=0.5` | 高斯平滑 |
| `dilate(radius)` | `radius=2` | 膨胀 |
| `erode(radius)` | `radius=2` | 腐蚀 |
| `buffer(pixels)` | `pixels=3` | 缓冲区 (膨胀+腐蚀) |
| `filter_by_area(min_area, max_area)` | 面积范围 | 按面积过滤 |
| `keep_largest()` | 无 | 只保留最大区域 |
| `convex_hull()` | 无 | 凸包处理 |
| `get_result()` | 无 | 返回处理后 Mask (bool) |
| `get_result_uint8()` | 无 | 返回 uint8 Mask (0/255) |
| `get_operations_log()` | 无 | 返回操作日志列表 |
| `get_statistics()` | 无 | 返回统计信息字典 |
| `save(output_path, reference_image)` | 文件路径 | 保存为 GeoTIFF |
| `visualize(original_image, save_path, figsize)` | 可视化参数 | 对比可视化 |

所有处理方法均返回 `self`，支持链式调用。

**`default_pipeline(mask, min_size=200, fill_holes_flag=True, smooth_sigma=1.5, opening_radius=2, closing_radius=3) -> np.ndarray`** (static)

一键后处理流程: 开运算 → 去小斑块 → 孔洞填充 → 闭运算 → 平滑。

---

### MaskVectorizer

Mask → Polygon 矢量化工具。

```python
from geoai_sam import MaskVectorizer
```

#### 方法

**`vectorize(mask, reference_image=None, min_area=10, simplify_tolerance=0) -> GeoDataFrame | List`**

将 Mask 转为矢量 Polygon。返回 GeoDataFrame（含 id, area_pixels, geometry 列）或多边形列表。

---

**`save(output_path, driver=None) -> str`**

保存矢量结果。driver 根据扩展名自动推断: .geojson → GeoJSON, .shp → Shapefile, .gpkg → GeoPackage, .parquet → GeoParquet。

---

**`add_attribute(name, values) -> self`**

为多边形添加属性字段。

---

**`filter_by_area(min_area=None, max_area=None) -> self`**

按面积范围过滤多边形。

---

**`get_polygon_count() -> int`** — 多边形数量。
**`get_total_area() -> float`** — 所有多边形总面积。
**`get_gdf() -> GeoDataFrame`** — 获取 GeoDataFrame。

---

### QualityMetrics

标签质量评估（全部静态方法）。

```python
from geoai_sam import QualityMetrics
```

#### 静态方法

| 方法 | 参数 | 返回 | 说明 |
|------|------|------|------|
| `iou(pred, gt)` | 两个 Mask | float | IoU (交并比) |
| `dice(pred, gt)` | 两个 Mask | float | Dice 系数 |
| `precision(pred, gt)` | 两个 Mask | float | 精确率 |
| `recall(pred, gt)` | 两个 Mask | float | 召回率 |
| `f1_score(pred, gt)` | 两个 Mask | float | F1 分数 |
| `pixel_accuracy(pred, gt)` | 两个 Mask | float | 像素精度 |
| `area_error(pred, gt)` | 两个 Mask | dict | 面积误差 (绝对+相对) |
| `boundary_error(pred, gt)` | 两个 Mask | dict | Hausdorff 距离 |
| `evaluate(pred, gt, include_boundary=True)` | 两个 Mask | dict | **综合评估** (全部指标) |
| `print_report(results)` | 评估字典 | None | 打印格式化报告 |
| `batch_evaluate(preds, gts)` | Mask 列表 | dict | 批量评估平均值 |
