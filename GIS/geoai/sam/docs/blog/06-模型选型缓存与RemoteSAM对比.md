# SAM、GroundingDINO、RemoteSAM：遥感标注项目中的模型选型与踩坑

## 导语

遥感标注项目里，模型选型经常会被低估。

同样是“用 SAM 做分割”，不同模型版本、不同提示方式、不同依赖组合，实际体验差异很大。更麻烦的是，模型权重经常自动下载到系统缓存目录，换机器后复现困难。

本文从工程角度梳理 SAM、GroundingDINO、LangSAM、RemoteSAM 的定位，以及项目中模型缓存和环境排错的经验。

## SAM1、SAM2、SAM3 怎么选

项目封装了统一入口：

```python
SAMWrapper(model_type="vit_l", sam_version="sam1")
```

常见模型类型：

| 模型 | 显存建议 | 特点 |
| --- | --- | --- |
| `vit_b` | 2~4GB | 快，适合调试 |
| `vit_l` | 4~8GB | 平衡，推荐默认 |
| `vit_h` | 8GB+ | 精度高，显存压力大 |

实际工程中，不建议一开始就上 `vit_h`。  
先用 `vit_b` 或 `vit_l` 跑通流程，再根据边界质量决定是否换大模型。

对于 6GB 显卡，`vit_l` 是比较合理的默认选择。

## 文本标注需要什么模型

SAM 本身不理解“building”这种文本类别。文本标注通常需要额外模型：

```text
文本提示
  -> GroundingDINO / LangSAM 找候选框
  -> SAM 对候选框分割
  -> 输出目标 Mask
```

所以文本模式的依赖比点、框模式更复杂。常见问题包括：

- GroundingDINO 未安装。
- transformers 版本不匹配。
- HuggingFace 模型没下载完整。
- 模型加载时触发 meta tensor 错误。

工程上要把文本模式当成增强能力，而不是基础能力。基础标注链路应保证点和框模式稳定可用。

## 模型缓存必须项目化

很多 Python 库默认把模型下载到用户目录，例如：

```text
C:\Users\<user>\.cache
```

这对项目复现很不友好。换机器、换用户、换环境后，模型是否存在很难判断。

本项目把缓存统一重定向到：

```text
models/
```

包括：

| 环境变量 | 目录 |
| --- | --- |
| `TORCH_HOME` | `models/torch` |
| `HF_HOME` | `models/huggingface` |
| `HF_HUB_CACHE` | `models/huggingface/hub` |
| `TRANSFORMERS_CACHE` | `models/huggingface/transformers` |
| `CLIP_CACHE` | `models/clip` |
| `SAMGEO_CACHE` | `models/samgeo` |

启动后可以访问：

```text
http://127.0.0.1:8001/api/config
```

检查 `model_dir` 和 `model_cache` 是否都指向项目目录。

## 模型文件完整性很重要

SAM 权重很大，下载中断后文件可能存在但不完整。

常见 SAM1 文件大小：

| 模型 | 文件名 | 大小 |
| --- | --- | --- |
| `vit_b` | `sam_vit_b_01ec64.pth` | 约 376MB |
| `vit_l` | `sam_vit_l_0b3195.pth` | 约 1192MB |
| `vit_h` | `sam_vit_h_4b8939.pth` | 约 2448MB |

如果文件明显偏小，应删除后重新下载。

## RemoteSAM 和本项目有什么区别

RemoteSAM 是面向地球观测的统一视觉模型，目标不只是交互标注。

它以文本表达和像素级预测为核心，进一步支持：

- Referring Expression Segmentation。
- Semantic Segmentation。
- Object Detection。
- Visual Grounding。
- Classification。
- Counting。
- Captioning。

而当前 GeoAI SAM 项目的定位是标注工具：

| 维度 | GeoAI SAM 标注平台 | RemoteSAM |
| --- | --- | --- |
| 目标 | 生产 GIS 标签 | 统一遥感视觉模型 |
| 输入 | 点、框、文本、自动模式 | 图像 + 文本表达或类别 |
| 输出 | Mask、GeoJSON、Shapefile、GPKG | 多任务预测结果 |
| 工程重点 | CRS、GeoTIFF、Web 标注、矢量导出 | 训练、评测、多任务模型 |
| 使用门槛 | 更偏应用 | 更偏研究 |

简单说：RemoteSAM 更像“遥感基础模型研究框架”，本项目更像“遥感标签生产工具”。

## 常见环境坑

### 1. CUDA out of memory

先换 `vit_b`，再降低瓦片尺寸。整图处理时可以把 `tile_size` 从 2048 降到 1024。

### 2. Cannot copy out of meta tensor

这通常和 transformers pipeline、device_map 或 meta tensor 初始化有关。工程策略是优先走更稳定的 LangSAM 路径，并把 HuggingFace 缓存固定到项目目录。

### 3. 前端页面打开但不处理

先访问：

```text
http://127.0.0.1:5173/api/config
```

如果不是 JSON，说明前端代理没有打到后端，可能是 5173 被错误服务占用。

### 4. rasterio 或 GDAL 导入失败

优先用 conda-forge 安装地理空间依赖：

```powershell
conda install -c conda-forge rasterio geopandas shapely
```

这比纯 pip 更容易解决 GDAL 依赖。

## 选型建议

如果是标注生产：

```text
框提示 + vit_l + 后处理 + GeoPackage
```

如果是批量预标注：

```text
文本提示 + GroundingDINO/LangSAM + 瓦片化整图处理
```

如果是研究遥感多任务基础模型：

```text
RemoteSAM
```

如果是低显存电脑快速验证：

```text
vit_b + 小瓦片 + 少量样例
```

## 小结

模型选型不是单纯追求最新或最大，而是围绕任务约束做工程权衡：

- 显存够不够。
- 权重能否稳定加载。
- 是否需要文本能力。
- 是否要导出 GIS 标签。
- 是否面向研究还是生产。

对于一个遥感标注平台，稳定、可复现、能导出正确坐标的结果，往往比单次 demo 的模型效果更重要。

