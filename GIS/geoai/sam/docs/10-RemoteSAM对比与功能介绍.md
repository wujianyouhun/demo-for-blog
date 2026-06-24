# RemoteSAM 对比与本项目功能介绍

本文从项目目标、技术路线、输入输出、适用场景和工程落地角度，对本项目与 RemoteSAM 进行对比，并说明本项目当前已经具备的功能边界。

> 说明：`F:\stduy\AI\GIS\RemoteSAM` 当前仍只有 `.git` 目录，HEAD 和工作区文件尚未检出完成。因此本文对 RemoteSAM 的分析基于其公开 README 与项目介绍信息，不包含本地源码级文件逐项审阅。待仓库下载完成后，可补充源码结构、模块调用和运行脚本级对比。

## 1. 项目定位对比

| 维度 | 本项目 GeoAI SAM | RemoteSAM |
| --- | --- | --- |
| 核心定位 | 面向遥感影像标注生产的半自动标注平台 | 面向地球观测场景的通用遥感基础模型 |
| 主要目标 | 加载 GeoTIFF、交互式提示分割、后处理、矢量化导出训练标签 | 用统一模型处理多类遥感视觉任务 |
| 用户入口 | Web 标注平台、FastAPI 接口、Python demo、CLI 交互脚本 | 模型推理、评测、数据集和论文复现实验 |
| 典型用户 | 标注人员、GIS 工程师、GeoAI 数据生产人员 | 遥感大模型研究者、算法工程师 |
| 工作方式 | 人机协同：用户给点/框/文本提示，SAM 生成 Mask | 模型驱动：统一视觉语言模型理解遥感图像和任务 |
| 输出重点 | Mask、GeoTIFF、GeoJSON、Shapefile、GeoPackage | 多任务结果，如分割、检测、分类、计数、描述等 |

一句话概括：本项目偏“工程标注工具链”，RemoteSAM 偏“遥感通用模型能力”。

## 2. RemoteSAM 功能概览

RemoteSAM 是面向 Remote Sensing / Earth Observation 的通用模型项目，公开介绍中强调 Unified Model for Remote Sensing。它关注的是在遥感图像上统一处理多种视觉任务，而不只是交互式分割。

从公开说明看，RemoteSAM 覆盖的能力包括：

| 能力 | 说明 |
| --- | --- |
| 分割 | 对遥感影像中的目标或区域生成像素级结果 |
| 检测 | 定位遥感目标，输出目标框或区域 |
| 分类 | 对图像、区域或目标进行类别判断 |
| 计数 | 统计遥感图像中特定目标数量 |
| 描述 | 结合视觉语言能力生成图像或目标描述 |
| 多任务统一 | 将不同遥感视觉任务放在统一模型框架中处理 |

RemoteSAM 更适合用来研究“一个模型能否理解多种遥感任务”。如果要把它接入生产标注平台，需要额外建设数据加载、Web 交互、GIS 坐标处理、矢量导出和人工修订流程。

## 3. 本项目功能概览

本项目已经形成从遥感影像到训练标签的端到端流程：

```text
GeoTIFF 影像
  -> Web/脚本加载
  -> 点标注 / 框标注 / 文本标注
  -> SAM 或 Grounded-SAM 生成 Mask
  -> Mask 后处理
  -> 矢量化
  -> GeoJSON / Shapefile / GeoPackage 导出
```

### 3.1 影像加载

本项目通过后端 `ImageService` 使用 `rasterio` 读取 GeoTIFF 元数据，包括：

| 信息 | 用途 |
| --- | --- |
| width / height | 原始像素尺寸 |
| CRS | 坐标转换和导出地理参考 |
| bounds | 地图定位和 Mask 叠加范围 |
| transform | 地理坐标与像素坐标互转 |
| band_count | 判断影像波段数量 |

Web 端使用 TiTiler 动态切片显示大幅 GeoTIFF，浏览器只加载当前视口瓦片，避免一次性读入全图。

### 3.2 标注模式

| 模式 | 输入 | 后端接口 | 适用场景 |
| --- | --- | --- | --- |
| 点标注 | 前景点、背景点 | `/api/predict/point` | 快速交互、边界需要人工提示的目标 |
| 框标注 | 地图上拖出的矩形框 | `/api/predict/box` | 建筑物、水体、光伏板等范围明确目标 |
| 文本标注 | 文本提示词 | `/api/predict/text` | 批量预标注、开放词汇目标 |
| 后处理 | Mask + 参数 | `/api/postprocess` | 去噪、补洞、平滑、连通 |
| 矢量导出 | Mask + 面积阈值 + 格式 | `/api/export/vectorize` | 生成 GIS 训练标签 |

### 3.3 后处理与导出

后处理由 `MaskPostProcessor.default_pipeline()` 执行，默认包含小斑块移除、孔洞填充、开闭运算和平滑。

矢量化由 `MaskVectorizer` 执行，支持 GeoJSON、Shapefile、GeoPackage 等常见 GIS 格式，便于直接进入 QGIS、ArcGIS 或训练数据流水线。

![Mask 后处理对比](images/mask-postprocessing-comparison.png)

![矢量化结果](images/vectorization-result.png)

## 4. 技术路线对比

| 层级 | 本项目 | RemoteSAM |
| --- | --- | --- |
| 模型核心 | SAM / SAM2 / SAM3 / GroundingDINO / CLIP | RemoteSAM 统一遥感模型 |
| 遥感数据处理 | rasterio、pyproj、TiTiler、geopandas、shapely | 更偏模型和数据集任务处理 |
| 前端交互 | Vue3 + OpenLayers | README 未显示其以交互式 Web 标注为主 |
| 后端服务 | FastAPI + 会话状态 + 推理接口 | 通常以脚本或模型推理流程为主 |
| 输出类型 | PNG Mask、GeoTIFF Mask、GeoJSON、GPKG、SHP | 多任务预测输出 |
| 工程目标 | 标注生产闭环 | 模型能力评测和泛化 |

## 5. 互补关系

二者不是简单替代关系，更适合组合使用：

| 组合方式 | 价值 |
| --- | --- |
| 用 RemoteSAM 做预检测或预理解，本项目做人工修订和矢量导出 | 将大模型理解能力转成可用 GIS 标签 |
| 用本项目积累高质量标注数据，再用于 RemoteSAM 类模型微调或评测 | 提升遥感基础模型在本地数据上的表现 |
| 将 RemoteSAM 作为文本标注或自动标注后端候选模型 | 增强本项目开放词汇和多任务能力 |
| 保留本项目的 Web/GIS 工程层，替换或扩展模型层 | 降低模型接入成本 |

## 6. 本项目后续可借鉴 RemoteSAM 的方向

1. 增加多任务入口：在分割之外支持检测、计数、描述。
2. 增加结果解释：对文本标注结果输出类别、置信度、检测框和自然语言说明。
3. 增加批处理任务：支持一批影像自动预标注。
4. 增加模型适配层：把 SAM、GroundedSAM、RemoteSAM 等封装成统一 `ModelBackend`。
5. 增加评测集管理：对每批标注结果计算 IoU、Dice、面积误差和目标级召回。

## 7. 选型建议

| 场景 | 推荐 |
| --- | --- |
| 需要制作可训练的 GIS Polygon 标签 | 优先使用本项目 |
| 需要对遥感图像做多任务视觉理解研究 | 优先研究 RemoteSAM |
| 需要 Web 交互式标注和人工修订 | 使用本项目 |
| 需要开放词汇、计数、描述等更强语义能力 | 评估 RemoteSAM 或将其接入本项目 |
| 需要生产闭环 | 本项目作为主流程，RemoteSAM 作为模型增强模块 |

## 8. 当前本项目功能清单

| 模块 | 状态 | 说明 |
| --- | --- | --- |
| Web 影像加载 | 已实现 | 默认影像路径可通过后端配置注入 |
| TiTiler 瓦片显示 | 已实现 | 支持大幅 GeoTIFF 按需显示 |
| 点标注 | 已实现 | 支持前景点、背景点、Enter/双击执行 |
| 框标注 | 已实现 | 支持地图拖框，后端修正像素框方向 |
| 文本标注 | 已实现接口 | 依赖 GroundingDINO / LangSAM / samgeo 环境 |
| 后处理 | 已实现 | 参数可在 Web 侧调整 |
| 矢量化导出 | 已实现 | 支持 GeoJSON、GPKG、SHP |
| 质量评估 | Python 包已实现 | Web 端暂未暴露完整评估界面 |

