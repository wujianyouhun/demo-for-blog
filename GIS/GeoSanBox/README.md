# GeoSandbox: 原子级 GIS 智能体执行沙盒

GeoSandbox 是一个专为地理空间分析智能体（GeoAgents）设计的动态执行环境。它通过隔离的 Docker 容器和标准化的原子工具集，使 LLM 能够安全地编写并执行空间分析任务。

## 核心特性

- **原子化工具 (Atomic Tools)**: 每个 GIS 操作（如投影转换、缓冲区、叠加分析、栅格计算）被封装为独立的 Python 函数，并带有标准 JSON 描述，方便 LLM 检索与调用
- **环境隔离 (Isolation)**: 基于 Docker 的沙箱，预装 GDAL、PROJ、Geopandas、Rasterio 等专业库，代码执行安全隔离
- **反馈驱动 (Feedback Loop)**: 捕获运行时错误（Runtime Errors）并返回给智能体，支持其进行自我修复
- **安全沙箱**: 代码安全审查，阻止危险操作（subprocess、os.system 等），Docker 资源限制（内存/CPU/超时）
- **多模式运行**: 支持 Docker 隔离模式、本地开发模式、HTTP API 服务模式

## 项目结构

```
GeoSanBox/
├── main.py                  # CLI 入口（list / run / info / serve）
├── config.py                # 全局配置（环境变量驱动）
├── requirements.txt         # Python 依赖
├── core/
│   ├── __init__.py
│   ├── executor.py          # 沙盒执行引擎（Docker + 本地双模式）
│   └── tool_manager.py      # 原子工具注册表（发现、校验、OpenAI Schema 导出）
├── tools/
│   ├── __init__.py
│   ├── vector_tools.py      # 矢量原子工具（16 个）
│   └── raster_tools.py      # 栅格原子工具（13 个）
├── docker/
│   └── Dockerfile           # GIS 运行环境镜像
└── README.md
```

## 快速开始

### 1. 构建沙箱环境

```bash
docker build -t geosandbox-env ./docker
```

### 2. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 3. 运行

```bash
# 查看环境信息
python main.py info

# 列出所有工具
python main.py list

# 导出 OpenAI function-calling 格式
python main.py list --schema

# 筛选矢量工具
python main.py list --category vector

# 直接执行代码
python main.py run -c "from tools.vector_tools import buffer_analysis; print(buffer_analysis('input.shp', 100, 'output.shp'))"

# 从文件执行代码
python main.py run -f my_analysis.py

# 本地模式（不用 Docker）
python main.py run -f my_analysis.py --local

# 启动 HTTP API 服务
python main.py serve --port 8090
```

## 原子工具清单

### 矢量工具 (vector) — 16 个

| 工具名 | 描述 |
|--------|------|
| `buffer_analysis` | 缓冲区分析 |
| `clip_vector` | 矢量裁剪 |
| `intersect_vector` | 交集运算 |
| `union_vector` | 并集运算 |
| `difference_vector` | 差集运算 |
| `dissolve_vector` | 按字段融合 |
| `reproject_vector` | 投影转换 |
| `spatial_join` | 空间连接 |
| `centroid` | 几何中心点 |
| `simplify_vector` | 几何简化 |
| `select_by_attribute` | 属性筛选 |
| `select_by_location` | 空间筛选 |
| `export_geojson` | 导出 GeoJSON |
| `get_vector_info` | 元数据信息 |
| `area_calculate` | 面积计算 |
| `distance_calculate` | 最近距离计算 |

### 栅格工具 (raster) — 13 个

| 工具名 | 描述 |
|--------|------|
| `clip_raster` | 栅格裁剪 |
| `reproject_raster` | 栅格重投影 |
| `resample_raster` | 分辨率调整 |
| `get_raster_info` | 栅格元数据 |
| `raster_calculator` | 波段代数运算 |
| `raster_to_vector` | 栅格转矢量 |
| `merge_rasters` | 栅格镶嵌 |
| `raster_statistics` | 波段统计 |
| `calculate_slope` | DEM 坡度计算 |
| `calculate_hillshade` | 山体阴影 |
| `extract_raster_values` | 采样提取 |
| `reclassify_raster` | 重分类 |

## HTTP API

启动 API 服务后，可用的端点：

```bash
GET  /health              # 健康检查
GET  /tools               # 列出工具（支持 ?category=vector&schema=openai）
POST /execute             # 执行 Python 代码 {"code": "..."}
POST /tools/<tool_name>   # 直接调用工具 {"input_path": "...", ...}
```

## 配置

通过环境变量覆盖默认配置：

```bash
export GEOSANDBOX_WORKSPACE=./data        # 工作空间路径
export GEOSANDBOX_IMAGE=geosandbox-env    # Docker 镜像名
export GEOSANDBOX_TIMEOUT=300             # 执行超时（秒）
export GEOSANDBOX_MEMORY=2g               # 内存限制
export GEOSANDBOX_CPU=2.0                 # CPU 限制
export GEOSANDBOX_USE_DOCKER=true         # 是否使用 Docker
```

## 智能体集成示例

```python
from core.executor import GeoExecutor, CodeResult
from core.tool_manager import registry, get_registry

# 初始化工具注册表
reg = get_registry()
reg.auto_discover(["tools.vector_tools", "tools.raster_tools"])

# 获取 OpenAI function-calling schema 传给 LLM
tools_schema = reg.to_openai_schema()

# LLM 生成代码后，在沙盒中执行
executor = GeoExecutor(workspace_path="./data")
result = executor.run_code(llm_generated_code)

# 检查结果
code_result = CodeResult(result)
if code_result.success:
    print(code_result.logs)
else:
    # 将错误反馈给 LLM 进行自我修复
    print(f"Error: {code_result.logs}")
```
