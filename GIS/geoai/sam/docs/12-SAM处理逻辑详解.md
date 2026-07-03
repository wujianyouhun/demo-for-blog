# 12 - SAM 处理逻辑详解

本文说明当前 GeoAI SAM Web 平台从 GeoTIFF 影像到 Mask、后处理和矢量导出的完整处理链路。

## 1. 总体流程

```text
用户选择 GeoTIFF
  -> 后端 ImageService 读取元数据
  -> 后端返回 tile_url / mask_extent / session_id
  -> 前端 OpenLayers 加载 TiTiler 瓦片
  -> 用户点击点、拖框或输入文本
  -> 前端提交 WGS84 经纬度
  -> 后端转换到影像像素坐标
  -> SAMService 判断是否需要裁剪大图
  -> 在原图或子图上执行 SAM 推理
  -> 归一化 Mask
  -> 生成前端 PNG 叠加层
  -> 可选后处理
  -> 可选矢量化导出

整幅影像处理是独立任务链路：

```text
用户启动整图任务
  -> 后端按 tile_size / overlap 遍历 GeoTIFF
  -> 每个瓦片写出临时 GeoTIFF 并保留 transform
  -> 对瓦片执行文本 / 自动 / 点 / 框推理
  -> 瓦片 Mask 后处理和矢量化
  -> 合并所有瓦片 GeoDataFrame
  -> 输出 GeoPackage / GeoJSON / Shapefile
```

## 2. 影像加载

后端入口：

```http
POST /api/image/load
```

核心类：

```python
ImageService(image_path).load()
```

读取信息：

| 字段 | 来源 | 用途 |
| --- | --- | --- |
| `width` / `height` | `rasterio.open()` | 原图尺寸和 Mask 画布大小 |
| `crs` | GeoTIFF CRS | 坐标转换 |
| `bounds` | GeoTIFF bounds | 页面显示元数据 |
| `transform` | 仿射变换 | 地理坐标转像素坐标 |
| `mask_extent` | CRS -> EPSG:3857 | 前端 fit 和 Mask 叠加 |

后端会把 TiTiler URL 返回给前端：

```text
/tiles/cog/tiles/WebMercatorQuad/{z}/{x}/{y}.png?url=<abs_path>
```

## 3. 前端地图定位

前端组件：

```text
frontend/src/components/MapCanvas.vue
```

加载影像时优先使用：

```js
info.mask_extent
```

`mask_extent` 已经是 EPSG:3857 范围，可以直接传给 OpenLayers `View.fit()` 和 `ImageStatic.imageExtent`。只有当影像 CRS 是 EPSG:4326 时，才退回使用 `fromLonLat(info.bounds)`。

这样可以避免把投影坐标误当经纬度导致点、框坐标错位。

## 4. 坐标转换

前端点击和拖框得到的是地图坐标 EPSG:3857。提交后端前使用：

```js
toLonLat(evt.coordinate)
```

所以 API 收到的是 WGS84 经纬度。

后端转换流程：

```python
if image_crs != "EPSG:4326":
    lon, lat = rasterio.warp.transform("EPSG:4326", image_crs, [lon], [lat])

col, row = ~transform * (lon, lat)
```

转换后会裁剪到有效像素范围：

```python
col = max(0, min(col, width - 1))
row = max(0, min(row, height - 1))
```

如果 CRS 或坐标异常，会返回 400：坐标转换失败。

## 5. 大影像自动裁剪

SAM 不适合直接处理超大 GeoTIFF，因此当前 `SAMService` 对大影像做自动裁剪。

关键参数：

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `_MAX_SAM_DIM` | 2048 | 送入 SAM 的最大边长 |
| `_CROP_PADDING` | 200 | 框提示裁剪时额外扩展像素 |
| 临时目录 | `output/tmp/` | 保存裁剪子影像 |

判断逻辑：

```python
max(width, height) > _MAX_SAM_DIM
```

### 5.1 点标注裁剪

点标注先计算所有提示点中心：

```python
cx = mean(point_x)
cy = mean(point_y)
```

然后以该中心裁剪最大 `2048 x 2048` 子图。点坐标会转换为局部坐标：

```python
local_point = [x - col_off, y - row_off]
```

### 5.2 框标注裁剪

框标注根据框中心和框大小裁剪：

```python
crop_w = min(box_width + padding * 2, 2048, image_width)
crop_h = min(box_height + padding * 2, 2048, image_height)
```

框坐标转局部坐标：

```python
local_box = [
    xmin - col_off,
    ymin - row_off,
    xmax - col_off,
    ymax - row_off,
]
```

### 5.3 crop_offset 的作用

推理在子图上完成后，Mask 只覆盖子图区域。后端保存：

```python
session["crop_offset"] = (col_off, row_off, crop_w, crop_h)
```

生成 PNG 时会把子图 Mask 放回全图降采样画布的正确位置，保证前端叠加不偏移。

## 6. 点标注处理逻辑

接口：

```http
POST /api/predict/point
```

输入：

```json
{
  "session_id": "...",
  "points": [[lon, lat]],
  "labels": [1]
}
```

处理步骤：

1. 校验点列表非空。
2. 校验 `points` 和 `labels` 数量一致。
3. WGS84 经纬度转像素坐标。
4. 大图自动裁剪。
5. 调用：

```python
self._sam.generate_masks_by_points(points=local_points, point_labels=labels)
```

6. 归一化 Mask。
7. 生成 PNG 并返回。

常见错误：

| 错误 | 说明 |
| --- | --- |
| 400 点标注至少需要 1 个提示点 | 没有添加点就执行 |
| 400 点坐标数量必须与标签数量一致 | API 参数不合法 |
| 503 缺少基础 SAM 依赖 | `torch` 或 `samgeo` 不可用 |
| 503 模型文件不完整 | 权重下载失败 |

## 7. 框标注处理逻辑

接口：

```http
POST /api/predict/box
```

输入：

```json
{
  "session_id": "...",
  "box": [lon1, lat1, lon2, lat2]
}
```

处理步骤：

1. 校验框包含 4 个数值。
2. 两个角点从 WGS84 转像素坐标。
3. 排序为 `[xmin, ymin, xmax, ymax]`。
4. 大图自动裁剪。
5. 调用：

```python
self._sam.generate_masks_by_box(box=local_box)
```

6. 生成 PNG 并返回。

框标注比点标注更稳定，因为它限定了目标搜索范围。

## 8. 文本标注处理逻辑

接口：

```http
POST /api/predict/text
```

输入：

```json
{
  "session_id": "...",
  "text": "building",
  "box_threshold": 0.25,
  "text_threshold": 0.25
}
```

处理步骤：

1. 检查基础 SAM 依赖和模型权重。
2. 初始化 `GroundedSAMWrapper`。
3. 加载当前影像。
4. 调用：

```python
self._gsam.segment_by_text(text, box_threshold=..., text_threshold=...)
```

5. 归一化 Mask。
6. 生成 PNG 并返回。

交互式文本标注用于快速预览；若要处理完整大图，应使用整幅影像处理中的“文本”模式，它会逐瓦片执行 Grounded-SAM 并合并结果。

## 9. Mask 归一化

不同底层库可能返回：

| 形状/类型 | 说明 |
| --- | --- |
| `(H, W)` | 单个 Mask |
| `(N, H, W)` | 多候选 Mask |
| `(1, N, H, W)` | 批量维度 + 多候选 |
| `list[dict]` | 自动分割候选，含 `segmentation` |
| `dict` | 包含 `mask` 或 `masks` |

`SAMService._normalize_masks()` 会统一转成非空 `np.ndarray`。空结果返回 400。

## 10. PNG 叠加生成

函数：

```python
_downsample_mask_png(mask, orig_h, orig_w, crop_offset=None)
```

作用：

1. 取第一个候选 Mask。
2. 按原图最大边 2048 降采样。
3. 若存在 `crop_offset`，先创建全图画布，再把子图 Mask 放回正确位置。
4. 生成绿色半透明 RGBA PNG。

前端使用 `ImageStatic` 将 PNG 叠加到 `imageExtent3857`。

## 11. 后处理逻辑

接口：

```http
POST /api/postprocess
```

调用：

```python
MaskPostProcessor.default_pipeline(
    mask,
    min_size=200,
    fill_holes_flag=True,
    smooth_sigma=1.5,
    opening_radius=2,
    closing_radius=3,
)
```

当前后处理针对 `session["last_mask"]` 执行。如果前一次分割使用了裁剪子图，后处理 PNG 也会使用同一个 `crop_offset` 放回全图位置。

## 12. 矢量导出逻辑

接口：

```http
POST /api/export/vectorize
```

当前导出使用：

```python
MaskVectorizer.vectorize(mask, reference_image=image_path, min_area=min_area)
```

输出目录：

```text
output/web_export/
```

支持格式：

| 格式 | 扩展名 |
| --- | --- |
| GeoJSON | `.geojson` |
| GeoPackage | `.gpkg` |
| Shapefile | `.shp` |

注意：如果当前 Mask 来自裁剪子图，矢量化阶段需要确保 Mask 与原图坐标一致。当前前端显示已通过 `crop_offset` 纠偏；后续若要严格导出原图坐标下的裁剪 Mask，应在导出前把裁剪 Mask 回填到原图尺寸。

## 13. 整幅影像瓦片处理逻辑

接口：

```http
POST /api/process/full
GET /api/process/status
GET /api/process/download
```

核心类：

```python
FullImageProcessor(...)
```

关键参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `tile_size` | 2048 | 每个瓦片最大边长 |
| `overlap` | 256 | 相邻瓦片重叠像素 |
| `min_area` | 50 | 矢量化过滤最小面积 |
| `output_format` | gpkg | 输出格式 |
| `postprocess` | true | 每个瓦片矢量化前是否后处理 |

瓦片遍历使用 stride：

```python
stride = tile_size - overlap
```

最后一列和最后一行会强制贴齐影像边界，避免边缘区域漏处理。

### 13.1 文本模式

文本模式会处理全部瓦片：

```python
mask = SAMService.predict_by_text(tile_path, text)
```

适合建筑物、水体、道路等同类目标的全图预提取。

### 13.2 自动模式

自动模式会处理全部瓦片：

```python
SAMWrapper(automatic=True).generate_masks_auto(...)
```

它不需要提示词，但结果更依赖 SAM 自动候选质量，通常需要后续筛选。

### 13.3 点模式

点模式只处理包含提示点的瓦片。点坐标从全图像素坐标转为瓦片局部坐标：

```python
local_point = [x - tile_col_off, y - tile_row_off]
```

单个点不能代表整幅影像所有同类目标；如果目标需要全图提取，优先使用文本模式。

### 13.4 框模式

框模式只处理与提示框相交的瓦片。框会裁剪为瓦片局部框：

```python
local_box = intersection(box, tile_window)
```

如果想处理一个大范围，可以在地图上拖出覆盖该范围的大框。

### 13.5 合并和输出

每个瓦片 Mask 使用瓦片 GeoTIFF 的 transform 直接矢量化，因此生成的 Polygon 已经位于原始影像坐标系中。合并时会：

1. 拼接所有瓦片 GeoDataFrame。
2. 移除空几何。
3. 按 WKB 去除完全重复几何。
4. 输出到 `output/full_image/<task_id>/full_image_result.*`。

Shapefile 下载时会自动打包同名 `.shp/.shx/.dbf/.prj/.cpg`。

## 14. 结果正确性判断

| 阶段 | 正确表现 |
| --- | --- |
| 影像加载 | 地图定位正确，底部显示 CRS 和尺寸 |
| 点标注 | Mask 覆盖前景点附近目标，不大量覆盖背景点 |
| 框标注 | Mask 位于拖框范围内或边缘附近 |
| 文本标注 | Mask 与提示词语义一致 |
| 后处理 | 小碎片减少，孔洞变少，边界更平滑 |
| 矢量导出 | 多边形数量合理，可在 GIS 软件中打开 |

