# 从 GeoTIFF 到浏览器地图：一个 GeoAI SAM Web 标注平台的工程架构

## 导语

很多 SAM 示例都是本地脚本：读一张图片，点一下，输出一个 Mask。

但遥感标注平台要复杂得多。用户面对的是 GeoTIFF，不是普通 PNG；影像有 CRS、bounds、transform；浏览器不能一次性加载几万像素的大图；前端点击的是地图坐标，后端推理需要像素坐标；结果还要叠加回地图并导出矢量。

本文拆解一个 GeoAI SAM Web 标注平台的工程架构，看看从 GeoTIFF 到浏览器交互标注，中间到底发生了什么。

## 整体架构

平台由三部分组成：

```text
Vue3 + OpenLayers 前端
  -> FastAPI 后端 API
  -> TiTiler 动态瓦片服务
  -> geoai_sam 模型与 GIS 处理模块
```

核心链路如下：

```text
用户选择 GeoTIFF
  -> 后端读取影像元数据
  -> 返回瓦片 URL、影像范围、会话 ID
  -> 前端用 OpenLayers 加载瓦片
  -> 用户点选、拖框或输入文本
  -> 前端提交经纬度
  -> 后端转为影像像素坐标
  -> 执行 SAM 推理
  -> 返回 PNG Mask 叠加层
  -> 用户后处理、矢量导出
```

## 为什么需要 TiTiler

遥感影像经常是上万甚至数万像素。直接把整张图传给浏览器，不现实。

TiTiler 的作用是把 GeoTIFF 变成动态瓦片服务：

```text
/tiles/cog/tiles/WebMercatorQuad/{z}/{x}/{y}.png?url=<image_path>
```

浏览器只请求当前视口需要的瓦片。用户缩放或平移地图时，再按需加载新瓦片。

这样做有三个好处：

1. 大影像可以快速预览。
2. 前端内存压力低。
3. 地图交互体验接近在线底图。

## 会话设计

后端维护一个会话字典：

```text
session_id
  -> image_service
  -> sam_service
  -> image_path
  -> last_mask
  -> processed_mask
  -> crop_offset
```

用户加载影像时创建会话。后续点标注、框标注、后处理、导出都带着 `session_id` 调用接口。

这能避免每次请求都重新读影像和初始化服务，也方便在同一张影像上连续交互。

## 坐标转换是核心细节

前端地图使用 OpenLayers，点击得到的是 Web Mercator 坐标。提交给后端前转换为 WGS84 经纬度：

```js
toLonLat(evt.coordinate)
```

后端拿到经纬度后，再按影像 CRS 转换：

```python
if image_crs != "EPSG:4326":
    lon, lat = rasterio.warp.transform("EPSG:4326", image_crs, [lon], [lat])

col, row = ~transform * (lon, lat)
```

最终得到的是影像像素坐标，也就是 SAM 真正需要的输入。

如果这个环节错了，表现会很明显：

- 点在目标上，但 Mask 出现在别处。
- 框选区域和实际裁剪区域不一致。
- Mask 叠加到地图时发生偏移。

因此平台专门返回 `mask_extent`，让前端用 EPSG:3857 范围定位影像和叠加 Mask。

## 大影像不能直接进 SAM

Web 地图可以显示大图，但 SAM 推理不能直接吃几万像素的 GeoTIFF。

平台在普通交互标注中做了自动裁剪：

- 点标注：以提示点中心裁剪局部子图。
- 框标注：以框中心和框大小裁剪局部子图。
- 裁剪最大边长默认控制在 2048。

推理完成后，后端会记录：

```python
crop_offset = (col_off, row_off, crop_w, crop_h)
```

生成前端 PNG 叠加层时，再把子图 Mask 放回全图降采样画布的正确位置。

这就是为什么用户感觉是在整张地图上标注，但模型实际只处理局部区域。

## 前后端 API 分层

平台主要 API 包括：

| API | 作用 |
| --- | --- |
| `POST /api/image/load` | 加载影像，创建会话 |
| `POST /api/predict/point` | 点提示分割 |
| `POST /api/predict/box` | 框提示分割 |
| `POST /api/predict/text` | 文本提示分割 |
| `POST /api/postprocess` | Mask 后处理 |
| `POST /api/export/vectorize` | 矢量化导出 |
| `POST /api/process/full` | 整幅影像瓦片化处理 |
| `GET /api/process/status` | 查询整图任务进度 |

普通交互标注返回 PNG Mask；整图处理返回任务 ID，再通过轮询查询进度。

## 前端为什么要代理 API

开发环境中，前端运行在：

```text
http://127.0.0.1:5173
```

后端运行在：

```text
http://127.0.0.1:8001
```

Vite 代理负责把 `/api` 和 `/tiles` 转发到后端：

```js
proxy: {
  "/api": { target: "http://127.0.0.1:8001" },
  "/tiles": { target: "http://127.0.0.1:8001" }
}
```

如果页面能打开但不能处理，第一步不是怀疑模型，而是访问：

```text
http://127.0.0.1:5173/api/config
```

它必须返回 JSON。若是 404 或连接失败，说明前端没有正确代理到后端。

## 小结

一个可用的 GeoAI SAM Web 平台，本质上是四类能力的组合：

- WebGIS 显示能力：TiTiler + OpenLayers。
- 遥感坐标能力：CRS、transform、extent。
- 模型推理能力：SAM、GroundingDINO、LangSAM。
- 标签生产能力：Mask 后处理和矢量导出。

当这些模块边界清晰后，平台就不再只是一个模型 demo，而是一个可以持续扩展的遥感标注工程。

