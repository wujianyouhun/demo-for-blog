# time_b `_aligned.tif` 数据生成指南

本文档详细说明 `time_b` 目录下 `_aligned.tif` 文件的产生过程、涉及的每一步处理以及对应的代码位置。

---

## 一、文件产出概览

下载完成后，`data/raw/` 目录结构如下：

```
data/raw/
├── time_a/
│   └── s2_116.20_39.75_A.tif              # 时相 A 影像（参考影像）
└── time_b/
    ├── s2_116.20_39.75_B.tif               # 时相 B 原始影像（未对齐）
    └── s2_116.20_39.75_B_aligned.tif       # 时相 B 对齐影像 ← 最终使用此文件
```

- `s2_*_B.tif`：时相 B 的原始下载影像，保留其自身的 CRS、分辨率和像素尺寸。
- `s2_*_B_aligned.tif`：以时相 A 为参考，经过空间重投影对齐后的影像，**所有下游任务（样本制作、训练、推理）必须使用此文件**。

---

## 二、完整处理流程

`_aligned.tif` 的生成是双时相数据下载流程的一部分，整体分为 4 个阶段：

```
┌─────────────────────────────────────────────────────────────┐
│  1. STAC 搜索    →  查找满足条件的 Sentinel-2 影像          │
│  2. 下载时相 A   →  裁剪 + 归一化 → s2_*_A.tif             │
│  3. 下载时相 B   →  裁剪 + 归一化 → s2_*_B.tif             │
│  4. 空间对齐     →  重投影 B → A 网格 → s2_*_B_aligned.tif  │
└─────────────────────────────────────────────────────────────┘
```

### 阶段 1：STAC 搜索

**目的**：在 Planetary Computer 上搜索指定区域、日期范围内云量最低的 Sentinel-2 L2A 影像。

**处理步骤**：

1. 根据输入日期（如 `2022-06-01`）构造一个月的搜索时间范围（`2022-06-01/2022-07-01`）
2. 调用 `geoai.pc_stac_search` 搜索 `sentinel-2-l2a` 集合
3. 按云量升序排序，选取云量最低的影像
4. 使用 `planetary_computer.sign()` 对资产 URL 签名，获取可访问的下载链接

**关键代码**：[downloader.py](file:///f:/study/demo-for-blog/GIS/geoai/change-detection/cdd/downloader.py#L48-L76) → `_download_single` 方法中的搜索部分

```python
items = geoai.pc_stac_search(
    collection="sentinel-2-l2a",
    bbox=bbox,
    time_range=f"{date}/{date_end}",
    query={"eo:cloud_cover": {"lt": max_cloud}},
    max_items=10,
    quiet=True,
    endpoint=self.STAC_API_URL,
)
items = sorted(items, key=lambda it: it.properties.get("eo:cloud_cover", 100))
best = items[0]
signed = pc.sign(best)
```

### 阶段 2：下载时相 A 影像

**目的**：从 Sentinel-2 原始场景中裁剪出目标区域，生成时相 A 的参考影像。

**处理步骤**：

1. **打开远程影像**：通过签名后的 URL 用 rasterio 打开 Sentinel-2 资产
2. **坐标变换**：将用户提供的 WGS84 bbox 变换到影像的 CRS（通常是 UTM）
3. **窗口裁剪**：根据变换后的 bbox 计算 rasterio 窗口，取整数窗口并裁剪到影像有效范围
4. **逐波段读取**：按 B04（红）、B03（绿）、B02（蓝）顺序读取裁剪窗口内的数据
5. **值域归一化**：将 Sentinel-2 原始 DN 值（0~10000+）归一化到 uint8 范围：
   - 除以 3000 后乘 255
   - `clip(0, 255)` 截断
   - 转为 `uint8` 类型
6. **写入 GeoTIFF**：保存为 LZW 压缩的 GeoTIFF，保留 CRS 和 Transform 信息

**关键代码**：[downloader.py](file:///f:/study/demo-for-blog/GIS/geoai/change-detection/cdd/downloader.py#L87-L135) → `_download_single` 方法中的读取和写入部分

```python
# 坐标变换
img_bounds = transform_bounds("EPSG:4326", src.crs, *bbox)
window = from_bounds(*img_bounds, transform=src.transform)

# 窗口取整
win = window.round_offsets().round_lengths()
col_off = max(0, int(win.col_off))
row_off = max(0, int(win.row_off))
width = min(int(win.width), src.width - col_off)
height = min(int(win.height), src.height - row_off)
crop_window = rasterio.windows.Window(col_off, row_off, width, height)

# 值域归一化
stack = np.clip(np.stack(band_arrays) / 3000.0 * 255, 0, 255).astype(np.uint8)
```

### 阶段 3：下载时相 B 影像

**目的**：与时相 A 相同的流程，下载时相 B 的原始影像。

**处理步骤**：与阶段 2 完全一致，只是日期不同。注意：由于两幅影像可能来自不同的 Sentinel-2 轨道，它们的 CRS、分辨率、像素尺寸可能不同，这就是需要对齐的原因。

**关键代码**：同阶段 2，复用 `_download_single` 方法。

### 阶段 4：空间对齐（生成 `_aligned.tif`）

**目的**：将时相 B 的原始影像重投影到时相 A 的空间网格上，使两幅影像像素级对齐。

**处理步骤**：

1. **读取参考网格**：打开时相 A 影像，获取其 `transform`（仿射变换矩阵）、`crs`（坐标系）、`width` 和 `height`
2. **创建目标数组**：以时相 A 的空间维度（width × height）创建零值数组，波段数与时相 B 一致
3. **逐波段重投影**：对时相 B 的每个波段，使用 `rasterio.warp.reproject` 将其从原始网格重采样到时相 A 的网格：
   - 源：时相 B 的 transform 和 CRS
   - 目标：时相 A 的 transform 和 CRS
   - 重采样方法：**双线性插值**（`Resampling.bilinear`）
4. **写入对齐文件**：以时相 A 的空间参考信息写入新的 GeoTIFF，文件名在原始文件名后追加 `_aligned` 后缀

**关键代码**：[downloader.py](file:///f:/study/demo-for-blog/GIS/geoai/change-detection/cdd/downloader.py#L137-L157) → `_align_pair` 方法

```python
def _align_pair(self, ref_path, target_path):
    out_path = target_path.parent / f"{target_path.stem}_aligned.tif"
    if out_path.exists():
        return out_path

    with rasterio.open(ref_path) as ref:
        rt, rc, rw, rh = ref.transform, ref.crs, ref.width, ref.height

    with rasterio.open(target_path) as src:
        aligned = np.zeros((src.count, rh, rw), dtype=src.dtypes[0])
        for i in range(src.count):
            reproject(
                source=rasterio.band(src, i + 1),
                destination=aligned[i],
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=rt, dst_crs=rc,
                resampling=Resampling.bilinear,
            )

    profile = {
        "driver": "GTiff", "dtype": "uint8",
        "width": rw, "height": rh,
        "count": aligned.shape[0],
        "crs": rc, "transform": rt, "compress": "lzw",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(aligned)
```

---

## 三、对齐前后的变化对比

| 属性 | 对齐前（`_B.tif`） | 对齐后（`_B_aligned.tif`） |
|------|-------------------|--------------------------|
| CRS | 可能与 A 不同 | 与 A 完全一致 |
| Transform | 可能与 A 不同 | 与 A 完全一致 |
| 像素尺寸 | 可能与 A 不同 | 与 A 完全一致 |
| 波段数 | 3（RGB） | 3（RGB） |
| 数据类型 | uint8 | uint8 |
| 像素值 | 原始值 | 双线性插值重采样后的值 |
| 地理范围 | 原始范围 | 与 A 相同范围 |

**为什么需要对齐**：Sentinel-2 的不同轨道影像可能使用不同的 UTM 分区（如 EPSG:32650 vs EPSG:32651），即使同一轨道不同日期的影像，其像素网格也可能存在微小偏移。只有像素级对齐后，才能正确计算同一位置的变化。

---

## 四、调用入口

`_aligned.tif` 的生成有两个调用入口：

### 入口 1：命令行脚本

**文件**：[scripts/download_data.py](file:///f:/study/demo-for-blog/GIS/geoai/change-detection/scripts/download_data.py)

```bash
python scripts/download_data.py \
  --region beijing \
  --date-a 2022-06-01 \
  --date-b 2023-06-01 \
  --cloud 20
```

调用链：

```
scripts/download_data.py
  → BiTemporalDownloader.download_pair()        # cdd/downloader.py:30
    → _download_single() → time_a/A.tif        # cdd/downloader.py:48
    → _download_single() → time_b/B.tif        # cdd/downloader.py:48
    → _align_pair()     → time_b/B_aligned.tif # cdd/downloader.py:137
```

### 入口 2：Web API

**文件**：[backend/routers/data.py](file:///f:/study/demo-for-blog/GIS/geoai/change-detection/backend/routers/data.py#L25-L79)

```
POST /api/data/download-pair
  → _download_single() → time_a/A.tif
  → _download_single() → time_b/B.tif
  → _align_pair()     → time_b/B_aligned.tif
```

Web API 将下载过程放入后台任务执行，通过 `GET /api/data/download-pair/{task_id}` 查询进度。

---

## 五、核心代码文件索引

| 文件 | 职责 | 关键方法 |
|------|------|---------|
| [cdd/downloader.py](file:///f:/study/demo-for-blog/GIS/geoai/change-detection/cdd/downloader.py) | 数据下载与对齐核心逻辑 | `download_pair()`、`_download_single()`、`_align_pair()` |
| [scripts/download_data.py](file:///f:/study/demo-for-blog/GIS/geoai/change-detection/scripts/download_data.py) | 命令行下载入口 | `main()` |
| [backend/routers/data.py](file:///f:/study/demo-for-blog/GIS/geoai/change-detection/backend/routers/data.py) | Web API 下载入口 | `download_pair()` |
| [config.py](file:///f:/study/demo-for-blog/GIS/geoai/change-detection/config.py) | 全局配置（路径、区域、波段） | `RAW_DIR`、`PRESET_REGIONS`、`S2_BANDS` |

---

## 六、依赖库说明

| 库 | 用途 |
|----|------|
| `rasterio` | 读写 GeoTIFF、窗口裁剪、重投影 |
| `rasterio.warp.reproject` | 核心对齐操作：将影像从源网格重采样到目标网格 |
| `rasterio.warp.transform_bounds` | 坐标系间的范围变换（WGS84 → UTM） |
| `rasterio.windows.from_bounds` | 根据地理范围计算像素窗口 |
| `geoai` | 封装 Planetary Computer STAC 搜索（`pc_stac_search`） |
| `planetary_computer` | STAC 资产 URL 签名（`pc.sign()`） |
| `numpy` | 数组操作、值域归一化 |

---

## 七、手动对齐自有影像

如果使用自有影像数据（非 Sentinel-2），且两幅影像未对齐，可手动执行对齐：

```python
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

ref_path = "data/raw/time_a/your_image_A.tif"
src_path = "data/raw/time_b/your_image_B.tif"
out_path = "data/raw/time_b/your_image_B_aligned.tif"

# 读取参考影像的空间信息
with rasterio.open(ref_path) as ref:
    rt, rc, rw, rh = ref.transform, ref.crs, ref.width, ref.height

# 将 B 重投影到 A 的网格
with rasterio.open(src_path) as src:
    aligned = np.zeros((src.count, rh, rw), dtype=src.dtypes[0])
    for i in range(src.count):
        reproject(
            source=rasterio.band(src, i + 1),
            destination=aligned[i],
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=rt, dst_crs=rc,
            resampling=Resampling.bilinear,
        )

# 写入对齐结果
profile = {
    "driver": "GTiff", "dtype": "uint8",
    "width": rw, "height": rh,
    "count": aligned.shape[0],
    "crs": rc, "transform": rt, "compress": "lzw",
}
with rasterio.open(out_path, "w", **profile) as dst:
    dst.write(aligned)
```

也可使用 GDAL 命令行工具：

```bash
gdalwarp -t_srs EPSG:32650 -tr 10 10 -te 449760 4414080 463200 4427520 \
  image_B.tif image_B_aligned.tif
```

参数说明：
- `-t_srs`：目标坐标系（与时相 A 一致）
- `-tr`：目标分辨率（米，与时相 A 一致）
- `-te`：目标范围（与时相 A 的 bounds 一致）

---

## 八、验证对齐结果

```python
import rasterio

with rasterio.open("data/raw/time_a/s2_116.20_39.75_A.tif") as a, \
     rasterio.open("data/raw/time_b/s2_116.20_39.75_B_aligned.tif") as b:
    print(f"A: CRS={a.crs}, shape={a.shape}, bounds={a.bounds}")
    print(f"B: CRS={b.crs}, shape={b.shape}, bounds={b.bounds}")

    if a.crs == b.crs and a.shape == b.shape and a.bounds == b.bounds:
        print("对齐验证通过：两幅影像空间参考完全一致")
    else:
        print("对齐验证失败：空间参考不一致")
```

对齐成功的标志：
- CRS 完全相同
- 像素尺寸（width × height）完全相同
- 地理范围（bounds）完全相同
- Transform 矩阵完全相同
