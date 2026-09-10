# Landsat 8 ResNet 分类格网生成流水线

将 Landsat 8 Collection 2 Level-2 的 ZIP 场景批量处理成 ResNet 分类格网图，输出每个切片的类别标签和置信度。

适用场景：研究区跨多个 Landsat 场景、跨多个 UTM 投影带时，自动完成镶嵌、裁剪、云掩膜、反射率转换和切片分类。

---

## 一、环境准备

### 1. 依赖安装

```bash
pip install -r requirements_landsat8.txt
```

依赖清单（[requirements_landsat8.txt](requirements_landsat8.txt)）：

| 包        | 最低版本 | 用途                       |
| --------- | -------- | -------------------------- |
| rasterio  | >= 1.3   | 栅格读写、镶嵌、裁剪、投影 |
| numpy     | >= 1.24  | 数组运算                   |
| torch     | >= 2.0   | 深度学习推理               |
| torchvision | >= 0.15 | ResNet 预训练模型         |

### 2. 本地分类模型（可选）

脚本默认会查找本地训练产物：

```
../../models/Classification/checkpoints/best_model.pth
```

- **存在 checkpoint**：加载项目自训练的 ResNet（支持 resnet50 / resnet101），类别为遥感地物类别。
- **不存在 checkpoint**：自动回退到 ImageNet 预训练 ResNet50，类别为 ImageNet 1000 类，**仅用于验证数据链路是否打通，分类结果无遥感含义**。

可通过 `--checkpoint` 指定其他 checkpoint 路径。

---

## 二、数据准备

### 1. 输入数据要求

- **产品类型**：Landsat 8 Collection 2 Level-2（表面反射率 SR）。
- **文件格式**：USGS 分发的 ZIP 包，解压后应包含以下波段（文件名后缀大小写不敏感）：
  - `*_SR_B2.TIF`（蓝）
  - `*_SR_B3.TIF`（绿）
  - `*_SR_B4.TIF`（红）
  - `*_QA_PIXEL.TIF`（质量掩膜）
- **存放方式**：所有 ZIP 放入同一目录，支持递归查找子目录。

### 2. 推荐数据源

- USGS EarthExplorer：https://earthexplorer.usgs.gov/
- Google Cloud Storage 公共数据集（Landsat C2 L2）

### 3. 研究区范围

默认 `--bbox` 为 WGS84 经纬度：`97.38 36.48 103.77 39.73`（青海北部，跨 2 个 UTM 带）。可按需修改。需保证已下载的 ZIP 场景与 bbox 有空间重叠，否则会报 "研究区与已下载的影像没有重叠"。

---

## 三、启动脚本

在 `geoai` 项目根目录下运行：

```bash
python resNet/landsat8/landsat8_pipeline.py \
    --input-zips D:/landsat_zip \
    --output-dir outputs/l8
```

### 最小示例（仅必填参数）

```bash
python resNet/landsat8/landsat8_pipeline.py --input-zips ./data/landsat_zip --output-dir ./outputs/l8
```

### 完整示例（带研究区与推理参数）

```bash
python resNet/landsat8/landsat8_pipeline.py \
    --input-zips D:/landsat_zip \
    --output-dir outputs/l8 \
    --bbox 97.38 36.48 103.77 39.73 \
    --target-crs EPSG:3857 \
    --target-resolution 30.0 \
    --tile-size 64 \
    --stride 64 \
    --batch-size 32 \
    --device auto
```

---

## 四、处理步骤

脚本按以下顺序依次执行，每步输出会写到 `--output-dir` 对应子目录：

| 步骤 | 输出目录                                   | 说明                                                                                       |
| ---- | ------------------------------------------ | ------------------------------------------------------------------------------------------ |
| 1    | `01_extracted/`                            | 解压所有 ZIP，已解压的不会重复解压。                                                       |
| 2    | `02_mosaic_clipped/`                       | 对 B2、B3、B4、QA_PIXEL 四个波段分别：统一投影到目标 CRS → 跨场景镶嵌 → 按 bbox 裁剪。    |
| 3    | `03_rgb_reflectance.tif`                   | DN → 表面反射率（`×0.0000275 - 0.2`），应用 QA_PIXEL 云掩膜，按 B4/B3/B2 组成 RGB float32。 |
| 4    | `04_inference/`                            | 滑动切片 → ResNet 批量推理 → 输出分类格网和置信度格网。                                     |

### 步骤 2 细节

- 研究区跨多个 UTM 带时，统一投影到 **EPSG:3857** 才能正确镶嵌不同投影的场景。
- 使用 `from_bounds` 将 WGS84 bbox 转到目标 CRS 后取窗口，整数化后与影像边界相交，防止越界。
- 若裁剪窗口宽高 ≤ 0，说明研究区与影像无重叠，会直接报错。

### 步骤 3 细节（QA_PIXEL 云掩膜）

`QA_PIXEL` 中以下位被置 1 的像元视为无效（见 [qa_invalid_mask](landsat8_pipeline.py#L128-L131)）：

| 位 | 含义       |
| -- | ---------- |
| 0  | 填充值     |
| 1  | 膨胀云     |
| 2  | 卷云       |
| 3  | 云         |
| 4  | 云影       |

无效像元在 RGB 中置为 NaN，后续切片会据此跳过。

### 步骤 4 细节（切片与推理）

- 切片大小默认 64 像素 × 64 像素，约 1.92 km × 1.92 km（30 m 分辨率下）。
- 若某切片内有效像元比例 < 80%（即云/NoData/边缘超过 20%），跳过不输出类别（格网中记为 -1）。
- 切片缩放到模型输入尺寸（默认 224），并按 ImageNet 均值/方差标准化。
- 输出格网：**1 个像素 = 1 个切片**，像元尺寸 = `stride × 原始 Landsat 像元大小`。

### 输出文件清单

```
<output-dir>/
├── 01_extracted/                              # 解压结果
├── 02_mosaic_clipped/
│   ├── landsat8_SR_B2_mosaic_clip.tif
│   ├── landsat8_SR_B3_mosaic_clip.tif
│   ├── landsat8_SR_B4_mosaic_clip.tif
│   └── landsat8_QA_PIXEL_mosaic_clip.tif
├── 03_rgb_reflectance.tif                     # B4/B3/B2 RGB 反射率
└── 04_inference/
    ├── classification_grid.tif               # 分类格网 int16，-1 为无效
    ├── confidence_grid.tif                   # 置信度格网 float32
    └── tile_predictions.json                 # 每个切片的逐条预测记录
```

---

## 五、参数说明

### 必填参数

| 参数            | 类型   | 说明                                       |
| --------------- | ------ | ------------------------------------------ |
| `--input-zips`  | Path   | 存放 Landsat ZIP 的目录；递归查找 `*.zip`。 |
| `--output-dir`  | Path   | 输出根目录，会自动创建子目录。             |

### 可选参数

| 参数                  | 默认值                              | 说明                                                                                         |
| --------------------- | ----------------------------------- | -------------------------------------------------------------------------------------------- |
| `--bbox`              | `97.38 36.48 103.77 39.73`          | 裁剪范围（WGS84 经纬度），顺序：`MIN_LON MIN_LAT MAX_LON MAX_LAT`。                          |
| `--checkpoint`        | `models/Classification/checkpoints/best_model.pth` | ResNet checkpoint 路径；不存在时回退 ImageNet ResNet50。                       |
| `--tile-size`         | 64                                  | 原始 Landsat 像素切片边长；64 像素约 1.92 km。                                              |
| `--stride`            | 等于 `--tile-size`                 | 滑窗步长；小于 tile-size 时切片重叠，大于时切片有间隙。默认无重叠。                          |
| `--batch-size`        | 32                                  | ResNet 推理批大小。                                                                          |
| `--target-crs`        | `EPSG:3857`                         | 镶嵌坐标系。研究区跨 UTM 带时务必保持默认，否则不同场景无法正确拼接。                        |
| `--target-resolution` | 30.0                                | 镶嵌网格分辨率（米），默认与 Landsat 多光谱 30 m 一致。                                      |
| `--device`            | `auto`                              | 推理设备：`auto`（自动选 CUDA/CPU）、`cpu`、`cuda`。                                         |

### 参数选择建议

- **研究区跨 UTM 带**（如默认西北研究区）：保持 `--target-crs EPSG:3857`，不要改成某一 UTM 投影。
- **想要更细的格网**：减小 `--tile-size`（如 32），但切片越小，类别代表性越弱。
- **想要重叠切片以提升边界覆盖**：设 `--stride` 小于 `--tile-size`（如 `--tile-size 64 --stride 32`）。
- **显存不足**：调小 `--batch-size`（如 8 或 16），或设 `--device cpu`。
- **无本地 checkpoint 仅跑流程**：可忽略 `--checkpoint`，脚本会自动用 ImageNet ResNet50 走通链路。

---

## 六、常见问题

1. **报错 "未在 ... 找到 ZIP 文件"**
   检查 `--input-zips` 路径是否正确，ZIP 后缀是否为小写 `.zip`。

2. **报错 "未找到 SR_B4"**
   确认下载的是 **Collection 2 Level-2** 产品（含 `*_SR_B4.TIF`），不是 Level-1（只有 `*_B4.TIF`）。

3. **报错 "研究区与已下载的影像没有重叠"**
   检查 `--bbox` 与 ZIP 场景覆盖范围是否匹配，场景是否已完整覆盖 bbox。

4. **报错 "研究区小于切片尺寸"**
   减小 `--tile-size`（如改成 16 或 32）。

5. **分类结果类别是猫狗飞机等**
   说明未找到本地 checkpoint，脚本回退到 ImageNet ResNet50。请确认 `--checkpoint` 路径或放置训练好的遥感模型。
