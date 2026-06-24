# 训练样本制作指南

变化检测模型训练需要成对的双时相影像和对应的变化标签。本项目提供 **三种样本制作模式**，适用于不同场景和精度需求。

## 模式总览

| 模式 | 标签来源 | 是否需要真实数据 | 适用场景 | 标签精度 |
|------|---------|:---------------:|---------|:-------:|
| `synthetic` | 程序随机生成 | 否 | 快速验证训练流程、学习与调试 | 低（模拟数据） |
| `weak-label` | GeoAI ChangeStar 预测 | 需要双时相影像 | 利用真实影像跑通训练、半监督学习 | 中（模型预测） |
| `vector-label` | 人工标注矢量 | 需要双时相影像 + 矢量标签 | 正式监督训练、生产部署 | 高（人工标注） |

---

## 模式一：synthetic（模拟样本）

### 原理

通过程序随机生成模拟的双时相遥感影像和精确的变化标签，无需任何真实数据。

生成过程：
1. 随机生成基础影像（3 通道 RGB，像素值 50~200），模拟地物背景
2. 在基础影像上叠加 3~8 个随机矩形色块，模拟建筑、道路等地物
3. 复制基础影像作为时相 B，在 B 上叠加 1~5 个新矩形色块，模拟新增建筑等变化
4. 同时在时相 A 上也可能叠加矩形（模拟拆除），变化位置以 50% 概率出现在 A 或 B
5. 生成精确的二值标签：变化区域像素为 1，其余为 0
6. 对两时相影像添加高斯噪声（σ=5），模拟传感器差异
7. 写入 GeoTIFF（EPSG:4326，LZW 压缩），保留地理参考信息

### 使用命令

```bash
python scripts/generate_sample.py \
  --mode synthetic \
  --num-samples 100 \
  --tile-size 256
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--num-samples` | 100 | 生成样本对数量 |
| `--tile-size` | 256 | 每张样本的边长（像素） |
| `--keep-existing` | False | 追加样本，不清空已有 .tif 文件 |

### 输出结构

```
data/samples/
├── time_a/sample_0000.tif   # 时相 A 影像 (3, 256, 256)
├── time_a/sample_0001.tif
├── ...
├── time_b/sample_0000.tif   # 时相 B 影像 (3, 256, 256)
├── time_b/sample_0001.tif
├── ...
├── labels/sample_0000.tif   # 二值变化标签 (1, 256, 256)，0=未变，1=变化
├── labels/sample_0001.tif
└── ...
```

### 适用场景

- **快速验证**：无需下载数据即可跑通完整的训练→推理→评估流程
- **学习调试**：理解数据加载、模型结构、训练循环的交互方式
- **单元测试**：为自动化测试提供可复现的固定数据集（seed=42）

### 局限性

- 模拟影像与真实遥感影像差异大（无纹理、无光谱特征）
- 变化区域均为规则矩形，无法模拟真实地物边界
- 训练出的模型在真实数据上泛化能力有限

---

## 模式二：weak-label（弱标签样本）

### 原理

利用 GeoAI ChangeStar 预训练模型对真实双时相影像进行变化检测，将预测结果作为"弱标签"，然后按滑窗方式从原始影像中裁剪训练样本。

生成过程：
1. 调用 GeoAI ChangeStar 模型（默认 `s1_s1c1_vitb`）对整幅双时相影像执行变化检测
2. 输出变化概率掩膜（GeoTIFF），像素值表示变化概率
3. 按指定 `tile_size` 和 `stride` 对原始影像和变化掩膜进行滑窗裁剪
4. 过滤掉变化像素少于 `min_change_pixels` 的瓦片（保留有意义的正样本）
5. 将变化掩膜二值化（>0 为变化）作为训练标签

### 使用命令

```bash
python scripts/generate_sample.py \
  --mode weak-label \
  --image-a data/raw/time_a/s2_beijing_A.tif \
  --image-b data/raw/time_b/s2_beijing_B_aligned.tif \
  --tile-size 256 \
  --stride 256 \
  --min-change-pixels 20 \
  --geoai-model s1_s1c1_vitb \
  --threshold 0.5
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--image-a` | 必填 | 时相 A GeoTIFF 影像路径 |
| `--image-b` | 必填 | 时相 B GeoTIFF 影像路径 |
| `--tile-size` | 256 | 裁剪瓦片边长（像素） |
| `--stride` | 256 | 滑窗步长，小于 tile_size 时产生重叠瓦片 |
| `--min-change-pixels` | 20 | 最少变化像素数，低于此值的瓦片被丢弃 |
| `--max-tiles` | 无限制 | 最大瓦片数量，用于控制样本总量 |
| `--geoai-model` | s1_s1c1_vitb | GeoAI ChangeStar 模型名称 |
| `--threshold` | 0.5 | 变化检测阈值（0~1） |
| `--keep-existing` | False | 追加样本，不清空已有 .tif 文件 |

### 输出结构

```
data/samples/
├── time_a/weak_0000.tif   # 时相 A 瓦片
├── time_b/weak_0000.tif   # 时相 B 瓦片
├── labels/weak_0000.tif   # 弱标签（ChangeStar 预测结果二值化）
└── ...
```

### 适用场景

- **真实影像训练**：使用真实遥感影像训练自定义模型，验证模型在真实数据上的表现
- **半监督学习**：以弱标签为起点，结合少量人工标注进行迭代优化
- **数据增强**：在大范围影像上自动生成大量训练样本，减少人工标注成本

### 注意事项

- 首次运行会下载 GeoAI ChangeStar 预训练模型（约 300MB），缓存到 `data/pretrained/`
- 弱标签质量取决于 ChangeStar 模型精度，可能存在漏检和误检
- 建议先用 `--threshold` 调整检测灵敏度，再批量生成样本
- GeoAI 推理时使用 `tile_size=max(512, tile_size)` 以保证推理质量

---

## 模式三：vector-label（矢量标签样本）

### 原理

使用人工标注的变化矢量数据（GeoJSON/GPKG/SHP）作为真值标签，栅格化后与双时相影像配准裁剪，生成高质量的监督训练样本。

生成过程：
1. 读取矢量标签文件（GeoJSON/GPKG/SHP），自动重投影到与影像一致的 CRS
2. 使用 `rasterio.features.rasterize` 将矢量多边形栅格化为二值掩膜（`all_touched=True`，多边形覆盖的像素均标记为变化）
3. 将栅格化掩膜写入临时 GeoTIFF
4. 按指定 `tile_size` 和 `stride` 对双时相影像和栅格化标签进行滑窗裁剪
5. 过滤掉变化像素少于 `min_change_pixels` 的瓦片
6. 输出配对的训练样本

### 使用命令

```bash
python scripts/generate_sample.py \
  --mode vector-label \
  --image-a data/raw/time_a/s2_beijing_A.tif \
  --image-b data/raw/time_b/s2_beijing_B_aligned.tif \
  --vector-label labels/change.geojson \
  --tile-size 256 \
  --stride 256 \
  --min-change-pixels 20
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--image-a` | 必填 | 时相 A GeoTIFF 影像路径 |
| `--image-b` | 必填 | 时相 B GeoTIFF 影像路径 |
| `--vector-label` | 必填 | 变化矢量标签文件路径（支持 .geojson/.gpkg/.shp） |
| `--tile-size` | 256 | 裁剪瓦片边长（像素） |
| `--stride` | 256 | 滑窗步长，小于 tile_size 时产生重叠瓦片 |
| `--min-change-pixels` | 20 | 最少变化像素数，低于此值的瓦片被丢弃 |
| `--max-tiles` | 无限制 | 最大瓦片数量，用于控制样本总量 |
| `--keep-existing` | False | 追加样本，不清空已有 .tif 文件 |

### 输出结构

```
data/samples/
├── time_a/vector_0000.tif   # 时相 A 瓦片
├── time_b/vector_0000.tif   # 时相 B 瓦片
├── labels/vector_0000.tif   # 人工标注标签（矢量栅格化后的二值掩膜）
└── ...
```

### 矢量标签格式要求

- **格式**：GeoJSON（推荐）、GeoPackage、Shapefile
- **几何类型**：Polygon 或 MultiPolygon，每个多边形表示一处变化区域
- **坐标系**：任意 CRS，程序会自动重投影到影像坐标系
- **属性**：无需特定属性字段，仅使用几何信息

示例 GeoJSON 结构：

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[116.35, 39.85], [116.36, 39.85], [116.36, 39.86], [116.35, 39.86], [116.35, 39.85]]]
      },
      "properties": {}
    }
  ]
}
```

### 适用场景

- **正式监督训练**：使用高质量人工标注数据训练模型，追求最佳检测精度
- **生产部署**：模型上线前的最终训练，确保标签可靠性
- **基准评测**：作为 ground truth 评估模型性能

### 注意事项

- 矢量标签需与影像空间对齐（同一区域），坐标系不一致时会自动转换
- `all_touched=True` 意味着多边形边界经过的像素也会被标记为变化，可能产生略粗的标签边界
- 栅格化后的标签掩膜保存在 `data/output/` 目录下，文件名为 `{矢量文件名}_rasterized_label.tif`

---

## 通用说明

### 样本目录结构

无论哪种模式，最终产出的样本目录结构一致：

```
data/samples/
├── time_a/    # 时相 A 影像瓦片 (3 通道, uint8, GeoTIFF)
├── time_b/    # 时相 B 影像瓦片 (3 通道, uint8, GeoTIFF)
└── labels/    # 二值变化标签 (1 通道, uint8, 0=未变/1=变化, GeoTIFF)
```

三个目录中的 `.tif` 文件数量必须一致，且至少 2 对样本才能启动训练。

### 追加样本

默认每次生成会清空已有样本。使用 `--keep-existing` 可以追加样本而不覆盖：

```bash
# 先生成弱标签样本
python scripts/generate_sample.py --mode weak-label --image-a A.tif --image-b B.tif

# 追加矢量标签样本（不清空弱标签样本）
python scripts/generate_sample.py --mode vector-label --image-a A.tif --image-b B.tif --vector-label label.geojson --keep-existing
```

### 滑窗裁剪策略（weak-label / vector-label 共用）

两种真实影像模式共用同一套滑窗裁剪逻辑（`_tile_from_mask` 函数）：

```
原始影像 (W × H)
┌──────────────────────┐
│  ┌─────┐              │
│  │tile0│  ← stride →  │
│  └─────┘              │
│       ┌─────┐         │
│       │tile1│         │
│       └─────┘         │
│  ...                  │
└──────────────────────┘

stride < tile_size → 瓦片间有重叠，增加样本量
stride = tile_size → 无重叠，样本不重复
```

- 当 `stride < tile_size` 时，相邻瓦片存在重叠区域，可增加训练样本数量
- 边缘位置自动 clamp，确保覆盖影像全部区域
- `min_change_pixels` 过滤掉变化极少的瓦片，避免正负样本严重失衡

### 从样本到训练

样本生成后即可启动训练：

```bash
python scripts/train.py --model siamese_unet --epochs 30 --batch-size 4
```

训练脚本自动从 `data/samples/` 读取样本，按 85%/15% 划分训练/验证集，支持数据增强和混合精度训练。
