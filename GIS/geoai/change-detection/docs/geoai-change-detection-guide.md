# GeoAI 变化检测学习指南

## 这个项目如何使用 GeoAI

本项目把 GeoAI 放在主流程中：

1. `geoai.pc_stac_search` 搜索 Planetary Computer Sentinel-2 L2A 数据。
2. 本项目用 rasterio 按 bbox 窗口裁剪读取影像，避免下载整幅大图。
3. `geoai.changestar_detect` 使用预训练 ChangeStar 执行建筑变化检测。
4. `geoai` 生成的变化结果可进一步作为弱标签，切片成训练样本。

现有 `cdd` 模块保留自训练 Siamese U-Net / BiT 的教学流程。

## 下载影像不等于有训练样本

下载双时相影像只得到：

```text
T1 影像: data/raw/time_a/*.tif
T2 影像: data/raw/time_b/*.tif
```

训练监督变化检测模型还需要标签：

```text
label: data/samples/labels/*.tif
```

标签表示哪些像素发生了变化。没有标签时，训练会提示样本不足，这是正确行为。解决方式有三种：

- `synthetic`：生成模拟样本，学习训练流程。
- `weak-label`：用 GeoAI ChangeStar 先预测变化，再切片为弱标签样本。
- `vector-label`：使用人工或权威变化矢量作为真实监督标签。

## 推荐学习路径

### 路径 A：最快看到结果

```bash
python scripts/download_data.py --region beijing --date-a 2022-06-01 --date-b 2023-06-01
python scripts/predict.py --engine geoai --image-a data/raw/time_a/xxx_A.tif --image-b data/raw/time_b/xxx_B_aligned.tif --visualize
```

这条路径不训练模型，适合理解 GeoAI 变化检测效果。

### 路径 B：学习训练流程

```bash
python scripts/generate_sample.py --mode synthetic --num-samples 100
python scripts/train.py --model siamese_unet --epochs 30 --batch-size 4
python scripts/predict.py --engine cdd --image-a data/raw/time_a/xxx_A.tif --image-b data/raw/time_b/xxx_B_aligned.tif --model data/models/best_model.pth
```

这条路径适合理解数据集、训练循环和滑窗推理。

### 路径 C：真实影像弱标签训练

```bash
python scripts/generate_sample.py --mode weak-label --image-a data/raw/time_a/xxx_A.tif --image-b data/raw/time_b/xxx_B_aligned.tif
python scripts/train.py --model siamese_unet --epochs 30 --batch-size 4
```

弱标签不是人工真值，适合演示和迁移学习起步，不适合直接作为严谨评价依据。

## Web 使用流程

1. 打开“数据下载”，选择区域和两个日期，下载双时相影像。
2. 打开“变化检测”，默认引擎为 GeoAI，选择影像后直接执行变化检测。
3. 切到“对比分析”，查看变化图、统计和矢量结果。
4. 如需训练，先在“样本制作”中生成样本，再执行“训练模型”。
5. 如需使用自训练模型检测，把引擎切换为“自训练”，选择 `.pth` 权重。

## 常见问题

### 提示样本不足

训练需要 `time_a`、`time_b`、`labels` 三个目录数量一致，且至少 2 对样本。先运行：

```bash
python scripts/generate_sample.py --mode synthetic
```

### 执行变化检测提示缺少模型参数

GeoAI 模式不需要模型权重：

```bash
python scripts/predict.py --engine geoai --image-a ... --image-b ...
```

只有自训练模式才需要：

```bash
python scripts/predict.py --engine cdd --model data/models/best_model.pth --image-a ... --image-b ...
```

### 首次运行很慢或下载模型

GeoAI ChangeStar 首次运行可能下载预训练权重。请保持网络可用，并确认 `data/pretrained` 可写。

### 没有 GPU 能不能跑

可以。CPU 会慢很多，建议先用小 bbox 或较小影像验证流程。

### 结果为空

常见原因包括两个日期影像差异不明显、云/阴影干扰、阈值过高、bbox 内没有建筑变化。可以降低 `threshold`，或换更明显的城市建设区域和日期。

### 提示影像不重叠

两期影像必须有空间重叠。请使用同一个 bbox 下载，或检查输入影像 CRS 和范围。

## 输出文件说明

```text
data/output/*_change.tif        二值变化掩膜
data/output/*_change.gpkg       变化区域矢量
data/output/*_t1_semantic.tif   T1 建筑语义图
data/output/*_t2_semantic.tif   T2 建筑语义图
data/output/*_overlay.png       变化叠加预览图
```
