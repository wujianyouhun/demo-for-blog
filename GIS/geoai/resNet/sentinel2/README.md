# Sentinel-2 SAFE ResNet 分类格网生成流水线

该流水线处理 **Sentinel-2 Level-2A SAFE** 产品，输入既可以是解压后的 `.SAFE` 文件夹，也可以是下载完成的 `.SAFE.zip` 压缩包。处理顺序如下：

```text
SAFE.zip（自动解压）或 SAFE 产品
  → 读取 B02、B03、B04（10 m）和 SCL（20 m）
  → 统一投影并按波段跨场景镶嵌
  → 裁剪研究区
  → SCL 云、云影、雪冰和无效像元掩膜
  → B04/B03/B02 反射率 RGB 合成
  → 固定窗口滑动切片
  → 缩放到 224×224、ImageNet 标准化
  → ResNet 批量推理
  → 分类类别格网、置信度格网和 JSON 明细
```

## 需要的 SAFE 内容

每个 SAFE 中必须具有以下波段：

- `IMG_DATA/R10m/*_B02_10m.jp2`：蓝光
- `IMG_DATA/R10m/*_B03_10m.jp2`：绿光
- `IMG_DATA/R10m/*_B04_10m.jp2`：红光
- `IMG_DATA/R20m/*_SCL_20m.jp2`：场景分类层，用来掩膜云和云影

因此，输入必须是 Level-2A，而不是仅含 TOA 反射率的 Level-1C 产品。你的截图中 `S2A_MSIL2A_...SAFE.zip` 即属于脚本支持的输入格式。

## 安装和运行

```bash
pip install -r resNet/sentinel2/requirements_sentinel2.txt
python sentinel2_pipeline.py --input-safe E:\study\论文\祁连山国家公园\data\sentinel2_l2a_cdse\products\2023\06 --output-dir E:\study\demo-for-blog\GIS\geoai\resNet\sentinel2 --tile-size 192 --batch-size 32
```

其中 `D:/sentinel2_safe` 可以是存放多个 `.SAFE.zip` 的目录，也可以直接是某一个 `.SAFE.zip` 文件路径；不能填写压缩包内部的路径。脚本会解压到 `输出目录/00_extracted_safe/`，不会删除或修改原始 ZIP。

默认裁剪范围为 `97.38, 36.48, 103.77, 39.73`（经度、纬度，WGS84）。可自行覆盖：

```bash
python resNet/sentinel2/sentinel2_pipeline.py --input-safe D:/sentinel2_safe --output-dir D:/sentinel2_result --bbox 97.38 36.48 103.77 39.73
```

## 输出说明

- `01_mosaic_clipped/`：先镶嵌、后裁剪的 B02、B03、B04 与 SCL。
- `00_extracted_safe/`：从 `.SAFE.zip` 自动解压出的 SAFE 产品；输入本身已经是 SAFE 目录时，该目录为空。
- `02_rgb_reflectance.tif`：按 B04/B03/B02 合成、并应用 SCL 掩膜后的三波段反射率影像。
- `03_inference/classification_grid.tif`：每一个像元代表一个切片，`-1` 表示云或无效像元超过 20%。
- `03_inference/confidence_grid.tif`：对应的最大预测概率。
- `03_inference/tile_predictions.json`：每个格网的类别、概率与行列号。

默认 `192 × 192` 像素切片在 10 m Sentinel-2 数据上约为 `1.92 km × 1.92 km`，与 Landsat 8 流程默认 `64 × 64` 像素切片覆盖的地面范围相近。

## 反射率偏移量

多数 Sentinel-2 L2A 产品可以使用默认换算 `DN / 10000`。若 SAFE 的 `MTD_MSIL2A.xml` 中标注了 `BOA_ADD_OFFSET=-1000`，请将该值传入：

```bash
python resNet/sentinel2/sentinel2_pipeline.py --input-safe D:/sentinel2_safe --output-dir D:/sentinel2_result --reflectance-offset -1000
```

训练和推理必须使用一致的波段组合、反射率换算、云掩膜策略与切片尺度。若加载不到微调后的 `models/Classification/checkpoints/best_model.pth`，脚本会回退到 ImageNet ResNet50；这只能检验流程，不能作为可靠的地物分类成果。
