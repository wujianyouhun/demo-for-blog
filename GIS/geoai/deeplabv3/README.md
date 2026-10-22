# DeepLabV3：从基本调用到遥感影像语义分割

本示例采用 **torchvision DeepLabV3-ResNet50**，不是 DeepLabV3+。从普通 RGB 图片分割入门，再使用遥感切片微调模型，最后对整幅 GeoTIFF 做逐像元地物分类。脚本均位于 `deeplabv3/`，官方权重和训练权重默认位于项目根目录 `models/deeplabv3/`，路径通过脚本位置确定，与运行目录无关。

模型流程是：RGB 影像 → ResNet50 特征提取 → ASPP 多尺度上下文融合 → 类别分数图 → 插值回输入大小 → 每像元 argmax。ASPP 使用多种空洞率的卷积及全局池化，帮助模型结合不同尺度的地物信息。它输出的是像元类别；一栋建筑可能包含许多同类像元，模型本身不会为每栋建筑分配独立实例编号。

## 1. 安装依赖

建议 Python 3.10+，在仓库根目录运行：

```powershell
python -m pip install -r deeplabv3/requirements.txt
```

GPU 环境请先按照 [PyTorch 官网](https://pytorch.org/get-started/locally/)安装匹配的 torch / torchvision，再安装其他依赖。`--device auto` 自动选择 CUDA 或 CPU，也可明确指定 `--device cpu` 或 `--device cuda:0`。

## 2. 最小模型调用

```python
import sys
from pathlib import Path
import torch
from PIL import Image

sys.path.insert(0, str(Path("deeplabv3").resolve()))
from common import WEIGHTS, pretrained_model

model = pretrained_model().eval()  # 首次下载，后续读取 models/deeplabv3
image = Image.open("your_photo.jpg").convert("RGB")
inputs = WEIGHTS.transforms()(image).unsqueeze(0)  # [1, 3, H, W]
with torch.inference_mode():
    logits = model(inputs)["out"]  # [1, 21, H, W]，未经 softmax 的类别分数
    mask = logits.argmax(dim=1)[0]  # [H, W]，每个像元的类别编号
print(mask.shape)
```

这里的 `H/W` 对应预处理后的尺寸。完整脚本会将 logits 插值回原图大小，再生成类别掩膜。

```powershell
python deeplabv3/download_models.py
python deeplabv3/basic_inference.py --image "your_photo.jpg" --device cpu
```

输出默认位于 `deeplabv3/out/basic/`：

- `mask.png`：单波段类别编号，可用于后续分析。
- `colored_mask.png`：类别着色图。
- `overlay.png`：原图与分割结果叠加图。
- `classes.json`：类别名称、像元数和比例。

[官方权重说明](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.deeplabv3_resnet50.html)：模型用 COCO 的 VOC 类别子集训练，输出背景及 20 个物体类别，如人、车、飞机等；预处理使用短边 520 和 ImageNet 均值 / 标准差。**这些权重没有耕地、道路、裸地、光伏等遥感分类能力，不能直接将 VOC 的类别编号解释为遥感地物。**

## 3. 遥感闭环示例：生成数据 → 训练 → 整图识别

没有现成标注时，可先运行合成数据检查流程：

```powershell
python deeplabv3/generate_demo_data.py
python deeplabv3/train_remote_sensing.py --epochs 2 --batch-size 1 --device cpu
python deeplabv3/remote_sensing_inference.py --image deeplabv3/data/demo/scene.tif --device cpu
```

默认训练会加载官方预训练参数，替换最终分类层，以学习 `0=background`、`1=vegetation`、`2=building` 三类。合成数据只是颜色和几何形状模拟，短轮次结果仅说明流程能运行，不能用于真实遥感识别。CPU 较慢，正式训练建议使用 GPU。

不联网的流程检查可以给训练命令加 `--from-scratch`（随机初始化），并用生成脚本和训练脚本的 `--tile-size 64` 减小开销。已有数据目录不会被生成脚本覆盖；重复运行可换一个 `--output-dir`。

默认文件位置：

```text
models/deeplabv3/
  deeplabv3_resnet50_coco-cd0a2569.pth  # 官方权重，首次下载约 161 MB
  remote_sensing_best.pth             # 按验证集 mIoU 保存的自定义权重
  remote_sensing_best.history.json    # 每轮训练损失、验证损失和 mIoU
deeplabv3/out/remote_sensing/
  classification.tif                  # uint8 地物类别图，NoData=255
  classification.classes.json         # 各类别有效像元数、比例
```

权重已存在时训练默认报错，以防意外覆盖。使用 `--overwrite` 或指定新的 `--checkpoint models/deeplabv3/my_model.pth`；推理时传入同一个 `--checkpoint`。

## 4. 用真实遥感标注训练

将已配准的影像 / 标签切成相同大小的正方形 GeoTIFF。目录结构与本项目 U-Net 样本一致：

```text
dataset/
  train/
    images/tile_001.tif
    masks/tile_001_mask.tif
  val/
    images/tile_101.tif
    masks/tile_101_mask.tif
```

要求：标签为单波段整数类别 `0..C-1`，`255` 表示未标注 / 无效；类别 `0` 可以是背景，也可以是道路，由 `--class-names` 的顺序定义。影像和标签必须具有相同的尺寸、CRS 和地理变换。不要把有效类别 `0` 声明为标签 NoData。训练 / 验证应使用空间分离的区域，避免相邻重叠切片造成数据泄漏。

以下命令可以复用本项目已有的祁连五分类样本。检查到现有切片为四波段 uint16，抽查值域为 1839–6743，不能使用 uint8 的 `--scale 255`，否则数值会被截断为 1。下面以 RGB 位于第 1、2、3 波段、数值按 10000 缩放为示例，使用前仍需核对原始产品的波段顺序、缩放与偏移：

```powershell
python deeplabv3/train_remote_sensing.py --dataset-dir Unet/dataset/qilian_multiclass_5class --class-names road built_up cropland bare_land solar_farm --bands 1 2 3 --scale 10000 --tile-size 256 --epochs 30 --batch-size 2 --checkpoint models/deeplabv3/qilian_best.pth
python deeplabv3/remote_sensing_inference.py --image Unet/DATA/Qilian.tif --checkpoint models/deeplabv3/qilian_best.pth --output deeplabv3/out/qilian_classification.tif
```

这些命令会新训练 DeepLabV3 权重，U-Net 权重无法直接加载到 DeepLabV3。

本示例仅使用 3 个 RGB 通道：`--bands` 为从 1 开始的有序 RGB 波段编号，例如常见 B/G/R/NIR 四波段影像应使用 `--bands 3 2 1`。Sentinel-2 的波段位置依赖你的实际堆栈顺序，请核对元数据。`--scale 255` 适用于 uint8 RGB；若产品是反射率乘以 10000 的整数编码，通常用 `--scale 10000`；若已经是 `[0,1]` 浮点反射率，用 `--scale 1`。其他产品请先按其元数据处理缩放和偏移。

预处理固定为 `clip(image / scale, 0, 1)` 后再做 ImageNet 归一化，训练与推理共用同一实现。波段顺序、scale 和类别名均保存进 checkpoint，推理自动读取；**训练数据与待识别影像必须采用相同的波段语义和数值编码**。

## 5. 大图推理与 GIS 使用

```powershell
python deeplabv3/remote_sensing_inference.py --image "path/to/your_rgb.tif" --checkpoint models/deeplabv3/qilian_best.pth --tile-size 512 --halo 32 --output deeplabv3/out/landcover.tif
```

推理采用滑动窗口，内存主要由窗口大小决定，不会一次加载整幅影像。默认每次读取 512×512，边缘各留 32 像元作为上下文，只写入中心 448×448；边缘不足的部分补零，支持非整块尺寸及小于一个窗口的影像。上下文能够缓解接缝，但窗口结果与整图推理仍可能有差异。训练切片与推理窗口的尺度也会影响结果。

输出继承原图 CRS、仿射变换、宽高及分辨率，可直接加载到 QGIS / ArcGIS。任一选中波段无效或出现 NaN / Inf 时，对应输出像元为 `255`；统计排除 NoData。无地理参考的输入无法凭空生成真实坐标。输出包含调色板、类别名称元数据及统计 JSON，类别编号以 checkpoint 中的 `class_names` 顺序为准。

验证 mIoU 根据整个验证集的混淆矩阵计算，忽略 255，并排除预测与真值均未出现的类别。训练冻结 BatchNorm 的运行统计量以支持 batch_size=1。示例不包含自动标注、矢量化和额外多光谱通道改造。

## 6. 流程验证

```powershell
python deeplabv3/test_workflow.py
```

回归检查无需下载模型，验证窗口完整覆盖、极小影像、NoData、输出地理参考、固定归一化，以及标签网格错位时的报错。

本机已用合成 64×64 样本完成 1 轮 CPU 随机初始化训练，并对 101×147 的非整块影像完成推理。验证权重为 `models/deeplabv3/smoke_test.pth`，结果为 `deeplabv3/out/smoke/classification.tif`；该权重只用于流程测试。模型、数据与输出按仓库规则忽略，不随源码提交。

官方预训练权重也已下载并加载，普通图片入口已使用合成 PNG 验证，输出位于 `deeplabv3/out/basic_smoke/`。合成几何图在 VOC 模型中全部被预测为背景，符合它不具备这三种遥感地物类别的限制。

## 7. image1 / image2 真实标注六分类

`data/train/image1` 与 `image2` 的原始 class_id 含义不同，必须按中文类别名称统一后训练。当前统一为道路、建设用地/居民地、耕地、裸地、水体、光伏六类。

```powershell
python deeplabv3/prepare_training_data.py
python deeplabv3/train_remote_sensing.py --dataset-dir deeplabv3/data/landcover_6class --epochs 5 --batch-size 2 --device cpu --class-weighting inverse-sqrt --checkpoint models/deeplabv3/landcover_6class_best.pth
```

数据集的 `dataset.json` 会自动提供六类名称、波段和缩放参数；不要沿用前文旧祁连五分类的类别顺序。样本已生成时，制作脚本会拒绝覆盖，请更换输出目录。

详细核查、预览图、类别编码、验证划分和推理命令见 [六分类训练与数据核查](E:/study/demo-for-blog/GIS/geoai/deeplabv3/六分类训练与数据核查.md)。
