# DeepLabV3 遥感分类效果差怎么办？从问题诊断到优化实测

前面已经完成了遥感样本采集：裁剪影像，绘制地类边界，打断线并检查拓扑，构面，再根据影像赋类别属性。经过标签栅格化与模型训练，最终生成了自己的六分类权重 `landcover_6class_best.pth`。

接着把它用于遥感影像分类，却发现结果远没有想象中理想：人工标注里明明有道路、建设用地、水体和光伏，模型预测却主要剩下耕地和裸地。与项目中已经训练过的几个 U-Net 变体相比，DeepLabV3 的视觉效果也明显不足。

遇到这种情况，应该增加训练轮数、换模型，还是先改数据？这篇文章记录一次完整的排查与优化实践：核对原始训练结果，对比 U-Net 的实验条件，改造训练和推理代码，再用真实试验回答优化是否有效。

**先给出本次结果：优化候选的验证 mIoU 从 0.215706 变为 0.215951，提升很小，道路、水体和光伏仍没有预测像元。我们建立了更完整的优化与评估流程，但还没有解决六分类效果差的问题。**

文中的统计图、混淆矩阵、训练曲线和预测图均来自本项目实际运行记录，不使用示意数据代替实验结果。

## 先把“效果不好”变成可检查的问题

### 当前模型、数据和类别是什么

本次使用的是 torchvision 的 **DeepLabV3-ResNet50**。通过官方预训练参数初始化兼容部分，替换最终分类层，学习以下六个类别：

| 编号 | 类别 | 模型名称 |
|---|---|---|
| 0 | 道路 | road |
| 1 | 建设用地/居民地 | built_up |
| 2 | 耕地 | cropland |
| 3 | 裸地 | bare_land |
| 4 | 水体 | water |
| 5 | 光伏 | solar_farm |
| 255 | 未标注或无效像元 | 不参加训练与评价 |

这里的 0 是道路，不能把它设为标签 NoData。255 才是忽略值，也不是网络需要学习的第七个类别。

训练数据来自 image1 和 image2 两块人工标注样区，切片大小为 256×256。共生成 48 个训练切片、9 个验证切片。训练和验证没有像元交叠，但两者是相邻条带，仍然存在空间相关性；目前没有独立区域的测试结果。

首版模型只训练了 5 轮，最佳参数保存在第 4 轮。使用统一评价代码重新加载这份权重，在 489,649 个有效验证像元上，得到：

| 指标 | 原模型结果 |
|---|---:|
| 验证 mIoU | 0.215706 |
| 总体像元准确率 | 77.77% |
| 道路 IoU | 0 |
| 建设用地 IoU | 0 |
| 耕地 IoU | 0.740595 |
| 裸地 IoU | 0.553641 |
| 水体 IoU | 0 |
| 光伏 IoU | 0 |

仅看 77.77% 的总体准确率，可能觉得模型已经有一定效果；检查逐类结果之后才发现，它完全没有输出另外四个类别。

### 为什么总体准确率会掩盖问题

查看训练像元分布，原因之一就变得很直观。

![训练集六类像元分布](E:/study/demo-for-blog/GIS/geoai/deeplabv3/blog/assets/optimization/01_class_balance.png)

图中数据来自训练标签，排除了 255：

| 类别 | 训练像元 |
|---|---:|
| 道路 | 20,990 |
| 建设用地/居民地 | 116,264 |
| 耕地 | 1,083,068 |
| 裸地 | 1,303,907 |
| 水体 | 15,701 |
| 光伏 | 19,867 |

**耕地与裸地合计占训练有效像元约 93.25%。** 道路、水体和光伏分别不足 1%。在这样的数据分布下，即使模型没有学会小类别，也可能获得不低的总体准确率。

类别失衡是已经观察到的事实，但它不是经过消融实验确认的唯一原因。预处理、训练策略、目标尺度、标注口径与架构细节，也都需要检查。

## 看起来不如 U-Net，是否说明架构选错了

项目中几个 U-Net 变体的 refined 版本，确实保存了更高的验证 mIoU：

| 实验 | 轮数 | 最佳验证 mIoU |
|---|---:|---:|
| DeepLabV3 六分类首版 | 5 | 0.2157 |
| ResUNet refined 五分类 | 80 | 0.4525 |
| U-Net++ refined 五分类 | 80 | 0.4866 |
| TransUNet refined 五分类 | 80 | 0.5017 |
| Swin-UNet refined 五分类 | 80 | 0.4579 |

这些结果来自各自的真实训练历史，却不能直接组成一张公平的架构排行榜，因为实验条件不同：

- **类别不同**：DeepLabV3 是六类，包含水体；U-Net 现有实验是五类，没有水体。
- **样本不同**：DeepLabV3 使用 image1/image2 的 48/9 个训练与验证切片；U-Net 使用另一套 6/3 个切片，另有 3 个 test 切片。
- **输入不同**：DeepLabV3 使用三个波段；U-Net 使用四个波段。
- **归一化不同**：DeepLabV3 固定除以 10000，再使用 ImageNet 均值和标准差；U-Net 按每个切片各波段的 p99 归一化。
- **空间处理不同**：DeepLabV3 保持原图网格；U-Net 样本制作过程中做了重投影和重采样。

训练预算也要按实际更新次数理解。以 batch size=2 计算，DeepLabV3 五轮约更新 120 次参数，U-Net 的 80 轮约更新 240 次。**80/5 并不意味着 16 倍的参数更新次数。** 参数量、输入像元曝光和运行时间同样有影响。

所以，目前能确认的是“这份 DeepLabV3 权重表现不好”，还不能据此认定“DeepLabV3 一定不适合这项任务”。若要比较架构，应统一标签、数据划分、波段、预处理和训练预算，再进行实验。

## 在调模型前，先检查这些数据条件

### 类别编号与标注语义是否一致

两块原始标注曾出现过编号冲突：image1 的原始编号 5 是水体，image2 的原始编号 5 是光伏。当前准备脚本已经按中文类别名称统一了编号，不能退回到直接合并原始 `class_id` 的方式。

编号一致以后，还要确认语义一致。例如，建设用地是居民地整体范围，还是只标每栋建筑屋顶？光伏是仅标面板，还是包含场区内部裸地？如果两个样区口径不同，模型就会收到互相矛盾的监督信号。

拓扑有效也不等于类别正确。道路漏标、阴影判错、水岸边界偏移，都可能存在于几何合法的面要素中。应把类别面和标签栅格叠加回原影像复核。

### 波段顺序与缩放是否有产品依据

本项目影像为四波段 uint16，但源文件没有提供波段名称。目前采用“1、2、3 为 RGB，scale=10000”的假设，仍需结合原始产品说明确认。

uint16 表示存储类型，不能证明数据一定是反射率乘以 10000，也不能证明第四通道一定是近红外。

本次对训练有效像元进行抽样，三个选中通道都没有出现超过 10000 的数值。因此，尚未观察到“大量像元除错比例后被截断成 1”的问题；这仍然不能反过来证明波段与辐射处理已经正确。

### 道路宽度与像元尺度是否合适

一条只有一两个像元宽的道路，本身就比大块耕地更难识别。如果人工边界非常精细，而影像分辨率无法支持相应细节，标签并不会因此更可靠。

CRS 不同本身不会直接让神经网络分类错误，但重采样后的地面尺度会改变地物在图像中的宽度。EPSG:3857 中的 10 个投影坐标单位，也不能在所有纬度下都直接等同于精确的 10 米地面距离。

### 验证集能否代表实际应用

水体只来自 image1，光伏只来自 image2。若直接按整幅图划分训练与验证，某些类别可能只出现在一个集合中；若随机打散相邻或重叠切片，又可能让指标过于乐观。

后续需要新增包含少数类的独立样区，再按区域或采集批次分组。图像增强可以增加朝向变化，但不能创造新的传感器条件、季节或地理背景。

## 先建立统一评价，确认问题确实存在

本次首先增加了核查与评价入口：

```powershell
# 在项目根目录执行
conda activate python12

python .\deeplabv3\audit_classification.py

python .\deeplabv3\evaluate_checkpoint.py `
  --dataset-dir .\deeplabv3\data\landcover_6class `
  --checkpoint .\models\deeplabv3\landcover_6class_best.pth `
  --output .\deeplabv3\out\optimization\baseline_evaluation.json `
  --device cpu
```

评价代码在整个验证集上累积混淆矩阵，再计算 IoU、precision、recall 和 F1，不是简单平均每个切片的分数。报告同时记录真实像元数、预测像元数，以及完全没有预测的类别。

对类别 c：

```text
IoU(c) = TP(c) / [TP(c) + FP(c) + FN(c)]
```

当前验证集六类都有真实像元，因此 mIoU 是六类 IoU 的平均值。某类真实存在但没有预测时，IoU 为 0；precision 因没有预测像元而无定义，报告用 null 表示，不把它伪装成高精度。

![原模型与候选模型的混淆矩阵](E:/study/demo-for-blog/GIS/geoai/deeplabv3/blog/assets/optimization/04_confusion_matrices.png)

图中每行归一化为 100%，表示“某个真实类别最终被判到了哪里”。原模型的道路、水体、光伏全部被判为耕地，建设用地也几乎全部落入耕地。这比单看总分更能解释实际图面为什么不对。

## 优化训练代码：先让网络有机会学会小类别

新的训练入口是 `deeplabv3/train_optimized.py`，主要加入以下改动。

### 骨干与分割头使用不同学习率

预训练骨干已经具有一定特征提取能力，遥感分类层则需要适应新的类别。对两者采用不同学习率，能够分别控制更新幅度。

核心参数分组如下：

```python
optimizer = torch.optim.AdamW(
    [
        {"params": model.backbone.parameters(), "lr": 1e-5},
        {"params": model.classifier.parameters(), "lr": 3e-4},
    ],
    weight_decay=1e-4,
)
```

代码还支持先冻结骨干、短暂训练分割头，再联合微调，配合余弦学习率衰减、梯度裁剪和早停。

这是一组候选设置，不是已经证明最优的参数。冻结 BatchNorm 运行统计量可以支持很小的 batch 和 ASPP 的 1×1 池化分支，也可能限制遥感域的统计适配，需要在后续实验中权衡。

### 增加含少数类切片的出现机会

仅使用类别损失权重，不能保证每轮都充分看到少数类区域。新的采样器根据切片类别组成提高部分样本的抽中概率，并限制最大与最小采样权重比，默认不超过 3。

一个 epoch 的抽样数仍等于训练切片数，采用有放回采样，因此某些切片可能重复出现。这不会增加独立样区数量，也不能代替新增数据。

### 使用交叉熵与 Dice 组合损失

交叉熵约束逐像元分类，Dice 补充类别区域的重合程度。当前实现将二者相加，并严格排除 255：

```python
def segmentation_loss(logits, target, class_weights=None, dice_weight=0.3):
    valid = target != 255
    if not valid.any():
        return logits.sum() * 0

    ce = torch.nn.functional.cross_entropy(
        logits, target, weight=class_weights, ignore_index=255
    )
    if not dice_weight:
        return ce

    probs = logits.softmax(1) * valid[:, None]
    safe_target = target.masked_fill(~valid, 0)
    truth = torch.nn.functional.one_hot(
        safe_target, logits.shape[1]
    ).permute(0, 3, 1, 2)
    truth = truth * valid[:, None]

    present = truth.sum((0, 2, 3)) > 0
    intersection = (probs * truth).sum((0, 2, 3))
    denominator = probs.sum((0, 2, 3)) + truth.sum((0, 2, 3))
    dice = (2 * intersection + 1e-6) / (denominator + 1e-6)
    return ce + dice_weight * (1 - dice[present]).mean()
```

这段函数需要 `import torch`，输入为 `[N,C,H,W]` 的 logits 和 `[N,H,W]` 的整数标签；完整实现位于 `optimization.py`。

Dice 只对当前批次真值中出现的类别求平均，其他类别仍受交叉熵约束。不能简单认定类别权重、采样权重和 Dice 系数越大越好；多重加权可能同时放大误标样本和误检。

### 保持影像与标签同步增强

默认增加水平翻转、垂直翻转和 90° 旋转。影像与标签使用同一变换，验证集不做随机增强。

没有默认使用强烈颜色扰动，因为遥感波段值可能具有物理含义。增强应该服务于任务，而不是单纯追求变化数量。

### 保存可继续训练的完整断点

新的实验目录包含：

```text
landcover_6class_optimized_trial/
  experiment.json  # 类别、波段、预处理、样本统计与训练配方
  history.json     # 每轮损失、逐类指标与混淆矩阵
  best.pth         # 最佳验证 mIoU 对应参数，用于推理
  last.pth         # 模型、优化器、调度器、随机数状态，用于恢复
```

原来的 best 只适合初始化新实验，不能无损恢复优化器进度。新入口的 `--resume` 则读取 last，并检查训练配置是否一致。

恢复时，`--epochs` 仍是原计划总轮次。若需要更改学习率策略或延长计划，应以 best 初始化新的实验目录，避免混淆两次实验。

## 四波段和 ASPP 优化，为什么没有直接一并打开

四波段可能提供更丰富的光谱信息，但必须先确认第四通道的含义。新增入口支持 `--bands 1 2 3 4`，会构建四通道输入层并保存归一化参数；这需要重新训练，不是把已有三通道参数直接接到四波段输入上。

第四通道卷积核采用 RGB 核均值初始化，再调整整体幅度。第四通道的归一化初值使用 RGB 均值和标准差的平均值，只是一项可复现约定，不代表该通道的真实数据统计。

ASPP 同样需要结合输入大小考虑。用本机模型检查，256×256 输入对应 32×32 的骨干特征图，输出步长已是 8。默认空洞率为 12、24、36，较大分支的部分采样位置会落入填充区。

因此，代码提供 `--aspp-rates 6 12 18` 作为单独实验选项，并将配置写入 checkpoint，加载时自动恢复。不能只修改推理空洞率就声称完成优化。

本次六轮试验仍使用原来的三波段和默认 ASPP。这样能先观察训练配方的变化，避免把所有因素同时改动后无法解释结果。四波段、小空洞率、更高分辨率输入以及 DeepLabV3+ 解码器，都属于后续待验证的方向。

## 实际跑一组短程优化试验

本次从原六类最佳参数出发，运行了以下命令：

```powershell
python .\deeplabv3\train_optimized.py `
  --dataset-dir .\deeplabv3\data\landcover_6class `
  --init-checkpoint .\models\deeplabv3\landcover_6class_best.pth `
  --output-dir .\models\deeplabv3\landcover_6class_optimized_trial `
  --epochs 6 `
  --head-only-epochs 1 `
  --patience 4 `
  --device cpu --threads 2
```

运行环境为 torch 2.14.0+cpu。第一轮仅更新分割头，其余轮次联合训练；开启空间增强、有界类别采样、反平方根 CE 类别权重与系数为 0.3 的 Dice。

该实验目录现在已经存在，复现时请换一个输出目录。新入口会拒绝覆盖非空实验目录，避免把旧 best 和新 history 混在一起。

![六轮候选训练的实际验证曲线](E:/study/demo-for-blog/GIS/geoai/deeplabv3/blog/assets/optimization/02_training_curve.png)

| 轮次 | 验证 mIoU |
|---|---:|
| 1 | 0.199231 |
| 2 | 0.205935 |
| 3 | 0.186085 |
| 4 | **0.215951** |
| 5 | 0.213630 |
| 6 | 0.211627 |

最佳仍然出现在第 4 轮。重新加载保存的权重，用独立评价入口再次运行，复现了相同结果。

注意，这是一项独立的六轮余弦调度试验，不是默认 60 轮训练的前六轮，两者的学习率轨迹不同。本次也一次改变了多项训练设置，所以不能从结果中分离出每个组件的贡献。

## 优化后到底改善了什么

### 平均指标只发生很小变化

| 指标 | 原模型 | 候选最佳模型 |
|---|---:|---:|
| 验证 mIoU | 0.215706 | 0.215951 |
| 总体准确率 | 77.77% | 77.37% |

mIoU 只增加约 0.000245，即约 **0.0245 个百分点**，总体准确率反而略有下降。二者衡量方式不同，这并不矛盾；它说明必须结合逐类指标理解平均分。

![原模型与候选模型逐类 IoU 对照](E:/study/demo-for-blog/GIS/geoai/deeplabv3/blog/assets/optimization/03_class_iou.png)

| 类别 | 原 IoU | 候选 IoU |
|---|---:|---:|
| 道路 | 0.000000 | 0.000000 |
| 建设用地/居民地 | 0.000000 | 0.000318 |
| 耕地 | 0.740595 | 0.742654 |
| 裸地 | 0.553641 | 0.552736 |
| 水体 | 0.000000 | 0.000000 |
| 光伏 | 0.000000 | 0.000000 |

候选开始输出建设用地，共预测 5,624 个像元，但只有 12 个真阳性。该类 precision 约 0.21%，recall 约 0.037%。**“开始预测某类”与“已经学会某类”差距很大。**

道路、水体、光伏依然没有任何预测像元，因此不能把这次小幅分数变化视为六分类能力已经改善，更不能直接替换成业务模型。

### 同配置预测图也没有出现明显突破

为避免把更换权重和更换拼接方式混在一起，下面两份预测均使用 image2、256×256 窗口、halo=32 的同一推理入口。人工标注、原模型和候选模型采用相同颜色与叠加透明度。

![image2 原图、人工标注、原模型和候选模型的同配置对照](E:/study/demo-for-blog/GIS/geoai/deeplabv3/blog/assets/optimization/05_image2_controlled_comparison.jpg)

左上是影像，右上是人工标注，下方分别是两份权重的预测。可以看到，人工标注中的建设用地和光伏仍未得到有效恢复，预测主要由耕地和裸地组成。

image2 仅包含六类中的部分地物，而且包含参与训练的区域。这张图用于解释表现与检查流程，**不代表独立测试，也不用于计算上文的验证 mIoU**。

生成这组对照的命令如下：

```powershell
python .\deeplabv3\remote_sensing_inference.py `
  --image .\deeplabv3\data\train\image2.tif `
  --checkpoint .\models\deeplabv3\landcover_6class_best.pth `
  --tile-size 256 --halo 32 --device cpu `
  --output .\deeplabv3\out\blog_optimization\baseline_image2_center.tif

python .\deeplabv3\remote_sensing_inference.py `
  --image .\deeplabv3\data\train\image2.tif `
  --checkpoint .\models\deeplabv3\landcover_6class_optimized_trial\best.pth `
  --tile-size 256 --halo 32 --device cpu `
  --output .\deeplabv3\out\blog_optimization\trial_image2_center.tif
```

## 推理优化能做什么，不能做什么

训练优化之外，本次还增加了重叠概率融合和可选翻转 TTA。

原入口先对窗口做 argmax，再写入中心区域。新增入口则先保存每个窗口的 softmax 概率，用中心较高、边缘较低的正权重累加，归一化后再选类别：

```text
每个窗口的类别概率 × 空间权重
                  ↓ 重叠区域累加
        概率加权和 / 权重和
                  ↓ argmax
             最终类别栅格
```

这种方式可以作为接缝处理的候选方案，但不能创造网络根本没有学会的类别。本次已经跑通 image2 的概率融合与最大概率输出，尚未证明融合能提高验证精度。

```powershell
python .\deeplabv3\infer_optimized.py `
  --image .\deeplabv3\data\train\image2.tif `
  --checkpoint .\models\deeplabv3\landcover_6class_optimized_trial\best.pth `
  --tile-size 256 --stride 128 `
  --output .\deeplabv3\out\optimization\trial_image2_fused.tif `
  --confidence .\deeplabv3\out\optimization\trial_image2_max_probability.tif `
  --device cpu
```

该命令输出的 GeoTIFF 保留了 image2 原来的 728×581 尺寸、EPSG:4326 和仿射变换。可以直接在 GIS 中与原影像叠加。

添加 `--tta` 时，会对原图、水平翻转、垂直翻转和双向翻转分别推理，将概率逆变换回原位置后平均，每个窗口需要四次前向计算。

最大类别概率没有经过置信校准，不能把 0.9 理解为“这一像元有 90% 的可靠业务准确率”。对于 Qilian 这样的较大影像，概率累加采用磁盘映射，六类约需要 500 MiB 暂存空间。

若原影像未声明 NoData，而全零值实际表示边缘填充，可在确认后使用 `--zero-is-nodata`。该选项仅将所选波段同时为零的位置视为无效，不能无依据地用于所有影像。

## 让结果可以复现，也可以被否定

本次评价报告记录了验证影像和标签的文件指纹。对比程序会核对数据集、split、类别顺序、真值像元数量和文件 SHA256，避免在样本已经变化时继续比较旧分数。

```powershell
python .\deeplabv3\evaluate_checkpoint.py `
  --dataset-dir .\deeplabv3\data\landcover_6class `
  --checkpoint .\models\deeplabv3\landcover_6class_optimized_trial\best.pth `
  --output .\deeplabv3\out\optimization\trial_evaluation.json `
  --device cpu

python .\deeplabv3\compare_evaluations.py `
  --baseline .\deeplabv3\out\optimization\baseline_evaluation.json `
  --candidate .\deeplabv3\out\optimization\trial_evaluation.json `
  --output .\deeplabv3\优化短程试验对比.md
```

评价融合后的栅格时，应使用 `evaluate_raster.py`，确保预测与真值严格同网格。程序不会自动重采样来掩盖错位，也不会静默排除真值有效位置上的缺失预测。

当前已通过原流程 4 项、优化流程 5 项回归检查，覆盖忽略像元梯度、同步增强、采样权重上限、四通道前向、ASPP 配置、TTA 逆变换、窗口边缘覆盖、地理参考，以及 CPU 上完整断点恢复与连续训练的一致性。

```powershell
python .\deeplabv3\test_workflow.py
python .\deeplabv3\test_optimization.py
```

这些检查证明代码路径可以运行，不能替代分类精度与独立区域泛化评价。

## 接下来应该优先做哪些实验

根据本次结果，我会按以下顺序推进：

1. **复核并补充少数类样本。** 优先收集新的道路、水体、建设用地和光伏样区，统一边界与语义口径。特别是水体和光伏，不应始终各自只来自一块区域。
2. **确认波段与数值处理。** 根据产品说明核实 RGB 顺序、第四通道、缩放因子和偏移，不依赖“看起来颜色正常”作判断。
3. **运行更充分的 RGB 基线。** 先检查少数类是否开始获得有效召回，再进行采样、Dice 与学习率的消融实验。
4. **分别尝试四波段与较小 ASPP 空洞率。** 一次改变一个关键因素，保留失败实验，而不只记录最高分。
5. **在独立区域比较解码结构与推理策略。** 如果类别已经学会但窄目标边界仍不足，再比较 DeepLabV3+、U-Net 等结构。概率融合和 TTA 也需要独立验证收益。

下面是后续 60 轮候选实验的启动示例，**并非本文已经完成的训练结果**：

```powershell
python .\deeplabv3\train_optimized.py `
  --dataset-dir .\deeplabv3\data\landcover_6class `
  --init-checkpoint .\models\deeplabv3\landcover_6class_best.pth `
  --output-dir .\models\deeplabv3\landcover_6class_optimized `
  --epochs 60 --batch-size 2 --device auto
```

不传初始化参数则从官方预训练参数开始；额外指定 `--bands 1 2 3 4` 时，需要采用与四通道相匹配的初始化方式，不能继续加载原三通道 checkpoint。可使用新的输出目录，从官方参数构造四通道模型后重新训练。

本次最有价值的结果，是确认了模型的问题主要表现在哪里，并建立了能够复现、比较和拒绝无效改进的流程。六轮候选试验没有带来明显突破，这也提醒我们：样本中缺少的信息、类别不平衡和输入处理的不确定性，不能只靠叠加几个训练技巧解决。

## 代码与实验材料

- 诊断文档：`deeplabv3/遥感分类效果诊断与优化指南.md`
- 训练与损失：`train_optimized.py`、`optimization.py`
- 数据核查与评价：`audit_classification.py`、`evaluate_checkpoint.py`、`evaluate_raster.py`
- 结果对比：`compare_evaluations.py`
- 概率融合推理：`infer_optimized.py`
- 统一入口：`run_optimization.ps1`
- 图表生成：`blog/build_optimization_figures.py`

本文五张配图与评价记录副本位于 `blog/assets/optimization/`，保存了生成图表时的数据快照，避免后续重新训练改变本文引用的结果。发布到微信公众号时，将这些图片上传后替换正文中的本机图片地址即可。

模型结构及输入约定可参考 [torchvision DeepLabV3-ResNet50 官方文档](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.deeplabv3_resnet50.html)；损失中的权重与忽略值语义可参考 [PyTorch CrossEntropyLoss 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)。
