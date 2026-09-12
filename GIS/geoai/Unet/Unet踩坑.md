## U-Net介绍

U-Net 是语义分割领域最经典的模型之一，可以简单理解为，“输入一张影像，输出一张与原图尺寸基本一致的分类掩膜（Mask）。”特别适合建筑物提取、道路提取、水体分割、耕地分类、变化检测等“**像素级**分类”任务

U-Net有三个核心：**Encoder（编码器）**负责不断提取特征，看懂图像是什么；**Bottleneck（瓶颈层）**负责“汇总最深层语义信息”；**Decoder（解码器）**负责把语义恢复成像素级位置，是在哪里 。U-Net 还有一个重要的设计，Skip Connection 将 Encoder 的特征复制给 Decoder做拼接。简单说就是encoder深层知道这里是建筑物，但是浅层知道建筑物在哪里，Skip Connection的设计就是把两种信息结合，见高级语义和低级空间细节结合最终得到既知道是什么，有知道在哪里。

UNet可以理解为专项训练模型，SAM可以理解为通用分割模型。U-Net 优点： 精度稳定，任务专用，训练后部署轻量；SAM 优点： 通用，少样本，交互式标注强。生产训练中可以采用使用SAM进行半自动标注，生成大量的MASK，然后用于训练U-Net，在使用U-Net用于生产。

U-Net最初是2015版，后来慢慢出现了：U-Net++、 Attention U-Net、ResUNet 、ResUNet++、 UNet3+、TransUNet、Swin-UNet，进化路线如下：

```
U-Net
│
├── ResUNet
│     加 ResNet
│
├── Attention U-Net
│     加注意力
│
├── U-Net++
│     改进 Skip Connection
│
├── TransUNet
│     CNN + Transformer
│
└── Swin-UNet
      Transformer
```


**U-Net 在 GIS 中常见应用**

你以后基本都会碰到：

| 场景     | 输入             | 输出        |
| -------- | ---------------- | ----------- |
| 建筑提取 | 高分影像         | 建筑 Mask   |
| 道路提取 | 航片/卫星影像    | 道路 Mask   |
| 水体提取 | Sentinel-2       | 水体 Mask   |
| 林地分类 | 多光谱影像       | 林地 Mask   |
| 土地覆盖 | Sentinel/Landsat | LULC        |
| 农田提取 | 遥感影像         | 农田 Mask   |
| 洪水检测 | 灾前/灾后影像    | 洪水区域    |
| 滑坡识别 | UAV影像          | 滑坡区域    |
| 变化检测 | 双时相影像       | Change Mask |

## 本次实验准备

为了实际采过每个模型，现在准备两组实验数据，然后对UNet系列模型进行详细对比：

**实验 A：Sentinel-2 中分辨率遥感分割**
**实验 B：西安市城区高分辨率影像分割**

两组数据都采用真彩色数据，将试样数据的坐标系都转换成web墨卡托（3857），为了降低对设备的负荷，将数据裁切一小块，实验主要目的按图像分类、提取建筑、提取植被进行，

## UNet 进化模型对比

**ResUNet**：本质上是UNet与ResNet组合，让U-Net 更深，更好训练，比如建筑在高分辨率影像中可以看到小平房、高楼、工业厂房、农村建筑、密集居民区，ResUNet将保留 U-Net 像素定位能力，同时让 Encoder/Decoder 可以做得更深。

**Attention U-Net**：告诉神经网络重点看哪里。给U-Net增加了一个`Attention Gate`,本质上是将U-Net产生一张权重图，相当于将识别对象的进行了关联，比如建筑识别是，道路的权重会高一些，所以它特别适合：小目标、复杂背景、目标占比低、前景背景不平衡。

**U-Net++**：重新设计 Skip Connection，目的是然Encoder与Decoder之间拼接更加细致，Encoder 和 Decoder 之间不要一步跨过去，而是逐步缩小语义差距。特别适合目标大小变化很大的场景。

**TransUNet**：U-Net 开始拥抱 Transformer，让 Encoder 基于 CNN的识别建立全局关系。就比如不只是看局部纹理，看道路可能会看前面的部分和后面的部分。适用于道路网络、河流、大型建筑区、农田块、土地覆盖、变化检测。

**Swin-UNet**：Transformer 更适合高分辨率影像,同时计算量也猛增。