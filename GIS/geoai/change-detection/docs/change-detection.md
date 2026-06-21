# 变化检测算法

## 概述

本项目实现两种基于深度学习的二值变化检测模型：

1. **Siamese U-Net** — 经典孪生网络，两个时相共享编码器权重
2. **BiT (Bi-temporal Image Transformer)** — 引入 Transformer 交叉注意力

## Siamese U-Net

### 架构

```
时相 A 影像 ──→ [共享编码器] ──→ 特征 A_1..A_5 ──┐
                                                 ├── 多尺度拼接 → U-Net 解码器 → 变化图
时相 B 影像 ──→ [共享编码器] ──→ 特征 B_1..B_5 ──┘
```

- **编码器**: ResNet-34 (或 MobileNetV2 轻量版)，输出 5 个尺度特征
- **特征融合**: 每个尺度将 A/B 特征拼接 (concat)
- **解码器**: U-Net 风格逐步上采样，跳跃连接
- **输出**: 2 通道 softmax (变化概率/未变化概率)

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| encoder | resnet34 | 骨干网络 |
| in_channels | 3 | 输入通道 (RGB) |

## BiT (Bi-temporal Image Transformer)

### 架构

```
时相 A ──→ [CNN编码器] ──→ 特征 A ──┐
                                     ├── 拼接 → 投影 → Transformer → 解码器 → 变化图
时相 B ──→ [CNN编码器] ──→ 特征 B ──┘
```

- **编码器**: ResNet-50，取最深层特征
- **Transformer**: 4 层多头自注意力 (8 heads, embed_dim=256)
- **解码器**: 3 层转置卷积上采样

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| encoder | resnet50 | 骨干网络 |
| embed_dim | 256 | Transformer 嵌入维度 |
| num_heads | 8 | 注意力头数 |

## 评价指标

| 指标 | 公式 | 说明 |
|------|------|------|
| F1 | 2PR/(P+R) | 精确率和召回率的调和平均 |
| IoU | TP/(TP+FP+FN) | 交并比 |
| OA | (TP+TN)/(Total) | 总体精度 |
| Kappa | (OA-Pe)/(1-Pe) | 一致性系数 |

## 训练策略

- **损失函数**: 标准交叉熵 (CrossEntropyLoss)
- **优化器**: AdamW (lr=1e-4, weight_decay=1e-4)
- **调度器**: Cosine Annealing
- **数据增强**: 水平/垂直翻转、随机旋转90°、亮度对比度抖动
- **混合精度**: CUDA GPU 自动启用 AMP
- **早停**: 10 个 epoch 无改善则停止
