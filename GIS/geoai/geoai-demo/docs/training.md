# 模型训练指南

## 概述

本项目使用 **DeepLabV3+** 进行语义分割, 支持多种骨干网络。

## 支持模型

| 模型 | 骨干网络 | 参数量 | 推理速度 | 精度 |
|------|----------|--------|----------|------|
| deeplabv3p_resnet50 | ResNet-50 | ~40M | 快 | 好 |
| deeplabv3p_resnet101 | ResNet-101 | ~60M | 中 | 很好 |
| deeplabv3p_xception | Xception | ~45M | 中 | 很好 |
| deeplabv3p_mobilenetv2 | MobileNetV2 | ~8M | 很快 | 一般 |

## 训练流程

### 1. 准备数据

确保数据目录包含 `images/` 和 `labels/` 子目录:

```bash
python scripts/prepare_data.py --region beijing --date 2023-06-01
```

### 2. 启动训练

```bash
# 基础训练
python scripts/train.py --model deeplabv3p_resnet50 --epochs 10

# 自定义参数
python scripts/train.py \
  --model deeplabv3p_resnet101 \
  --epochs 50 \
  --batch-size 8 \
  --lr 0.0005 \
  --data data/beijing \
  --output output/models
```

### 3. 通过 Web 界面训练

1. 切换到 "模型训练" 标签
2. 选择模型和参数
3. 点击 "准备样本" (生成训练切片)
4. 点击 "开始训练"

## 训练参数

| 参数 | CLI 标志 | 默认值 | 说明 |
|------|----------|--------|------|
| 模型 | --model | deeplabv3p_resnet50 | 模型架构 |
| 训练轮数 | --epochs | 10 | 总训练轮数 |
| 批量大小 | --batch-size | 4 | 每批样本数 |
| 学习率 | --lr | 0.001 | 初始学习率 |
| 数据目录 | --data | data/beijing | 训练数据路径 |
| 输出目录 | --output | output/models | 检查点保存路径 |

## 数据增强

训练时使用以下 albumentations 增强策略:

- **RandomCrop(256, 256)** - 随机裁剪到 256x256
- **HorizontalFlip(p=0.5)** - 水平翻转
- **VerticalFlip(p=0.3)** - 垂直翻转
- **RandomRotate90(p=0.5)** - 随机 90 度旋转
- **Normalize** - ImageNet 均值/标准差归一化

## 训练输出

训练完成后输出:
- `model_checkpoint.json` - 模型检查点 (权重、训练历史)
- 训练损失和 mIoU 曲线

## 评估指标

- **mIoU (mean Intersection over Union)** - 主要评估指标
- **Loss** - 交叉熵损失

## API 接口

```
POST /api/train/prepare-samples
{ "model": "deeplabv3p_resnet50" }

POST /api/train/start
{
  "model": "deeplabv3p_resnet50",
  "epochs": 10,
  "batch_size": 4,
  "learning_rate": 0.001
}

GET /api/train/status/{task_id}
GET /api/train/models
```
