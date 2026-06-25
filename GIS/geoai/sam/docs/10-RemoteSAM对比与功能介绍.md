# 10 - RemoteSAM 源码级分析与本项目对比

本文基于本地仓库 `F:\stduy\AI\GIS\RemoteSAM` 的源码、README 和任务脚本，对 RemoteSAM 做源码级介绍，并与当前 GeoAI SAM 标注项目进行工程对比。

## 1. RemoteSAM 是什么

RemoteSAM 的论文题目为 **RemoteSAM: Towards Segment Anything for Earth Observation**。它的核心目标不是做一个 Web 标注工具，而是构建一个面向地球观测影像的统一视觉基础模型。

RemoteSAM 的设计思想是：以 **Referring Expression Segmentation (RES，指代表达分割)** 为核心，把像素级预测作为原子能力，再向上兼容更高层任务，例如语义分割、检测、分类、计数、视觉定位和图像描述。这样可以避免为每个遥感任务都训练一套完全独立模型。

### 1.1 与传统 SAM 的区别

| 维度 | Meta SAM / 本项目 SAM 封装 | RemoteSAM |
| --- | --- | --- |
| 核心输入 | 点、框、Mask、文本检测框 | 图像 + 文本表达或类别词 |
| 核心输出 | 分割 Mask | Mask，再派生框、分类、计数、描述 |
| 交互方式 | 人工给点/框提示 | 文本或类别驱动模型预测 |
| 工程定位 | 标注工具、标签生产 | 遥感统一模型、研究与评测 |
| GIS 能力 | 本项目补充 CRS、GeoTIFF、矢量导出 | 原仓库重点不在 GIS 坐标和矢量标签生产 |

## 2. RemoteSAM 目录结构

本地仓库主要目录如下：

```text
RemoteSAM/
├── README.md                    # 项目介绍、环境、训练、快速开始、评估命令
├── args.py                      # 训练/评估/推理参数定义
├── train.py                     # 分布式训练入口
├── transforms.py                # 图像和 Mask 预处理
├── utils.py                     # 指标、分布式、Mask/Box 转换、EPOC 等工具
├── RemoteSAM.pdf                # 论文 PDF
├── requirements.txt             # Python 依赖
├── assets/                      # README 图片、demo.jpg
├── arc/                         # 自适应旋转卷积相关实现
├── bert/                        # 项目内置 BERT 实现/配置/tokenizer
├── data/                        # 训练数据集读取
├── lib/                         # 模型主体、Swin backbone、解码头、融合模块
├── loss/                        # 训练损失
├── refer/                       # Refer 数据集接口
└── tasks/                       # 八类任务评估脚本和 demo API 封装
```

### 2.1 关键入口文件

| 文件 | 作用 |
| --- | --- |
| `args.py` | 定义训练、测试、任务评估、数据路径、模型结构等参数 |
| `train.py` | 构建 RemoteSAM-270K 数据集、模型、优化器、DDP 训练和验证 |
| `tasks/code/model.py` | 对外演示 API，封装 `RemoteSAM` 类和八类任务方法 |
| `lib/segmentation.py` | 模型构建入口，提供 `lavt` 和 `lavt_one` 两种网络 |
| `data/all_dataset.py` | RemoteSAM-270K 数据读取，加载图像、RLE Mask 和文本表达 |
| `tasks/*.sh` | 各任务评估命令脚本 |

## 3. 环境准备

RemoteSAM README 明确验证环境为：

| 项目 | 推荐版本 |
| --- | --- |
| Python | 3.8 |
| PyTorch | 1.13.0 |
| CUDA 示例 | 11.6 |
| torchvision | 0.14.0 |
| torchaudio | 0.13.0 |
| mmcv-full | 1.7.1 |

安装步骤：

```bash
conda create -n RemoteSAM python==3.8
conda activate RemoteSAM

pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 \
  --extra-index-url https://download.pytorch.org/whl/cu116

pip install mmcv-full==1.7.1 \
  -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13.0/index.html

pip install -r requirements.txt
```

`requirements.txt` 还包括 `opencv-python==4.5.3.56`、`matplotlib==3.6.1`、`scikit-image==0.19.3`、`scipy==1.9.2`、`timm==0.6.11`、`transformers==4.30.2`、`mmdet`、`mmsegmentation==0.17.0`、`pycocotools`、`shapely`、`torchmetrics` 等。

## 4. 权重和数据准备

### 4.1 训练初始化权重

训练需要 Swin Transformer 分类预训练权重：

```text
./pretrained_weights/swin_base_patch4_window12_384_22k.pth
```

准备方式：

```bash
mkdir ./pretrained_weights
```

然后从 Swin Transformer 官方地址下载 `swin_base_patch4_window12_384_22k.pth` 放入该目录。

### 4.2 RemoteSAM checkpoint

推理和评估使用：

```text
./pretrained_weights/checkpoint.pth
```

README 中示例：

```python
checkpoint = "./pretrained_weights/checkpoint.pth"
```

### 4.3 RemoteSAM-270K 数据集

RemoteSAM-270K 是 270K Image-Text-Mask 三元组数据集，覆盖 1000+ 目标类别和丰富属性表达。

README 建议数据目录：

```text
refer/data/
└── RemoteSAM-270K/
    ├── JPEGImages/
    ├── Annotations/
    ├── refs(unc).p
    └── instances.json
```

源码 `data/all_dataset.py` 实际读取以下参数：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--imageFolder` | `/data/RemoteSAM-270K/JPEGImages/` | 图像目录 |
| `--ref_file_path` | `/data/RemoteSAM-270K/refs(unc)_RemoteSAM.p` | 文本引用与 Mask 映射 |
| `--instances_path` | `/data/RemoteSAM-270K/instances.json` | COCO 风格实例标注 |

## 5. 训练流程

训练入口是 `train.py`。核心流程：

```text
解析 args.py
  -> 初始化分布式训练
  -> 构建 All_Dataset
  -> 加载 MultiModalSwinTransformer / LAVTOne
  -> AdamW 优化器
  -> LambdaLR 学习率调度
  -> 每轮 train_one_epoch
  -> evaluate 计算 mIoU / overall IoU
  -> 保存 model_best_RemoteSAM.pth 和 model_last_RemoteSAM.pth
```

README 中 8 卡训练命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch \
  --nproc_per_node 8 --master_port 12345 train.py \
  --epochs 40 --img_size 896 2>&1 | tee ./output
```

常用训练参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--epochs` | `40` | 训练轮数 |
| `--batch-size` | `2` | 单进程 batch size |
| `--lr` | `0.00003` | 初始学习率 |
| `--weight-decay` | `1e-2` | 权重衰减 |
| `--img_size` | `896` | 输入图像尺寸 |
| `--model` | `lavt_one` | 模型结构 |
| `--swin_type` | `base` | Swin 规模 |
| `--window12` | false | 使用 window size 12 |
| `--mha` | 空字符串 | 四个 stage 的融合头数 |
| `--fusion_drop` | `0.0` | PWAM 融合 dropout |
| `--output-dir` | `./checkpoints/` | checkpoint 输出目录 |

## 6. 快速开始：推理 Demo

RemoteSAM 提供 `tasks/code/model.py` 作为快速调用入口。

### 6.1 初始化模型

```python
from tasks.code.model import RemoteSAM, init_demo_model
import cv2

device = "cuda:0"
checkpoint = "./pretrained_weights/checkpoint.pth"

model = init_demo_model(checkpoint, device)
model = RemoteSAM(model, device, use_EPOC=True)
```

`init_demo_model()` 内部会：

1. 调用 `args.py` 获取默认参数。
2. 设置 `args.device` 和 `args.window12=True`。
3. 构造 `segmentation.__dict__["lavt_one"]`。
4. 从 checkpoint 的 `model` 字段加载权重。
5. 将模型放到指定设备。

### 6.2 Referring Expression Segmentation

```python
image = cv2.imread("./assets/demo.jpg")
mask = model.referring_seg(
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
    sentence="the airplane on the right"
)
```

输入是图像和自然语言描述，输出是目标二值 Mask。

### 6.3 Semantic Segmentation

```python
result = model.semantic_seg(
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
    classnames=["airplane", "vehicle"]
)
mask_airplane = result["airplane"]
```

输入类别词列表，输出每个类别对应的 Mask。

### 6.4 Object Detection

```python
result = model.detection(
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
    classnames=["airplane", "vehicle"]
)
boxes = result["airplane"]
```

源码中 detection 先调用 semantic segmentation 得到 Mask，再通过 `utils.M2B()` 将 Mask 转成水平框，并使用 `torchvision.ops.nms()` 做 NMS。

### 6.5 Visual Grounding

```python
box = model.visual_grounding(
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
    sentence="the airplane on the right"
)
```

源码中 visual grounding 先执行 referring segmentation，再把 Mask 转成一个覆盖目标的框 `[xmin, ymin, xmax, ymax]`。

### 6.6 Multi-label Classification

```python
labels = model.multi_label_cls(
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
    classnames=["airplane", "vehicle"]
)
```

源码中会对每个类别的前景概率做全局最大池化和平均池化，再用 `MLC_balance_factor` 融合，超过 0.5 即判为存在。

### 6.7 Multi-class Classification

```python
label = model.multi_class_cls(
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
    classnames=["airplane", "vehicle"]
)
```

返回得分最高且超过 0.5 的类别，否则返回 `None`。

### 6.8 Image Captioning

```python
caption = model.captioning(
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
    classnames=["airplane", "vehicle"],
    region_split=9
)
```

源码中 captioning 先检测目标框，再调用规则化描述函数 `single_captioning()` 生成空间描述。

### 6.9 Object Counting

```python
counts = model.counting(
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
    classnames=["airplane", "vehicle"]
)
```

源码中 counting 复用 detection，按类别统计框数量。

## 7. 八类任务评估脚本

RemoteSAM 在 `tasks/` 下提供评估脚本：

| 脚本 | 任务 | 关键数据参数 |
| --- | --- | --- |
| `REF.sh` | Referring Expression Segmentation | `--dataset rrsisd --split val` |
| `SEG.sh` | Semantic Segmentation | `--task SEG --save_path ./result/SEG/` |
| `DET.sh` | Object Detection | DIOR 类别循环，`--task DET --EPOC` |
| `VG.sh` | Visual Grounding | RSVG，`--task VG` |
| `MLC.sh` | Multi-label Classification | DIOR，`--task MLC` |
| `MCC.sh` | Multi-class Classification | UCM，`--task MCC` |
| `CAP.sh` | Image Captioning | UCM captions，`--task CAP` |
| `CNT.sh` | Object Counting | DIOR counting，`--task CNT --EPOC` |

示例：

```bash
bash tasks/REF.sh
bash tasks/SEG.sh
bash tasks/DET.sh
bash tasks/VG.sh
bash tasks/MLC.sh
bash tasks/MCC.sh
bash tasks/CAP.sh
bash tasks/CNT.sh
```

## 8. 本项目与 RemoteSAM 的工程对比

| 维度 | GeoAI SAM 项目 | RemoteSAM |
| --- | --- | --- |
| 工程目标 | 标注生产、Mask 后处理、矢量导出 | 遥感多任务统一模型 |
| 主要入口 | Web 页面、FastAPI、脚本 demo | Python API、训练脚本、评估脚本 |
| 影像格式 | GeoTIFF、PNG/JPG，保留 CRS | 常规图像数据集，主要按图像/标注文件读取 |
| 坐标处理 | EPSG:4326、影像 CRS、EPSG:3857、像素坐标互转 | 重点在像素级任务，不处理 GIS 坐标 |
| 大图处理 | 自动裁剪到 `_MAX_SAM_DIM=2048` | 统一 resize 到 `img_size=896` |
| 输出 | PNG Mask、GeoTIFF Mask、GeoJSON/GPKG/SHP | Mask、Box、类别、计数、描述 |
| 用户 | 标注员、GIS 工程师、数据生产人员 | 算法研究者、遥感模型工程师 |

## 9. 如何组合使用

推荐组合方式：

1. 用 RemoteSAM 对遥感图像做类别理解、计数、粗检测或文本驱动预分割。
2. 将 RemoteSAM 输出的 Mask/Box 转成当前项目可接收的点、框或预标注 Mask。
3. 在 GeoAI SAM Web 平台中人工修订。
4. 执行后处理和矢量化，导出 GIS 训练标签。

当前本文只做文档说明，不实现 RemoteSAM 接入。

