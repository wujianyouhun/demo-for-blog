# YOLO目标检测项目

基于YOLOv8的目标检测演示项目，支持GPU/CPU自动切换。

## 项目结构

```
yolo-env/
├── detection.py          # 主检测脚本
├── requirements.txt      # 项目依赖
├── README.md            # 项目说明
├── images/              # 待检测图片目录
│   └── 357548.jpg
└── runs/                # 检测结果输出目录(自动生成)
    └── detect/
        └── results/     # 检测结果图片和标注
```

## 功能特性

- ✅ **自动设备检测**: 优先使用GPU，无GPU时自动切换到CPU
- ✅ **设备信息显示**: 显示当前使用的设备类型和详细信息
- ✅ **检测结果可视化**: 自动保存带有检测框的图片
- ✅ **详细结果输出**: 显示检测到的目标类别、置信度和位置信息
- ✅ **多模型支持**: 支持YOLOv8系列不同大小的模型

## 环境要求

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA (可选，用于GPU加速)

## 安装步骤

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 如需使用GPU加速

首先检查系统是否安装了CUDA:

```bash
nvidia-smi
```

根据CUDA版本安装对应的PyTorch版本:

**CUDA 11.8:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.1:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**仅CPU版本:**
```bash
pip install torch torchvision
```

## 使用方法

### 基本使用

直接运行脚本检测 `images/357548.jpg`:

```bash
python detection.py
```

### 自定义使用

可以修改 `detection.py` 中的参数:

```python
# 修改图片路径
image_path = "your_image_path.jpg"

# 选择不同大小的模型
model_name = 'yolov8n.pt'  # 最小最快
model_name = 'yolov8s.pt'   # 小型
model_name = 'yolov8m.pt'   # 中型
model_name = 'yolov8l.pt'   # 大型
model_name = 'yolov8x.pt'   # 最大最准
```

## 输出说明

运行后会在终端输出:

1. **设备信息**: 使用的GPU或CPU
2. **检测进度**: 模型加载和检测状态
3. **检测结果**:
   - 检测到的目标数量
   - 每个目标的类别、置信度和位置
4. **结果保存路径**: 标注后的图片保存位置

检测结果将保存在 `runs/detect/results/` 目录下:
- `*.jpg`: 带有检测框的图片
- `*.txt`: 检测结果的文本格式

## 模型说明

| 模型 | 大小 | mAP | 速度 | 推荐场景 |
|------|------|-----|------|----------|
| yolov8n | 最小 | 中等 | 最快 | 实时检测、移动设备 |
| yolov8s | 小 | 较好 | 快 | 一般应用 |
| yolov8m | 中 | 好 | 中等 | 平衡性能和速度 |
| yolov8l | 大 | 很好 | 慢 | 高精度要求 |
| yolov8x | 最大 | 最好 | 最慢 | 最高精度 |

## 常见问题

### 1. 模型下载慢
首次运行会自动下载模型，如下载慢可手动下载后放入 `~/.ultralytics/` 目录。

### 2. CUDA out of memory
使用更小的模型(如yolov8n.pt)或减少batch size。

### 3. 检测不到目标
尝试降低置信度阈值，修改 `conf` 参数:
```python
conf=0.15  # 默认0.25，降低可检测更多目标
```

## 技术栈

- **PyTorch**: 深度学习框架
- **Ultralytics YOLOv8**: 目标检测模型
- **OpenCV**: 图像处理
- **NumPy**: 数值计算

## 参考资料

- [YOLOv8 官方文档](https://docs.ultralytics.com/)
- [PyTorch 官方文档](https://pytorch.org/docs/)

