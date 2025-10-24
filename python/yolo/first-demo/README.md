# YOLO v8 多模型目标检测

本项目演示如何使用不同的 YOLO v8 模型（nano、small、medium、large、extra large）对图片进行目标检测，并保存检测结果。

## 功能特点

- 🔥 支持 5 种 YOLO v8 模型（yolov8n, yolov8s, yolov8m, yolov8l, yolov8x）
- 📷 自动处理输入图片
- 📁 结果按"图片名称-模型名称"格式保存
- 🎯 置信度阈值可配置（默认 0.25）
- 📊 显示检测统计信息
- 🚀 自动下载预训练模型
- 🛠️ PyTorch 2.6 兼容性支持

## 环境要求

- Python 3.8+
- PyTorch (兼容 2.6 版本)
- OpenCV
- Ultralytics

安装依赖：
```bash
pip install ultralytics opencv-python torch torchvision
```

## 使用方法

1. 将要检测的图片放在当前目录中（默认为 pic.jpg 和 pic2.png）
2. 运行检测脚本：
```bash
python yolo_detected.py
```

## 输出结果

检测结果将保存在 `result/` 目录中，文件命名格式为：
- `pic-yolov8n.jpg` - 使用 YOLOv8n 模型检测 pic.jpg
- `pic-yolov8s.jpg` - 使用 YOLOv8s 模型检测 pic.jpg
- `pic-yolov8m.jpg` - 使用 YOLOv8m 模型检测 pic.jpg
- `pic-yolov8l.jpg` - 使用 YOLOv8l 模型检测 pic.jpg
- `pic-yolov8x.jpg` - 使用 YOLOv8x 模型检测 pic.jpg
- `pic2-yolov8n.png` - 使用 YOLOv8n 模型检测 pic2.png
- `pic-yolov8s.png` - 使用 YOLOv8s 模型检测 pic2.png
- `pic-yolov8m.png` - 使用 YOLOv8m 模型检测 pic2.png
- `pic-yolov8l.png` - 使用 YOLOv8l 模型检测 pic2.png
- `pic-yolov8x.png` - 使用 YOLOv8x 模型检测 pic2.png

## 模型说明

### YOLOv8n (nano)
- **特点**: 最轻量级，速度最快
- **适用场景**: 实时检测、资源受限环境
- **文件大小**: ~6MB

### YOLOv8s (small)
- **特点**: 平衡性能和精度
- **适用场景**: 通用目标检测
- **文件大小**: ~22MB

### YOLOv8m (medium)
- **特点**: 更高精度，但速度较慢
- **适用场景**: 需要更高检测精度的场景
- **文件大小**: ~50MB

### YOLOv8l (large)
- **特点**: 高精度，检测能力强
- **适用场景**: 需要极高精度的复杂场景
- **文件大小**: ~83MB

### YOLOv8x (extra large)
- **特点**: 最高精度，但速度最慢
- **适用场景**: 研究或对精度要求极高的场景
- **文件大小**: ~130MB

## 检测结果示例

运行脚本时，控制台会显示：
```
=== 使用模型: yolov8n ===
成功加载模型: yolov8n
检测图片: pic.jpg
检测到 5 个物体
  - dog: 5
结果已保存到: result\pic-yolov8n.jpg
```

## 自定义配置

### 修改检测图片
编辑 `multi_model_detection.py` 文件中的 `input_images` 列表：
```python
input_images = ['your_image1.jpg', 'your_image2.png']  # 修改为你自己的图片
```

### 修改置信度阈值
在预测时调整 `conf` 参数：
```python
results = model(image_name, conf=0.3)  # 提高置信度阈值
```

### 修改检测模型
在 `model_names` 列表中添加或移除模型：
```python
model_names = ['yolov8n', 'yolov8s', 'yolov8m']  # 默认包含全部5种模型
```

## 技术说明

### PyTorch 2.6 兼容性
脚本通过 `torch.load` 修补解决了 PyTorch 2.6 的安全加载问题，确保 YOLO 模型能正常加载。

### 自动模型下载
首次运行时，Ultralytics 会自动从官方仓库下载预训练模型。

### 输出图片格式
- 保持原图片格式（jpg → jpg，png → png）
- 在原图基础上绘制检测框和标签
- 显示置信度分数

## 项目结构

```
first-demo/
├── yolo_detected.py         # 主检测脚本
├── pic.jpg                   # 示例图片1
├── pic2.png                  # 示例图片2
├── result/                   # 检测结果输出目录
│   ├── pic-yolov8n.jpg
│   ├── pic-yolov8s.jpg
│   ├── pic-yolov8m.jpg
│   ├── pic-yolov8l.jpg
│   ├── pic-yolov8x.jpg
│   ├── pic2-yolov8n.png
│   ├── pic2-yolov8s.png
│   ├── pic2-yolov8m.png
│   ├── pic2-yolov8l.png
│   └── pic2-yolov8x.png
├── yolov8n.pt               # 模型文件（自动下载）
├── yolov8s.pt               # 模型文件（自动下载）
├── yolov8m.pt               # 模型文件（自动下载）
├── yolov8l.pt               # 模型文件（自动下载）
├── yolov8x.pt               # 模型文件（自动下载）
└── README.md               # 说明文档
```

## 注意事项

1. **首次运行**：会下载模型文件，需要网络连接
2. **图片格式**：支持 JPG、PNG 等常见图片格式
3. **内存要求**：较大模型（如 yolov8m）需要更多内存
4. **检测精度**：不同模型可能会有不同的检测结果

## 参考资源

- [Ultralytics YOLO v8 官方文档](https://docs.ultralytics.com/)
- [YOLO v8 模型性能对比](https://docs.ultralytics.com/models/yolov8/)
- [PyTorch 官方文档](https://pytorch.org/)

