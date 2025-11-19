# 🍎 YOLO Fruit Detection Project

Intelligent fruit detection system based on YOLOv8, supporting object detection and instance segmentation with automatic fruit recognition and counting.

## 🚀 Features

- **🔍 Dual Model Detection**: Combines YOLOv8n (object detection) and YOLOv8n-seg (instance segmentation)
- **🎯 Multi-Fruit Recognition**: Supports apples, bananas, oranges, broccoli, carrots, and more
- **⚡ GPU Acceleration**: Automatic GPU detection with CPU fallback
- **📊 Visual Results**:
  - Bounding box detection images (showing fruit locations and confidence scores)
  - Instance segmentation images (showing precise fruit contours)
- **📈 Statistical Reports**: Generates detailed detection reports and statistics
- **💾 Multiple Output Formats**: JPG images and TXT/JSON data reports
- **🚄 Performance Optimized**: Model warmup, batch processing, GPU memory management
- **🌍 English Labels**: All outputs use English labels for international compatibility

## 📁 Project Structure

```
yolo-fruit/
├── fruit_detector.py       # Main detection script (optimized)
├── requirements.txt        # Dependencies
├── image/                  # Input images folder
│   ├── *.jpg
│   ├── *.jpeg
│   └── *.png
└── result/                 # Output results folder
    ├── *_detection.jpg     # Bounding box detection results
    ├── *_segmentation.jpg  # Instance segmentation results
    ├── detection_report_*.txt   # Text reports
    └── detection_report_*.json  # JSON data reports
```

## 🍓 Supported Food Types

| English Name | Description |
|--------------|-------------|
| Apple | Red or green round fruit |
| Banana | Yellow elongated fruit |
| Orange | Orange round citrus fruit |
| Broccoli | Green vegetable |
| Carrot | Orange root vegetable |
| Pizza | Baked Italian dish |
| Cake | Sweet dessert |
| Sandwich | Bread with filling |
| Hot Dog | Sausage in bread |
| Donut | Ring-shaped pastry |
| Cup | Drinking vessel |
| Fork | Eating utensil |
| Knife | Cutting utensil |
| Spoon | Eating utensil |
| Bowl | Food container |

## 🛠️ 安装与环境配置

### 1. 环境要求

- Python 3.8+
- CUDA 11.0+ (可选，用于GPU加速)
- 至少4GB内存

### 2. 安装依赖

```bash
# 安装PyTorch (根据你的CUDA版本选择)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install -r requirements.txt
```

或者手动安装：

```bash
pip install ultralytics>=8.0.0
pip install opencv-python>=4.5.0
pip install numpy>=1.20.0
pip install Pillow>=8.0.0
```

### 3. 模型下载

首次运行时，脚本会自动下载：
- `yolov8x.pt` - YOLOv8x目标检测模型
- `yolov8x-seg.pt` - YOLOv8x实例分割模型

## 🏃‍♂️ Usage

### Quick Start

1. **Prepare Images**: Place images to detect in the `image/` folder
2. **Run Script**:
   ```bash
   python fruit_detector.py
   ```
3. **View Results**: Check the `result/` folder for outputs

### Supported Image Formats

- JPG (.jpg, .jpeg)
- PNG (.png)
- Case-insensitive extensions

### Advanced Usage

#### Custom Confidence Threshold

Initialize the detector with custom settings:

```python
from fruit_detector import FruitDetector

# Create detector with custom confidence threshold
detector = FruitDetector(
    confidence_threshold=0.3,  # Lower threshold for higher recall
    model_size='n',            # n=nano, s=small, m=medium, l=large, x=xlarge
    verbose=True
)

# Process images
results = detector.process_directory()
```

#### Process Single Image

```python
from pathlib import Path
from fruit_detector import FruitDetector

detector = FruitDetector()
result = detector.detect_image(Path('image/my_fruit.jpg'))
print(f"Detected: {result}")
```

## 📊 Output Results

### Image Outputs

- **Detection Results**: `imagename_detection.jpg`
  - Green bounding boxes marking detected fruits
  - Shows fruit names (in English) and confidence scores
  - Black text on green background for better visibility

- **Segmentation Results**: `imagename_segmentation.jpg`
  - Yellow semi-transparent masks covering detected fruits
  - Shows precise fruit contours

### Text Report

**Example Format**:
```
🍎 FRUIT DETECTION REPORT 🍎
============================================================
Detection Time: 2025-11-19 18:45:00
Device: CUDA
Model: YOLOv8n
Confidence Threshold: 0.5
Total Images Processed: 2

PER-IMAGE RESULTS:
------------------------------------------------------------

📷 apple.jpg
   • Apple: 3
   • Banana: 1

📷 fruits.jpg
   • Orange: 2

============================================================
SUMMARY STATISTICS:
------------------------------------------------------------
Apple: 3
Banana: 1
Orange: 2

🎯 Total Fruits Detected: 6
```

### JSON Report

Contains detailed detection data for programmatic processing and analysis, including metadata, per-image results, and summary statistics.

## ⚙️ Configuration Options

### Performance Optimizations

- **GPU Usage**: Automatic GPU detection with CPU fallback
- **Model Warmup**: Pre-runs inference for consistent performance
- **Memory Management**: GPU cache clearing after each image
- **Batch Processing**: Efficient processing of multiple images with progress tracking
- **Performance Metrics**: FPS and processing time tracking

### Detection Parameters

- **Confidence Threshold**: Default 0.5, adjustable (0.0-1.0)
- **Model Size**: Default 'n' (nano), options: n/s/m/l/x
- **Device**: Auto-detected, can be manually set to 'cuda' or 'cpu'
- **Verbose Mode**: Detailed progress information (default: True)

## 🔧 故障排除

### 常见问题

1. **模型下载失败**
   ```
   解决方案: 检查网络连接，或手动下载模型文件到当前目录
   ```

2. **CUDA内存不足**
   ```
   解决方案: 重启脚本或减少同时处理的图片数量
   ```

3. **图片无法读取**
   ```
   解决方案: 检查图片格式和路径，确保文件未损坏
   ```

4. **检测结果不准确**
   ```
   解决方案:
   - 调整置信度阈值
   - 确保图片清晰度和光线充足
   - 尝试不同角度和距离的图片
   ```

### 性能建议

- **使用GPU**: 确保安装了CUDA版本的PyTorch
- **图片预处理**: 建议图片分辨率在640x640以上
- **批量处理**: 适合处理大量图片的自动化场景

## 📈 Performance Metrics

- **Detection Speed** (YOLOv8n):
  - GPU (CUDA): ~15-30ms/image (~30-60 FPS)
  - CPU: ~200-400ms/image (~2-5 FPS)
- **Accuracy**: Trained on COCO dataset, mAP@0.5 > 0.5
- **Supported Resolution**: Adaptive, recommended 640x640+
- **Memory Usage**: 
  - YOLOv8n: ~6MB model size
  - GPU VRAM: ~500MB-1GB during inference

## 🔬 Technical Details

### Model Architecture

- **YOLOv8n**: Latest YOLO series object detection model (nano version)
- **YOLOv8n-seg**: YOLOv8 variant with instance segmentation support
- **Backbone**: CSPDarknet
- **Neck**: PANet
- **Head**: Detection head + Segmentation head

### Detection Pipeline

1. **Image Preprocessing**: Resize and normalize
2. **Model Inference**: Object detection + Instance segmentation
3. **Post-processing**: NMS and confidence filtering
4. **Result Visualization**: Draw bounding boxes and masks
5. **Data Statistics**: Fruit counting and report generation
6. **Performance Tracking**: FPS and timing measurements

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

本项目基于MIT许可证开源。

## 🙏 致谢

- [Ultralytics](https://ultralytics.com/) - YOLOv8模型开发
- [COCO Dataset](https://cocodataset.org/) - 训练数据集

---

**Happy Fruit Detection! 🍎🍌🍊**