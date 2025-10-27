# YOLO v8 多模型检测使用说明

## 概述

本目录包含了多个 YOLO v8 模型的使用示例，涵盖不同类型的计算机视觉任务。

## 文件说明

### 1. `yolo_class.py` - 模型信息查看程序
- **功能**: 显示 YOLO 模型的详细信息并保存到文本文件
- **输出**: `yolov8n-pose_model_info.txt`
- **用途**: 了解模型的检测类别、框架信息等

### 2. `yolo_multi_model_demo.py` - 完整的多模型演示程序
- **功能**: 演示所有 YOLO 模型的使用方法
- **特点**: 详细的输出信息和完整的错误处理

### 3. `quick_demo.py` - 快速演示脚本
- **功能**: 简化的模型使用示例
- **特点**: 代码简洁，适合快速测试

### 4. `yolo_examples.py` - 详细的示例集
- **功能**: 包含5个独立的使用示例
- **特点**: 详细的注释和说明

## 模型说明

### 1. yolov8n-cls.pt - 图像分类模型
- **功能**: 对图像进行分类
- **输入**: `animal.jpg`
- **输出**: 分类结果和置信度
- **示例代码**:
```python
model = YOLO('yolov8n-cls.pt')
results = model('animal.jpg')
```

### 2. yolov8n-pose.pt - 姿态检测模型
- **功能**: 检测人体关键点和骨架
- **输入**: `sport.jpg`
- **输出**: 17个人体关键点坐标
- **示例代码**:
```python
model = YOLO('yolov8n-pose.pt')
results = model('sport.jpg')
```

### 3. yolov8n-obb.pt - 旋转目标检测模型
- **功能**: 检测旋转的物体
- **输入**: `test.png`
- **输出**: 旋转边界框（OBB）
- **示例代码**:
```python
model = YOLO('yolov8n-obb.pt')
results = model('test.png')
```

### 4. yolov8n-seg.pt - 实例分割模型
- **功能**: 对物体进行精确分割
- **输入**: `动物.jpg`, `car.png`, `people.png`
- **输出**: 实例分割掩码
- **示例代码**:
```python
model = YOLO('yolov8n-seg.pt')
results = model(['动物.jpg', 'car.png', 'people.png'])
```

### 5. yolov8n.pt - 标准目标检测模型
- **功能**: 标准目标检测
- **输入**: 所有图片文件
- **输出**: 边界框和类别
- **示例代码**:
```python
model = YOLO('yolov8n.pt')
results = model(['animal.jpg', 'sport.jpg', 'test.png'])
```

## 使用步骤

### 1. 环境准备
```bash
# 安装必要的库
pip install ultralytics opencv-python torch torchvision
```

### 2. 准备图片文件
确保以下图片文件存在：
- `animal.jpg` - 动物图片（用于分类和检测）
- `sport.jpg` - 运动图片（用于姿态检测）
- `test.png` - 测试图片（用于旋转检测）
- `动物.jpg` - 中文命名的动物图片（用于分割）
- `car.png` - 汽车图片（用于分割和检测）
- `people.png` - 人物图片（用于分割和检测）

### 3. 运行程序

#### 方式一：运行完整演示
```bash
python yolo_multi_model_demo.py
```

#### 方式二：运行快速演示
```bash
python quick_demo.py
```

#### 方式三：运行详细示例
```bash
python yolo_examples.py
```

#### 方式四：查看模型信息
```bash
python yolo_class.py
```

## 输出文件说明

运行程序后会生成以下文件：

### 分类结果
- `classification_animal_result.jpg` - 图像分类结果

### 姿态检测结果
- `pose_sport_result.jpg` - 姿态检测结果

### 旋转检测结果
- `obb_test_result.png` - 旋转目标检测结果

### 实例分割结果
- `动物_segmentation_result.png` - 动物分割结果
- `car_segmentation_result.png` - 汽车分割结果
- `people_segmentation_result.png` - 人物分割结果

### 标准检测结果
- `animal_detection_result.jpg` - 动物检测结果
- `sport_detection_result.jpg` - 运动检测结果
- `test_detection_result.jpg` - 测试图片检测结果
- `动物_detection_result.jpg` - 中文动物检测结果
- `car_detection_result.jpg` - 汽车检测结果
- `people_detection_result.jpg` - 人物检测结果

### 模型信息文件
- `yolov8n-pose_model_info.txt` - 模型详细信息

## 注意事项

1. **模型自动下载**: 首次运行时，模型会自动下载，请确保网络连接正常

2. **图片要求**:
   - 支持的格式：jpg, png, jpeg
   - 建议图片大小：640x640 或保持原始比例
   - 图片内容应与模型任务匹配

3. **内存要求**:
   - 同时运行多个模型需要足够的内存
   - 建议至少 4GB 可用内存

4. **性能优化**:
   - 可以使用 `model.to('cuda')` 启用GPU加速（需要CUDA支持）
   - 大批量处理时可以使用批处理功能

## 常见问题

### Q: 模型下载失败怎么办？
A: 请检查网络连接，或手动下载模型文件到当前目录

### Q: 图片路径错误怎么办？
A: 确保图片文件存在于当前目录，或使用完整路径

### Q: 运行速度很慢怎么办？
A: 可以使用更小的模型（如 yolov8n.pt）或启用GPU加速

### Q: 检测结果不理想怎么办？
A: 可以调整置信度阈值：`results = model(image, conf=0.5)`

## 技术支持

如遇到问题，请检查：
1. Python 版本（建议 3.8+）
2. 依赖库版本
3. 图片文件格式和路径
4. 网络连接（模型下载）

---

作者: Claude
日期: 2025-10-27