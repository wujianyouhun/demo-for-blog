"""
YOLO v8 快速演示脚本
==================

本脚本提供各个YOLO模型的快速使用示例

使用方法:
1. 确保已安装依赖: pip install ultralytics opencv-python
2. 准备相应的图片文件
3. 运行脚本: python quick_demo.py
"""

from ultralytics import YOLO
import cv2
import torch

# PyTorch 2.6 补丁
def patch_torch_load():
    original_load = torch.load
    def patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    torch.load = patched_load

patch_torch_load()

# 示例1: 图像分类
def classification_example():
    """使用 yolov8n-cls 检测 animal.jpg"""
    print("=== 图像分类示例 ===")
    model = YOLO('yolov8n-cls.pt')
    results = model('animal.jpg')
    result = results[0]
    print(f"预测类别: {model.names[int(result.probs.top1)]}")
    print(f"置信度: {result.probs.top1conf:.4f}")

# 示例2: 姿态检测
def pose_example():
    """使用 yolov8n-pose 检测 sport.jpg"""
    print("=== 姿态检测示例 ===")
    model = YOLO('yolov8n-pose.pt')
    results = model('sport.jpg')
    result = results[0]
    result.save('pose_result.jpg')
    print(f"检测到 {len(result.keypoints)} 个人体姿态")

# 示例3: 旋转目标检测
def obb_example():
    """使用 yolov8n-obb 检测 test.png"""
    print("=== 旋转目标检测示例 ===")
    model = YOLO('yolov8n-obb.pt')
    results = model('test.png')
    result = results[0]
    result.save('obb_result.png')
    print(f"检测到 {len(result.obb)} 个旋转目标")

# 示例4: 实例分割
def segmentation_example():
    """使用 yolov8n-seg 检测多张图片"""
    print("=== 实例分割示例 ===")
    model = YOLO('yolov8n-seg.pt')
    images = ['动物.jpg', 'car.png', 'people.png']
    for img in images:
        try:
            results = model(img)
            result = results[0]
            output_name = f"seg_{img}"
            result.save(output_name)
            print(f"{img}: 检测到 {len(result.masks)} 个实例")
        except Exception as e:
            print(f"{img}: 检测失败 - {e}")

# 示例5: 标准目标检测
def detection_example():
    """使用 yolov8n 检测所有图片"""
    print("=== 标准目标检测示例 ===")
    model = YOLO('yolov8n.pt')
    images = ['animal.jpg', 'sport.jpg', 'test.png', '动物.jpg', 'car.png', 'people.png']
    for img in images:
        try:
            results = model(img)
            result = results[0]
            output_name = f"det_{img}"
            result.save(output_name)
            print(f"{img}: 检测到 {len(result.boxes)} 个目标")
        except Exception as e:
            print(f"{img}: 检测失败 - {e}")

if __name__ == "__main__":
    print("YOLO v8 快速演示程序")
    print("确保以下图片文件存在:")
    print("- animal.jpg")
    print("- sport.jpg")
    print("- test.png")
    print("- 动物.jpg")
    print("- car.png")
    print("- people.png")
    print()

    # 运行所有示例
    classification_example()
    pose_example()
    obb_example()
    segmentation_example()
    detection_example()

    print("\n演示完成！检查生成的结果文件。")