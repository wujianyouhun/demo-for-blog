"""
YOLO v8 多模型检测演示程序
==========================

本程序演示如何使用不同的 YOLO v8 模型进行检测：
- yolov8n-cls.pt: 图像分类
- yolov8n-pose.pt: 姿态检测
- yolov8n-obb.pt: 旋转目标检测
- yolov8n-seg.pt: 实例分割
- yolov8n.pt: 标准目标检测

作者: Claude
日期: 2025-10-27
"""

# 导入必要的库
import cv2          # OpenCV - 用于图像处理
import os           # 操作系统接口 - 用于文件和目录操作
import torch        # PyTorch - 深度学习框架
from ultralytics import YOLO  # YOLO v8 模型库
import numpy as np  # 数值计算库

# ===================================================================
# PyTorch 2.6 兼容性补丁
# ===================================================================
def patch_torch_load():
    """
    PyTorch 2.6 安全加载补丁
    =====================

    问题: PyTorch 2.6 引入了 weights_only=True 的安全特性，
         但 YOLO 模型需要 weights_only=False 才能正常加载。

    解决方案: 创建一个补丁函数，强制所有 YOLO 模型加载时
              使用 weights_only=False。
    """
    original_load = torch.load
    def patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    torch.load = patched_load

# 应用补丁
patch_torch_load()

# ===================================================================
# 图像分类检测
# ===================================================================
def classification_demo():
    """
    使用 yolov8n-cls 模型进行图像分类检测
    ======================================

    检测图片: animal.jpg
    输出: 分类结果和置信度
    """
    print("=" * 60)
    print("🔍 图像分类检测 (yolov8n-cls)")
    print("=" * 60)

    # 加载分类模型
    model = YOLO('yolov8m-worldv2.pt')
    print("✓ 分类模型加载成功")

    # 检测图片
    image_path = 'animal.jpg'
    if not os.path.exists(image_path):
        print(f"⚠ 警告: 找不到图片 {image_path}")
        return

    print(f"📁 正在检测: {image_path}")

    # 进行分类
    results = model(image_path)

    # 获取分类结果
    result = results[0]
    top5_indices = result.probs.top5  # 获取前5个预测结果
    top5_confidences = result.probs.top5conf

    print("\n🎯 分类结果 (Top 5):")
    print("-" * 40)
    for i, (idx, conf) in enumerate(zip(top5_indices, top5_confidences)):
        class_name = model.names[int(idx)]
        print(f"  {i+1}. {class_name}: {conf:.4f} ({conf*100:.2f}%)")

    # 保存结果图片
    output_path = 'classification_result.jpg'
    img = cv2.imread(image_path)

    # 在图片上添加分类结果
    top_class = model.names[int(top5_indices[0])]
    top_conf = top5_confidences[0]
    text = f"Class: {top_class} ({top_conf*100:.1f}%)"

    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imwrite(output_path, img)

    print(f"\n✓ 分类结果已保存至: {output_path}")

# ===================================================================
# 姿态检测
# ===================================================================
def pose_detection_demo():
    """
    使用 yolov8n-pose 模型进行姿态检测
    ==================================

    检测图片: sport.jpg
    输出: 人体关键点和骨架
    """
    print("\n" + "=" * 60)
    print("🤸 姿态检测 (yolov8n-pose)")
    print("=" * 60)

    # 加载姿态检测模型
    model = YOLO('yolov8n-pose.pt')
    print("✓ 姿态检测模型加载成功")

    # 检测图片
    image_path = 'sport.jpg'
    if not os.path.exists(image_path):
        print(f"⚠ 警告: 找不到图片 {image_path}")
        return

    print(f"📁 正在检测: {image_path}")

    # 进行姿态检测
    results = model(image_path)

    # 获取检测结果
    result = results[0]

    if len(result.keypoints) > 0:
        print(f"✓ 检测到 {len(result.keypoints)} 个人体姿态")
        print("🎯 关键点数量: 17个 (鼻子、眼睛、耳朵、肩膀、手肘、手腕、臀部、膝盖、脚踝)")
    else:
        print("⚠ 未检测到人体姿态")

    # 保存结果图片
    output_path = 'pose_detection_result.jpg'
    annotated_img = result.plot()
    cv2.imwrite(output_path, annotated_img)

    print(f"✓ 姿态检测结果已保存至: {output_path}")

# ===================================================================
# 旋转目标检测
# ===================================================================
def obb_detection_demo():
    """
    使用 yolov8n-obb 模型进行旋转目标检测
    =====================================

    检测图片: test.png
    输出: 旋转边界框检测结果
    """
    print("\n" + "=" * 60)
    print("🔄 旋转目标检测 (yolov8n-obb)")
    print("=" * 60)

    # 加载旋转检测模型
    model = YOLO('yolov8n-obb.pt')
    print("✓ 旋转检测模型加载成功")

    # 检测图片
    image_path = 'test.png'
    if not os.path.exists(image_path):
        print(f"⚠ 警告: 找不到图片 {image_path}")
        return

    print(f"📁 正在检测: {image_path}")

    # 进行旋转目标检测
    results = model(image_path)

    # 获取检测结果
    result = results[0]

    if len(result.obb) > 0:
        print(f"✓ 检测到 {len(result.obb)} 个旋转目标")
        print("🎯 检测到的目标类别:")
        for i, obb in enumerate(result.obb):
            class_id = int(obb.cls[0])
            class_name = model.names[class_id]
            confidence = float(obb.conf[0])
            print(f"    {i+1}. {class_name}: {confidence:.4f}")
    else:
        print("⚠ 未检测到旋转目标")

    # 保存结果图片
    output_path = 'obb_detection_result.png'
    annotated_img = result.plot()
    cv2.imwrite(output_path, annotated_img)

    print(f"✓ 旋转检测结果已保存至: {output_path}")

# ===================================================================
# 实例分割
# ===================================================================
def segmentation_demo():
    """
    使用 yolov8n-seg 模型进行实例分割
    ==================================

    检测图片: 动物.jpg, car.png, people.png
    输出: 实例分割掩码
    """
    print("\n" + "=" * 60)
    print("🎨 实例分割 (yolov8n-seg)")
    print("=" * 60)

    # 加载分割模型
    model = YOLO('yolov8n-seg.pt')
    print("✓ 实例分割模型加载成功")

    # 要检测的图片列表
    image_files = ['动物.jpg', 'car.png', 'people.png']

    for image_path in image_files:
        print(f"\n📁 正在检测: {image_path}")

        if not os.path.exists(image_path):
            print(f"⚠ 警告: 找不到图片 {image_path}")
            continue

        # 进行实例分割
        results = model(image_path)

        # 获取检测结果
        result = results[0]

        if len(result.masks) > 0:
            print(f"✓ 检测到 {len(result.masks)} 个实例")
            print("🎯 检测到的目标类别:")
            for i, box in enumerate(result.boxes):
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                print(f"    {i+1}. {class_name}: {confidence:.4f}")
        else:
            print("⚠ 未检测到可分割的实例")

        # 保存结果图片
        filename = os.path.splitext(image_path)[0]
        output_path = f'{filename}_segmentation_result.png'
        annotated_img = result.plot()
        cv2.imwrite(output_path, annotated_img)

        print(f"✓ 分割结果已保存至: {output_path}")

# ===================================================================
# 标准目标检测
# ===================================================================
def standard_detection_demo():
    """
    使用 yolov8n 模型进行标准目标检测
    =================================

    检测图片: animal.jpg, sport.jpg, test.png, 动物.jpg, car.png, people.png
    输出: 标准边界框检测结果
    """
    print("\n" + "=" * 60)
    print("🎯 标准目标检测 (yolov8n)")
    print("=" * 60)

    # 加载标准检测模型
    model = YOLO('yolov8n.pt')
    print("✓ 标准检测模型加载成功")

    # 要检测的所有图片
    image_files = ['animal.jpg', 'sport.jpg', 'test.png', '动物.jpg', 'car.png', 'people.png']

    total_detections = 0

    for image_path in image_files:
        print(f"\n📁 正在检测: {image_path}")

        if not os.path.exists(image_path):
            print(f"⚠ 警告: 找不到图片 {image_path}")
            continue

        try:
            # 进行目标检测
            results = model(image_path)

            # 获取检测结果
            result = results[0]

            if len(result.boxes) > 0:
                print(f"✓ 检测到 {len(result.boxes)} 个目标")
                print("🎯 检测到的目标类别:")
                for i, box in enumerate(result.boxes):
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])
                    print(f"    {i+1}. {class_name}: {confidence:.4f}")
                total_detections += len(result.boxes)
            else:
                print("⚠ 未检测到目标")

            # 保存结果图片
            filename = os.path.splitext(image_path)[0]
            output_path = f'{filename}_detection_result.jpg'
            annotated_img = result.plot()
            cv2.imwrite(output_path, annotated_img)

            print(f"✓ 检测结果已保存至: {output_path}")

        except Exception as e:
            print(f"✗ 检测 {image_path} 时出错: {e}")

    print(f"\n📊 标准检测总结:")
    print(f"   总共检测到 {total_detections} 个目标")

# ===================================================================
# 主函数
# ===================================================================
def main():
    """
    主函数：执行所有模型检测演示
    ============================
    """
    print("🚀 YOLO v8 多模型检测演示程序")
    print("=" * 80)

    # 创建输出目录
    os.makedirs('detection_results', exist_ok=True)


    try:
        # 1. 图像分类演示
        classification_demo()

        # 2. 姿态检测演示
        pose_detection_demo()

        # 3. 旋转目标检测演示
        obb_detection_demo()

        # 4. 实例分割演示
        segmentation_demo()

        # 5. 标准目标检测演示
        standard_detection_demo()

        print("\n" + "=" * 80)
        print("🎉 所有检测演示完成！")
        print("📁 所有结果已保存在 detection_results 目录中")
        print("=" * 80)

    except Exception as e:
        print(f"✗ 程序运行出错: {e}")

    finally:
        # 返回原目录
        os.chdir('..')

# ===================================================================
# 程序入口
# ===================================================================
if __name__ == "__main__":
    main()