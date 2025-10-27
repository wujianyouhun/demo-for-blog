"""
YOLO v8 多模型完整示例集
======================

本文件包含了各种 YOLO v8 模型的使用示例：

1. yolov8n-cls.pt: 图像分类模型
   - 输入: animal.jpg
   - 输出: 分类结果和置信度

2. yolov8n-pose.pt: 姿态检测模型
   - 输入: sport.jpg
   - 输出: 人体关键点和骨架

3. yolov8n-obb.pt: 旋转目标检测模型
   - 输入: test.png
   - 输出: 旋转边界框

4. yolov8n-seg.pt: 实例分割模型
   - 输入: 动物.jpg, car.png, people.png
   - 输出: 实例分割掩码

5. yolov8n.pt: 标准目标检测模型
   - 输入: 所有上述图片
   - 输出: 标准边界框

作者: Claude
日期: 2025-10-27
"""

# ===================================================================
# 必要的导入
# ===================================================================
import cv2
import os
import torch
import numpy as np
from ultralytics import YOLO

# ===================================================================
# PyTorch 2.6 兼容性补丁
# ===================================================================
def patch_torch_load():
    """
    PyTorch 2.6 兼容性补丁
    确保YOLO模型能正常加载
    """
    try:
        original_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        torch.load = patched_load
        print("✓ PyTorch 2.6 兼容性补丁已应用")
    except Exception as e:
        print(f"⚠ PyTorch 补丁应用失败: {e}")

# 应用补丁
patch_torch_load()

# ===================================================================
# 示例 1: 图像分类 (yolov8n-cls)
# ===================================================================
def example_1_classification():
    """
    图像分类示例
    ============

    模型: yolov8n-cls.pt
    功能: 对图像进行分类，输出类别和置信度
    输入: animal.jpg
    """

    print("=" * 60)
    print("示例 1: 图像分类 (yolov8n-cls)")
    print("=" * 60)

    try:
        # 1. 加载分类模型
        print("🔄 加载 yolov8n-cls 模型...")
        model = YOLO('yolov8n-cls.pt')
        print("✓ 模型加载成功")

        # 2. 指定输入图片
        input_image = 'animal.jpg'

        if not os.path.exists(input_image):
            print(f"❌ 错误: 找不到图片 {input_image}")
            print(f"   请确保 {input_image} 文件存在于当前目录")
            return

        # 3. 进行分类
        print(f"🔄 正在对 {input_image} 进行分类...")
        results = model(input_image)

        # 4. 处理结果
        result = results[0]

        # 获取Top-5预测结果
        if hasattr(result, 'probs') and result.probs is not None:
            top5_indices = result.probs.top5
            top5_confidences = result.probs.top5conf

            print("\n🎯 分类结果 (Top 5):")
            print("-" * 50)
            for i, (idx, conf) in enumerate(zip(top5_indices, top5_confidences)):
                class_name = model.names[int(idx)]
                print(f"  {i+1}. {class_name}: {conf:.4f} ({conf*100:.2f}%)")

            # 5. 保存结果图片
            img = cv2.imread(input_image)
            if img is not None:
                top_class = model.names[int(top5_indices[0])]
                top_conf = top5_confidences[0]

                # 在图片上添加结果
                text = f"Top-1: {top_class} ({top_conf*100:.1f}%)"
                cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                output_path = 'classification_animal_result.jpg'
                cv2.imwrite(output_path, img)
                print(f"\n✓ 结果已保存: {output_path}")
            else:
                print("❌ 无法读取图片")
        else:
            print("❌ 分类结果为空或格式错误")

    except Exception as e:
        print(f"❌ 分类示例运行出错: {e}")

# ===================================================================
# 示例 2: 姿态检测 (yolov8n-pose)
# ===================================================================
def example_2_pose_detection():
    """
    姿态检测示例
    ============

    模型: yolov8n-pose.pt
    功能: 检测人体关键点和骨架
    输入: sport.jpg
    """

    print("\n" + "=" * 60)
    print("示例 2: 姿态检测 (yolov8n-pose)")
    print("=" * 60)

    try:
        # 1. 加载姿态检测模型
        print("🔄 加载 yolov8n-pose 模型...")
        model = YOLO('yolov8n-pose.pt')
        print("✓ 模型加载成功")

        # 2. 指定输入图片
        input_image = 'sport.jpg'

        if not os.path.exists(input_image):
            print(f"❌ 错误: 找不到图片 {input_image}")
            print(f"   请确保 {input_image} 文件存在于当前目录")
            return

        # 3. 进行姿态检测
        print(f"🔄 正在对 {input_image} 进行姿态检测...")
        results = model(input_image)

        # 4. 处理结果
        result = results[0]

        if hasattr(result, 'keypoints') and len(result.keypoints) > 0:
            print(f"✓ 检测到 {len(result.keypoints)} 个人体")
            print("📊 每个人体的关键点数量: 17个")
            print("   (鼻子、眼睛、耳朵、肩膀、手肘、手腕、臀部、膝盖、脚踝)")
        else:
            print("⚠ 未检测到人体")

        # 5. 保存结果图片
        output_path = 'pose_sport_result.jpg'
        result.save(output_path)
        print(f"✓ 结果已保存: {output_path}")

    except Exception as e:
        print(f"❌ 姿态检测示例运行出错: {e}")

# ===================================================================
# 示例 3: 旋转目标检测 (yolov8n-obb)
# ===================================================================
def example_3_obb_detection():
    """
    旋转目标检测示例
    ================

    模型: yolov8n-obb.pt
    功能: 检测旋转的物体，输出旋转边界框
    输入: test.png
    """

    print("\n" + "=" * 60)
    print("示例 3: 旋转目标检测 (yolov8n-obb)")
    print("=" * 60)

    try:
        # 1. 加载旋转检测模型
        print("🔄 加载 yolov8n-obb 模型...")
        model = YOLO('yolov8n-obb.pt')
        print("✓ 模型加载成功")

        # 2. 指定输入图片
        input_image = 'test.png'

        if not os.path.exists(input_image):
            print(f"❌ 错误: 找不到图片 {input_image}")
            print(f"   请确保 {input_image} 文件存在于当前目录")
            return

        # 3. 进行旋转目标检测
        print(f"🔄 正在对 {input_image} 进行旋转目标检测...")
        results = model(input_image)

        # 4. 处理结果
        result = results[0]

        if hasattr(result, 'obb') and len(result.obb) > 0:
            print(f"✓ 检测到 {len(result.obb)} 个旋转目标")
            print("🎯 检测详情:")
            for i, obb in enumerate(result.obb):
                class_id = int(obb.cls[0])
                class_name = model.names[class_id]
                confidence = float(obb.conf[0])
                print(f"  目标 {i+1}: {class_name} (置信度: {confidence:.4f})")
        else:
            print("⚠ 未检测到旋转目标")

        # 5. 保存结果图片
        output_path = 'obb_test_result.png'
        result.save(output_path)
        print(f"✓ 结果已保存: {output_path}")

    except Exception as e:
        print(f"❌ 旋转目标检测示例运行出错: {e}")

# ===================================================================
# 示例 4: 实例分割 (yolov8n-seg)
# ===================================================================
def example_4_segmentation():
    """
    实例分割示例
    ============

    模型: yolov8n-seg.pt
    功能: 对物体进行实例分割，输出精确掩码
    输入: 动物.jpg, car.png, people.png
    """

    print("\n" + "=" * 60)
    print("示例 4: 实例分割 (yolov8n-seg)")
    print("=" * 60)

    try:
        # 1. 加载分割模型
        print("🔄 加载 yolov8n-seg 模型...")
        model = YOLO('yolov8n-seg.pt')
        print("✓ 模型加载成功")

        # 2. 指定输入图片列表
        input_images = ['动物.jpg', 'car.png', 'people.png']

        for input_image in input_images:
            print(f"\n🔄 正在处理 {input_image}...")

            if not os.path.exists(input_image):
                print(f"❌ 错误: 找不到图片 {input_image}")
                continue

            try:
                # 3. 进行实例分割
                results = model(input_image)
                result = results[0]

                # 4. 处理结果
                if hasattr(result, 'masks') and len(result.masks) > 0:
                    print(f"✓ 检测到 {len(result.masks)} 个实例")
                    print("🎯 实例详情:")
                    for i, (box, mask) in enumerate(zip(result.boxes, result.masks)):
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        confidence = float(box.conf[0])
                        print(f"  实例 {i+1}: {class_name} (置信度: {confidence:.4f})")
                else:
                    print("⚠ 未检测到可分割的实例")

                # 5. 保存结果图片
                filename = os.path.splitext(input_image)[0]
                output_path = f'{filename}_segmentation_result.png'
                result.save(output_path)
                print(f"✓ 结果已保存: {output_path}")

            except Exception as e:
                print(f"❌ 处理 {input_image} 时出错: {e}")

    except Exception as e:
        print(f"❌ 实例分割示例运行出错: {e}")

# ===================================================================
# 示例 5: 标准目标检测 (yolov8n)
# ===================================================================
def example_5_standard_detection():
    """
    标准目标检测示例
    ================

    模型: yolov8n.pt
    功能: 标准目标检测，输出边界框
    输入: animal.jpg, sport.jpg, test.png, 动物.jpg, car.png, people.png
    """

    print("\n" + "=" * 60)
    print("示例 5: 标准目标检测 (yolov8n)")
    print("=" * 60)

    try:
        # 1. 加载标准检测模型
        print("🔄 加载 yolov8n 模型...")
        model = YOLO('yolov8n.pt')
        print("✓ 模型加载成功")

        # 2. 指定输入图片列表
        input_images = ['animal.jpg', 'sport.jpg', 'test.png', '动物.jpg', 'car.png', 'people.png']
        total_detections = 0

        for input_image in input_images:
            print(f"\n🔄 正在处理 {input_image}...")

            if not os.path.exists(input_image):
                print(f"❌ 错误: 找不到图片 {input_image}")
                continue

            try:
                # 3. 进行目标检测
                results = model(input_image)
                result = results[0]

                # 4. 处理结果
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    print(f"✓ 检测到 {len(result.boxes)} 个目标")
                    print("🎯 目标详情:")
                    for i, box in enumerate(result.boxes):
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        confidence = float(box.conf[0])
                        print(f"  目标 {i+1}: {class_name} (置信度: {confidence:.4f})")
                    total_detections += len(result.boxes)
                else:
                    print("⚠ 未检测到目标")

                # 5. 保存结果图片
                filename = os.path.splitext(input_image)[0]
                output_path = f'{filename}_detection_result.jpg'
                result.save(output_path)
                print(f"✓ 结果已保存: {output_path}")

            except Exception as e:
                print(f"❌ 处理 {input_image} 时出错: {e}")

        print(f"\n📊 检测总结: 总共检测到 {total_detections} 个目标")

    except Exception as e:
        print(f"❌ 标准目标检测示例运行出错: {e}")

# ===================================================================
# 运行所有示例
# ===================================================================
def run_all_examples():
    """
    运行所有示例
    """
    print("🚀 YOLO v8 多模型检测示例")
    print("=" * 80)

    examples = [
        ("图像分类", example_1_classification),
        ("姿态检测", example_2_pose_detection),
        ("旋转目标检测", example_3_obb_detection),
        ("实例分割", example_4_segmentation),
        ("标准目标检测", example_5_standard_detection)
    ]

    try:
        for i, (name, func) in enumerate(examples, 1):
            print(f"\n📋 运行示例 {i}: {name}")
            print("-" * 40)
            try:
                func()
                print(f"✓ 示例 {i}: {name} 完成")
            except Exception as e:
                print(f"❌ 示例 {i}: {name} 失败 - {e}")
                # 继续运行下一个示例
                continue

        print("\n" + "=" * 80)
        print("🎉 所有示例运行完成！")
        print("📁 请检查生成的结果文件")
        print("=" * 80)

    except Exception as e:
        print(f"❌ 程序运行出错: {e}")

# ===================================================================
# 主程序入口
# ===================================================================
if __name__ == "__main__":
    # 检查必要的图片文件
    required_images = ['animal.jpg', 'sport.jpg', 'test.png', '动物.jpg', 'car.png', 'people.png']

    print("📋 YOLO v8 多模型检测示例")
    print("=" * 80)

    try:
        # 检查图片文件
        missing_images = []
        for img in required_images:
            if not os.path.exists(img):
                missing_images.append(img)

        if missing_images:
            print("❌ 缺少以下图片文件:")
            for img in missing_images:
                print(f"   - {img}")
            print("\n请确保所有必需的图片文件都存在，然后重新运行程序。")
            print("\n💡 提示：")
            print("   - 程序会继续运行，但某些示例可能会失败")
            print("   - 您可以准备相应的图片文件后重新运行")
        else:
            print("✓ 所有必需的图片文件都已存在")

        print()

        # 运行所有示例
        run_all_examples()

    except Exception as e:
        print(f"❌ 主程序运行出错: {e}")
        print("\n请检查以下事项：")
        print("1. Python 版本是否为 3.8+")
        print("2. 是否已安装 ultralytics 库")
        print("3. 网络连接是否正常（用于下载模型）")
        print("4. 图片文件路径是否正确")