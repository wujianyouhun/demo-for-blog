# -*- coding: utf-8 -*-
import os
import cv2
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np
from collections import defaultdict
import json
from datetime import datetime
import warnings

# 忽略警告信息
warnings.filterwarnings('ignore')

class FruitDetector:
    def __init__(self):
        # 初始化设备
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # 设置基础路径
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.image_dir = os.path.join(self.base_dir, 'image')
        self.result_dir = os.path.join(self.base_dir, 'result')

        # 加载模型
        try:
            print("Loading YOLO models...")
            self.detector_model = YOLO('yolov8x.pt')  # 目标检测模型
            self.seg_model = YOLO('yolov8n-seg.pt')   # 实例分割模型
            print("Models loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Falling back to CPU models...")
            # 尝试使用CPU模式
            self.device = 'cpu'
            self.detector_model = YOLO('yolov8n.pt')  # 使用更小的模型
            self.seg_model = YOLO('yolov8n-seg.pt')
            print("CPU models loaded!")

        # 扩展的水果类别映射（基于COCO数据集）
        self.fruit_classes = {
            'apple': '苹果',
            'banana': '香蕉',
            'orange': '橙子',
            'broccoli': '西兰花',
            'carrot': '胡萝卜',
            'pizza': '披萨',
            'cake': '蛋糕',
            'sandwich': '三明治',
            'hot dog': '热狗',
            'donut': '甜甜圈',
            'cup': '杯子',
            'fork': '叉子',
            'knife': '刀',
            'spoon': '勺子',
            'bowl': '碗',
            'cell phone': '手机',  # 可能被误识别的水果相关物品
        }

        # 创建结果目录
        os.makedirs(self.result_dir, exist_ok=True)
        print(f"Result directory: {self.result_dir}")
        print(f"Image directory: {self.image_dir}")

    def detect_fruits(self, image_path, save_prefix):
        """
        检测单张图片中的水果
        """
        print(f"Processing image: {image_path}")

        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            print(f"Cannot read image: {image_path}")
            return None

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]

        # 使用检测模型进行目标检测
        detection_results = self.detector_model(img_rgb, device=self.device, conf=0.5)

        # 使用分割模型进行实例分割
        seg_results = self.seg_model(img_rgb, device=self.device, conf=0.5)

        # 统计检测结果
        fruit_count = defaultdict(int)

        # 处理检测结果（边界框）
        detection_img = img.copy()
        for result in detection_results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    class_name = result.names[cls_id]

                    if class_name in self.fruit_classes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])

                        # 绘制边界框
                        cv2.rectangle(detection_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # 添加标签
                        label = f"{self.fruit_classes[class_name]} {confidence:.2f}"
                        cv2.putText(detection_img, label, (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        fruit_count[self.fruit_classes[class_name]] += 1

        # 处理分割结果（掩膜）
        seg_img = img.copy()
        for result in seg_results:
            if result.masks is not None:
                masks = result.masks.data
                boxes = result.boxes
                if boxes is not None:
                    for i, mask in enumerate(masks):
                        cls_id = int(boxes.cls[i])
                        class_name = result.names[cls_id]

                        if class_name in self.fruit_classes:
                            # 将mask转换为二进制图像
                            mask_binary = (mask.cpu().numpy() * 255).astype(np.uint8)
                            mask_binary = cv2.resize(mask_binary, (width, height))

                            # 创建彩色掩膜
                            colored_mask = np.zeros_like(img)
                            colored_mask[mask_binary > 0] = [0, 255, 255]  # 黄色掩膜

                            # 将掩膜叠加到原图
                            alpha = 0.5
                            seg_img = cv2.addWeighted(seg_img, 1, colored_mask, alpha, 0)

        # 保存结果图片
        detection_result_path = os.path.join(self.result_dir, f"{save_prefix}_yolov8x_detection.jpg")
        seg_result_path = os.path.join(self.result_dir, f"{save_prefix}_yolov8x-seg.jpg")

        cv2.imwrite(detection_result_path, detection_img)
        cv2.imwrite(seg_result_path, seg_img)

        print(f"Detection result saved to: {detection_result_path}")
        print(f"Segmentation result saved to: {seg_result_path}")

        return fruit_count

    def process_all_images(self):
        """
        处理image文件夹下的所有图片
        """
        supported_formats = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')

        # 检查image文件夹是否存在
        if not os.path.exists(self.image_dir):
            print(f"Image directory does not exist: {self.image_dir}")
            return

        # 获取所有图片文件
        try:
            image_files = [f for f in os.listdir(self.image_dir)
                          if f.lower().endswith(supported_formats)]
        except Exception as e:
            print(f"Error reading image directory: {e}")
            return

        if not image_files:
            print(f"No image files found in {self.image_dir}")
            print("Supported formats: jpg, jpeg, png")
            return

        print(f"Found {len(image_files)} image files in {self.image_dir}")

        # 显示找到的图片文件
        for i, img_file in enumerate(image_files[:5]):  # 只显示前5个
            print(f"  {i+1}. {img_file}")
        if len(image_files) > 5:
            print(f"  ... and {len(image_files) - 5} more files")

        # 结果统计
        all_results = {}

        for image_file in image_files:
            image_path = os.path.join(self.image_dir, image_file)
            # 获取文件名（不含扩展名）作为前缀
            file_prefix = os.path.splitext(image_file)[0]

            # 检测水果
            fruit_count = self.detect_fruits(image_path, file_prefix)

            if fruit_count:
                all_results[image_file] = dict(fruit_count)
                print(f"Image {image_file} detection results: {dict(fruit_count)}")
            else:
                print(f"No fruits detected in image {image_file}")

        # 保存检测结果到文本文件
        self.save_results_to_text(all_results)

        return all_results

    def save_results_to_text(self, results):
        """
        将检测结果保存到文本文件
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        text_file = os.path.join(self.result_dir, f"fruit_detection_results_{timestamp}.txt")

        with open(text_file, 'w', encoding='utf-8') as f:
            f.write("水果检测结果报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"检测模型: YOLOv8x (目标检测) + YOLOv8x-seg (实例分割)\n")
            f.write(f"使用设备: {self.device}\n")
            f.write("\n")

            total_fruits = defaultdict(int)

            for image_file, fruit_counts in results.items():
                f.write(f"图片: {image_file}\n")
                for fruit, count in fruit_counts.items():
                    f.write(f"  {fruit}: {count}个\n")
                    total_fruits[fruit] += count
                f.write("\n")

            f.write("统计汇总:\n")
            f.write("-" * 20 + "\n")
            for fruit, count in total_fruits.items():
                f.write(f"{fruit}: {count}个\n")

            total_count = sum(total_fruits.values())
            f.write(f"\n总计检测到水果: {total_count}个\n")

        print(f"Detection results saved to: {text_file}")

        # 同时保存为JSON格式
        json_file = os.path.join(self.result_dir, f"fruit_detection_results_{timestamp}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                'detection_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model': 'YOLOv8x + YOLOv8x-seg',
                'device': self.device,
                'results': results,
                'total_fruits': dict(total_fruits),
                'total_count': total_count
            }, f, ensure_ascii=False, indent=2)

        print(f"Detection results (JSON format) saved to: {json_file}")


def main():
    """
    主函数
    """
    print("Starting fruit detection program...")

    try:
        # 创建检测器实例
        detector = FruitDetector()

        # 处理所有图片
        results = detector.process_all_images()

        if results:
            print("\nFruit detection completed!")
            print("Result images saved to result folder")
            print("Detection results saved to text file in result folder")
        else:
            print("No fruits detected")

    except Exception as e:
        print(f"Program error: {e}")
        print("Please check:")
        print("1. Required libraries installed (ultralytics, opencv-python, torch)")
        print("2. GPU available (optional)")
        print("3. Image files exist in image folder")


if __name__ == "__main__":
    main()