# -*- coding: utf-8 -*-
"""
水果检测最终版本 - 兼容PyTorch 2.6+
"""
import os
import cv2
import warnings
warnings.filterwarnings('ignore')

# 临时修改torch.load的默认行为
import torch
original_torch_load = torch.load

def safe_torch_load(f, *args, **kwargs):
    """强制设置weights_only=False"""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(f, *args, **kwargs)

torch.load = safe_torch_load

try:
    from ultralytics import YOLO
    import numpy as np
    from collections import defaultdict
    import json
    from datetime import datetime
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install required packages: pip install ultralytics opencv-python numpy")
    exit(1)

class FruitDetector:
    def __init__(self):
        # 初始化设备
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Using device: {self.device}")

        # 设置基础路径
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.image_dir = os.path.join(self.base_dir, 'image')
        self.result_dir = os.path.join(self.base_dir, 'result')

        # 水果类别映射
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
        }

        # 创建结果目录
        os.makedirs(self.result_dir, exist_ok=True)
        print(f"📁 Result directory: {self.result_dir}")
        print(f"📂 Image directory: {self.image_dir}")

        # 加载模型
        self._load_models()

    def _load_models(self):
        """加载YOLO模型"""
        try:
            print("📥 Loading YOLO models...")

            # 使用小模型确保兼容性和速度
            self.detector_model = YOLO('yolov8n.pt')
            self.seg_model = YOLO('yolov8n-seg.pt')

            print("✅ Models loaded successfully!")

        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print("💡 Please ensure you have internet connection for model download")
            raise

    def process_single_image(self, image_path):
        """处理单张图片"""
        print(f"🔍 Processing: {os.path.basename(image_path)}")

        try:
            # 检测
            detection_results = self.detector_model(image_path, conf=0.5, verbose=False)
            seg_results = self.seg_model(image_path, conf=0.5, verbose=False)

            # 读取原图
            img = cv2.imread(image_path)
            if img is None:
                print(f"❌ Cannot read image: {image_path}")
                return None

            # 统计结果
            fruit_count = defaultdict(int)

            # 处理检测结果
            detection_img = img.copy()
            for result in detection_results:
                if result.boxes is not None:
                    for box in result.boxes:
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

            # 处理分割结果
            seg_img = img.copy()
            for result in seg_results:
                if result.masks is not None and result.boxes is not None:
                    for i, mask in enumerate(result.masks.data):
                        cls_id = int(result.boxes.cls[i])
                        class_name = result.names[cls_id]

                        if class_name in self.fruit_classes:
                            # 处理mask
                            mask_np = mask.cpu().numpy()
                            mask_binary = (mask_np * 255).astype(np.uint8)

                            if len(mask_binary.shape) == 3:
                                mask_binary = mask_binary[0]

                            mask_binary = cv2.resize(mask_binary, (img.shape[1], img.shape[0]))

                            # 创建彩色掩膜
                            colored_mask = np.zeros_like(img)
                            colored_mask[mask_binary > 0] = [0, 255, 255]  # 黄色

                            # 叠加掩膜
                            alpha = 0.5
                            seg_img = cv2.addWeighted(seg_img, 1, colored_mask, alpha, 0)

            # 保存结果
            save_name = os.path.splitext(os.path.basename(image_path))[0]
            detection_path = os.path.join(self.result_dir, f"{save_name}_detection.jpg")
            seg_path = os.path.join(self.result_dir, f"{save_name}_segmentation.jpg")

            cv2.imwrite(detection_path, detection_img)
            cv2.imwrite(seg_path, seg_img)

            print(f"✅ Detection result: {detection_path}")
            print(f"✅ Segmentation result: {seg_path}")

            return dict(fruit_count)

        except Exception as e:
            print(f"❌ Error processing image: {e}")
            return None

    def process_all_images(self):
        """处理所有图片"""
        print("\n🎯 Starting batch processing...")

        supported_formats = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')

        # 检查文件夹
        if not os.path.exists(self.image_dir):
            print(f"❌ Image directory not found: {self.image_dir}")
            return

        # 获取图片列表
        try:
            image_files = [f for f in os.listdir(self.image_dir)
                          if f.lower().endswith(supported_formats)]
        except Exception as e:
            print(f"❌ Error reading directory: {e}")
            return

        if not image_files:
            print(f"❌ No images found in {self.image_dir}")
            return

        print(f"📸 Found {len(image_files)} images")

        # 处理每张图片
        all_results = {}
        for image_file in image_files:
            image_path = os.path.join(self.image_dir, image_file)
            result = self.process_single_image(image_path)

            if result:
                all_results[image_file] = result
                print(f"📊 {image_file}: {result}")
            else:
                print(f"⚠️  No fruits detected in {image_file}")

        # 保存汇总报告
        self._save_report(all_results)
        return all_results

    def _save_report(self, results):
        """保存检测报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 文本报告
        text_file = os.path.join(self.result_dir, f"detection_report_{timestamp}.txt")

        with open(text_file, 'w', encoding='utf-8') as f:
            f.write("🍎 水果检测报告 🍎\n")
            f.write("=" * 50 + "\n")
            f.write(f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"检测设备: {self.device}\n")
            f.write(f"处理图片数量: {len(results)}\n\n")

            total_fruits = defaultdict(int)

            for image_file, fruits in results.items():
                f.write(f"📷 图片: {image_file}\n")
                for fruit, count in fruits.items():
                    f.write(f"   {fruit}: {count}个\n")
                    total_fruits[fruit] += count
                f.write("\n")

            f.write("📈 统计汇总:\n")
            f.write("-" * 20 + "\n")
            for fruit, count in total_fruits.items():
                f.write(f"{fruit}: {count}个\n")

            total = sum(total_fruits.values())
            f.write(f"\n🎯 总计检测: {total}个水果\n")

        # JSON报告
        json_file = os.path.join(self.result_dir, f"detection_report_{timestamp}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                'detection_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'device': self.device,
                'total_images': len(results),
                'results': results,
                'summary': dict(total_fruits),
                'total_fruits': sum(total_fruits.values())
            }, f, ensure_ascii=False, indent=2)

        print(f"\n📋 Report saved:")
        print(f"   Text: {text_file}")
        print(f"   JSON: {json_file}")


def main():
    """主函数"""
    print("🍎🍌🍊 YOLO水果检测系统 🍎🍌🍊")
    print("=" * 50)

    try:
        detector = FruitDetector()
        results = detector.process_all_images()

        if results:
            print("\n🎉 检测完成！")
            print(f"📁 所有结果已保存到: {detector.result_dir}")
        else:
            print("\n⚠️  未检测到任何水果")

    except Exception as e:
        print(f"\n❌ 程序错误: {e}")
        print("\n🔧 解决方案:")
        print("1. 检查网络连接（需要下载模型）")
        print("2. 确保安装了ultralytics: pip install ultralytics")
        print("3. 确保image文件夹中有图片文件")

    finally:
        # 恢复原始的torch.load
        torch.load = original_torch_load


if __name__ == "__main__":
    main()