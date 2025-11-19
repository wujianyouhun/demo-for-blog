# -*- coding: utf-8 -*-
"""
YOLO水果检测运行脚本
解决PyTorch 2.6+兼容性问题
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.modules
import torch.nn.modules.container
import ultralytics.nn.tasks

# 添加所有必要的安全全局变量
torch.serialization.add_safe_globals([
    ultralytics.nn.tasks.DetectionModel,
    ultralytics.nn.tasks.SegmentationModel,
    nn.Sequential,
    nn.Module,
    nn.ModuleList,
    nn.Conv2d,
    nn.BatchNorm2d,
    nn.ReLU,
    nn.SiLU,
    nn.Upsample,
    nn.Identity,
])

# 现在导入检测脚本
from detection_fruit import FruitDetector

def main():
    print("Starting fruit detection with PyTorch 2.6+ compatibility...")

    try:
        # 创建检测器并运行
        detector = FruitDetector()
        results = detector.process_all_images()

        if results:
            print("\n🎉 Fruit detection completed successfully!")
            print(f"📁 Results saved to: {detector.result_dir}")
        else:
            print("⚠️  No fruits detected")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check internet connection for model download")
        print("2. Ensure ultralytics is properly installed")
        print("3. Verify image files exist in the image folder")

if __name__ == "__main__":
    main()