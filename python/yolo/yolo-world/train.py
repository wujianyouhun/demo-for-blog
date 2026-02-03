"""
YOLO-World 自定义训练脚本

使用方法:
1. 准备数据集（YOLO 格式）
2. 修改下面的配置参数
3. 运行: python train.py
"""

from ultralytics import YOLO
import torch
from torch.nn.modules.container import Sequential

# 设置 PyTorch 安全加载
torch.serialization.add_safe_globals([Sequential])

# ================== 训练配置 ==================
# 基础模型路径
BASE_MODEL = "yolov8m-worldv2.pt"

# 自定义类别
CLASSES = ["person", "car", "truck", "bus"]

# 数据集路径（YOLO 格式）
# 需要准备 data.yaml 文件
DATA_YAML = "data.yaml"

# 训练参数
EPOCHS = 100
BATCH_SIZE = 16
IMAGE_SIZE = 640
DEVICE = "0" if torch.cuda.is_available() else "cpu"  # 0 表示使用第一个 GPU

# ================================================

def main():
    print("🚀 YOLO-World 自定义训练开始")
    print(f"📋 检测类别: {CLASSES}")
    print(f"🔧 设备: {'GPU' if DEVICE != 'cpu' else 'CPU'}")

    # 加载基础模型
    print(f"\n📥 加载基础模型: {BASE_MODEL}")
    model = YOLO(BASE_MODEL)

    # 设置自定义类别
    model.set_classes(CLASSES)
    print("✅ 类别配置完成")

    # 开始训练
    print(f"\n🎯 开始训练...")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Image Size: {IMAGE_SIZE}")

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        project="runs/train",
        name="custom_model",
        patience=20,  # 早停机制
        save=True,
        plots=True,
        verbose=True
    )

    print("\n✅ 训练完成！")
    print(f"📁 最佳模型保存在: runs/train/custom_model/weights/best.pt")

if __name__ == "__main__":
    main()
