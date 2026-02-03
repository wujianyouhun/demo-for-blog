"""
YOLO目标检测脚本
支持GPU/CPU自动切换，检测指定图片中的目标
"""

import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
import os

# 为 PyTorch 2.6+ 添加安全全局变量,允许加载 ultralytics 模型
torch.serialization.add_safe_globals([DetectionModel])


def check_device():
    """
    检测并返回可用的计算设备（GPU优先）
    Returns:
        torch.device: 可用的计算设备
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ 使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU数量: {torch.cuda.device_count()}")
        print(f"  显存大小: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = torch.device("cpu")
        print("✓ 使用CPU (未检测到可用的GPU)")
    return device


def detect_objects(image_path, model_name='yolov8n.pt'):
    """
    使用YOLO模型检测图片中的目标

    Args:
        image_path (str): 待检测图片路径
        model_name (str): YOLO模型名称，默认yolov8n.pt
    """
    # 检查图片是否存在
    if not os.path.exists(image_path):
        print(f"错误: 图片文件不存在 - {image_path}")
        return

    # 检测设备
    device = check_device()

    print(f"\n正在加载YOLO模型: {model_name}")

    try:
        # 加载YOLO模型
        model = YOLO(model_name)

        # 将模型移动到指定设备
        model.to(device)

        print(f"\n开始检测图片: {image_path}")
        print("-" * 60)

        # 执行检测
        results = model.predict(
            image_path,
            device=device,
            save=True,  # 保存检测结果
            save_txt=True,  # 保存检测结果为txt格式
            conf=0.25,  # 置信度阈值
            project='images',
            name='results'
        )

        # 打印检测结果
        for i, result in enumerate(results):
            print(f"\n检测结果:")
            print(f"  检测到 {len(result.boxes)} 个目标")

            # 打印每个检测到的目标信息
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()

                print(f"\n  目标 {len(result.boxes) - list(result.boxes).index(box)}:")
                print(f"    类别: {cls_name} (ID: {cls_id})")
                print(f"    置信度: {conf:.2%}")
                print(f"    位置: [{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]")

        print(f"\n✓ 检测完成!")
        print(f"✓ 结果已保存至: images/results")

    except Exception as e:
        print(f"\n错误: {str(e)}")
        print("提示: 请确保已安装ultralytics库，首次运行会自动下载模型")


def main():
    """主函数"""
    # 图片路径
    image_path = "images/357548.jpg"

    # 可选模型:
    # YOLOv8: yolov8n.pt(最小最快), yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt(最大最准)
    # YOLOv11: yolo11n.pt(最小最快), yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt(最大最准)
    model_name = 'yolov11n.pt'  # 临时使用v8,等待ultralytics安装

    print("=" * 60)
    print("YOLOv11目标检测程序")
    print("=" * 60)

    detect_objects(image_path, model_name)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
