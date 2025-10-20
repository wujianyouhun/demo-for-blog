#!/usr/bin/env python3
import sys
try:
    from ultralytics import YOLO
    import cv2
    import torch
    import numpy as np
    
    print("✅ 核心依赖检查通过")
    
    # PyTorch 信息
    print(f"PyTorch版本: {torch.__version__}")
    device = 'CUDA' if torch.cuda.is_available() else 'CPU'
    print(f"推理设备: {device}")
    
    # Ultralytics 版本
    import ultralytics
    print(f"Ultralytics: {ultralytics.__version__}")
    
    # 测试模型加载（免费预训练模型）
    print("📥 加载 YOLOv8n 模型...")
    model = YOLO('yolov8n.pt')
    
    # 测试推理
    test_img = np.zeros((640, 640, 3), dtype=np.uint8)
    results = model(test_img, verbose=False)
    
    print("✅ 模型推理测试成功")
    print(f"检测到 {len(results[0].boxes)} 个对象")
    print(f"支持 {len(model.names)} 个类别")
    
    # RTSP连接测试（可选）
    try:
        cap = cv2.VideoCapture("rtsp://admin:Aa147258@192.168.109.213")
        if cap.isOpened():
            print("✅ RTSP流连接正常")
            ret, frame = cap.read()
            if ret:
                print(f"帧尺寸: {frame.shape}")
            cap.release()
        else:
            print("⚠️ RTSP连接测试跳过（网络问题正常）")
    except:
        print("⚠️ RTSP测试跳过")
    
    print("\n🚀 环境就绪，可运行监控程序!")
    
except ImportError as e:
    print(f"❌ 缺少依赖: {e}")
    print("请执行: pip install ultralytics opencv-python torch")
except Exception as e:
    print(f"❌ 安装验证失败: {e}")