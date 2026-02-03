import torch
from torch.nn.modules.container import Sequential
torch.serialization.add_safe_globals([Sequential])

import cv2
import time
import math
import os
import requests
from ultralytics import YOLO
from collections import defaultdict

# 设置 PyTorch 允许加载自定义类的模型（兼容 PyTorch 2.6+）
try:
    from ultralytics.nn.tasks import WorldModel
    torch.serialization.add_safe_globals([WorldModel])
except (ImportError, AttributeError):
    pass

# ================== 配置区 ==================
RTSP_URL = "rtsp://admin:Aa147258@192.168.109.213"

MODEL_PATH = "yolov8m-worldv2.pt"   # 你本地的 YOLO-World 模型
CLASSES = ["person", "car", "truck", "bus"]

# YOLO-World 官方模型下载链接
MODEL_DOWNLOAD_URL = "https://github.com/AILab-CVC/YOLO-World/releases/download/v1.0/yolov8m-worldv2.pt"

def download_model(url, dest_path):
    """下载模型文件"""
    print(f"📥 正在下载模型从 {url}")
    print(f"💾 保存到 {dest_path}")
    print("⏳ 请稍候，这可能需要几分钟...")

    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0

        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r📊 下载进度: {percent:.1f}%", end='', flush=True)

        print(f"\n✅ 模型下载成功！")
        return True

    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False

def train_custom_model(base_model_path, classes, epochs=10):
    """使用自定义类别训练模型"""
    print(f"\n🎯 开始训练自定义模型")
    print(f"📋 自定义类别: {classes}")

    # 加载基础模型
    model = YOLO(base_model_path)

    # 设置自定义类别
    model.set_classes(classes)

    # 保存模型（这会自动适配新的类别）
    custom_model_path = base_model_path.replace('.pt', '_custom.pt')
    model.save(custom_model_path)

    print(f"✅ 自定义模型已保存到: {custom_model_path}")
    print(f"💡 提示: 如需完整训练，请准备标注数据集并使用 model.train()")

    return custom_model_path

# 自动下载或准备模型
if not os.path.exists(MODEL_PATH):
    print(f"⚠️  模型文件 {MODEL_PATH} 不存在")
    user_input = input("是否自动下载预训练模型? (y/n): ").strip().lower()

    if user_input == 'y':
        if download_model(MODEL_DOWNLOAD_URL, MODEL_PATH):
            print("✅ 模型准备完成！")
        else:
            print("❌ 自动下载失败，请手动下载:")
            print(f"   1. 访问: {MODEL_DOWNLOAD_URL}")
            print(f"   2. 保存为: {MODEL_PATH}")
            exit(1)
    else:
        print("❌ 需要模型文件才能运行")
        exit(1)

DURATION_THRESHOLD = {
    "person": 30,        # 秒
    "car": 600,
    "truck": 600,
    "bus": 600
}

MOVE_THRESHOLD = 0.05    # 归一化距离
ALERT_COOLDOWN = 60      # 秒

# ===========================================

# 加载并配置模型
print(f"\n🔧 正在加载模型 {MODEL_PATH}...")
model = YOLO(MODEL_PATH)

# 设置自定义类别
print(f"📋 配置检测类别: {CLASSES}")
model.set_classes(CLASSES)

# 询问是否训练自定义模型
custom_model_path = MODEL_PATH.replace('.pt', '_custom.pt')
if not os.path.exists(custom_model_path):
    train_input = input("\n是否生成自定义类别模型? (推荐) (y/n): ").strip().lower()
    if train_input == 'y':
        custom_model_path = train_custom_model(MODEL_PATH, CLASSES)
        MODEL_PATH = custom_model_path
        model = YOLO(MODEL_PATH)
        print("✅ 使用自定义模型")
    else:
        print("💡 使用默认模型配置")
else:
    print(f"✅ 发现已存在的自定义模型: {custom_model_path}")
    use_custom = input("是否使用自定义模型? (y/n): ").strip().lower()
    if use_custom == 'y':
        MODEL_PATH = custom_model_path
        model = YOLO(MODEL_PATH)
        print("✅ 已加载自定义模型")

cap = cv2.VideoCapture(RTSP_URL)

track_states = {}

def norm_center(box, w, h):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    return cx, cy

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

print("🚀 开始监控...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        time.sleep(1)
        continue

    h, w = frame.shape[:2]
    now = time.time()

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.3,
        verbose=False
    )

    if not results or results[0].boxes.id is None:
        continue

    boxes = results[0].boxes
    ids = boxes.id.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy()

    for tid, box, c in zip(ids, xyxy, cls):
        tid = int(tid)
        cls_name = model.names[int(c)]

        center = norm_center(box, w, h)

        # 初始化
        if tid not in track_states:
            track_states[tid] = {
                "class": cls_name,
                "first_time": now,
                "first_pos": center,
                "last_pos": center,
                "move_dist": 0.0,
                "last_alert": 0
            }
        else:
            d = distance(track_states[tid]["last_pos"], center)
            track_states[tid]["move_dist"] += d
            track_states[tid]["last_pos"] = center

        state = track_states[tid]
        duration = now - state["first_time"]

        # 告警判断
        if duration >= DURATION_THRESHOLD.get(cls_name, 9999):
            if state["move_dist"] <= MOVE_THRESHOLD:
                if now - state["last_alert"] >= ALERT_COOLDOWN:
                    state["last_alert"] = now

                    print(
                        f"🚨 告警触发 | ID={tid} | 类别={cls_name} | "
                        f"停留={int(duration)}s | 位移={state['move_dist']:.3f}"
                    )

                    # 可扩展：写数据库 / 发HTTP / 推MQ

        # 画框
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{cls_name}-{tid}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow("YOLO-World Monitor", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
