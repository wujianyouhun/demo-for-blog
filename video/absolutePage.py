import cv2
import numpy as np
import os
from datetime import datetime

# RTSP 地址
rtsp_url = "rtsp://admin:Aa147258@192.168.109.213"

# 输出目录
output_dir = "./frames"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("无法连接 RTSP 流！请检查地址或网络。")
    exit()

prev_frame = None
frame_count = 0
save_count = 0
diff_threshold = 30  # 控制敏感度

while True:
    ret, frame = cap.read()
    if not ret:
        print("RTSP 流中断或结束。")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_frame is not None:
        diff = cv2.absdiff(gray, prev_frame)
        score = np.mean(diff)
        if score > diff_threshold:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{output_dir}/keyframe_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            save_count += 1
            print(f"保存关键帧：{filename} (变化值={score:.2f})")

    prev_frame = gray
    frame_count += 1

cap.release()
print(f"共检测帧数: {frame_count}, 保存关键帧数: {save_count}")
