import cv2

# RTSP 地址
rtsp_url = "rtsp://admin:Aa147258@192.168.109.213"

# 打开视频流
cap = cv2.VideoCapture(rtsp_url)

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取RTSP流")
        break

    cv2.imshow("RTSP Stream", frame)

    # 按 q 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()