import cv2
from ultralytics import YOLO  # 仅导入YOLO，跟踪内置
import time
import numpy as np
import logging
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import threading
from queue import Queue

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('monitor.log')
    ]
)
logger = logging.getLogger(__name__)

# RTSP URL
RTSP_URL = "rtsp://admin:Aa147258@192.168.109.213"

# 保存目录
BASE_DIR = Path("detection_results")
BASE_DIR.mkdir(exist_ok=True)

IMAGES_DIR = BASE_DIR / "images"
EVENTS_DIR = BASE_DIR / "events"
DETECTIONS_DIR = BASE_DIR / "detections"

for directory in [IMAGES_DIR, EVENTS_DIR, DETECTIONS_DIR]:
    directory.mkdir(exist_ok=True)

# 加载YOLO模型（内置BOTSORT跟踪）
model = YOLO('yolov8m.pt')

# 检测参数
person_class = 0
vehicle_classes = [2, 3, 5, 7]
conf_threshold = 0.5
classes_to_detect = [person_class] + vehicle_classes

# 保存配置
SAVE_IMAGES = True
SAVE_EVENTS = True
EVENT_DURATION = 10  # 秒
PERSON_DEDUPE_FRAMES = 30  # 去重间隔

# 跟踪记录
track_history = {}
stats = defaultdict(int)
total_detections = 0
unique_tracks = set()

# 事件录制
event_writer = None
event_start_time = None
save_queue = Queue(maxsize=50)

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

def classify_object(cls_id):
    """对象分类"""
    if cls_id == person_class:
        return "person"
    elif cls_id in vehicle_classes:
        return "vehicle"
    return "other"

def analyze_behavior(bbox, frame_shape):
    """行为分析"""
    x1, y1, x2, y2 = bbox
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    h, w = frame_shape[:2]
    
    behavior = "center"
    if center_x < w * 0.3: behavior = "left"
    elif center_x > w * 0.7: behavior = "right"
    if center_y < h * 0.3: behavior = behavior + "_top"
    elif center_y > h * 0.7: behavior = behavior + "_bottom"
    
    return behavior

def should_save_person(track_id, current_frame):
    """人员去重"""
    if track_id not in track_history:
        return True
    last_saved = track_history[track_id].get("last_frame", 0)
    return current_frame - last_saved >= PERSON_DEDUPE_FRAMES

def console_output(detections, frame_id):
    """控制台输出"""
    global total_detections

    if not detections:
        return

    unique_ids = {det['track_id'] for det in detections}
    print("\n" + "="*80)
    print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 帧:{frame_id} | 唯一ID:{len(unique_ids)}")

    persons = [d for d in detections if d['type'] == 'person']
    vehicles = [d for d in detections if d['type'] == 'vehicle']

    print(f"👤 人员:{len(persons)} | 🚗 车辆:{len(vehicles)}")

    for det in detections:
        emoji = "👤" if det['type'] == 'person' else "🚗"
        print(f"  {emoji} ID:{det['track_id']} | {det['type']} | {det['behavior']} | {det['confidence']:.2f}")

    stats['person'] += len(persons)
    stats['vehicle'] += len(vehicles)
    total_detections += len(detections)
    print(f"📊 累计: 人={stats['person']}, 车={stats['vehicle']}, 总计={total_detections}")
    print("="*80)

def save_detection_image(frame, detections, frame_id):
    """保存去重图像"""
    # 只保存符合条件的人员
    persons_to_save = [
        det for det in detections 
        if det['type'] == 'person' and should_save_person(det['track_id'], frame_id)
    ]
    
    if not persons_to_save:
        return None
    
    # 绘制所有检测框
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        color = (0, 255, 0) if det['type'] == 'person' else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{det['track_id']} {det['confidence']:.1f}"
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # 图像信息
    timestamp = get_timestamp()
    filename = f"person_{frame_id}_{timestamp}.jpg"
    filepath = IMAGES_DIR / filename
    
    summary = f"帧:{frame_id} | 人员:{len(persons_to_save)}"
    cv2.putText(frame, summary, (10, frame.shape[0]-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imwrite(str(filepath), frame)
    logger.info(f"💾 保存图像: {filepath}")
    
    # 更新历史
    for person in persons_to_save:
        track_history[person['track_id']] = {
            "last_frame": frame_id,
            "save_count": track_history.get(person['track_id'], {}).get("save_count", 0) + 1
        }
    
    unique_tracks.update([p['track_id'] for p in persons_to_save])
    return str(filepath)

def save_detection_data(detections, image_path, frame_id):
    """保存JSON数据"""
    record = {
        "timestamp": datetime.now().isoformat(),
        "frame_id": frame_id,
        "image_path": image_path,
        "detections": detections,
        "statistics": dict(stats)
    }
    
    timestamp = get_timestamp()
    filename = f"detection_{frame_id}_{timestamp}.json"
    filepath = DETECTIONS_DIR / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False, indent=2, default=str)
    
    logger.info(f"📝 保存数据: {filepath}")
    return str(filepath)

def start_event_recording(frame):
    """事件视频"""
    global event_writer, event_start_time
    
    if event_writer:
        event_writer.release()
    
    timestamp = get_timestamp()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 25
    size = frame.shape[1], frame.shape[0]
    
    filename = f"event_{len(list(EVENTS_DIR.glob('*.mp4'))):04d}_{timestamp}.mp4"
    event_path = EVENTS_DIR / filename
    
    event_writer = cv2.VideoWriter(str(event_path), fourcc, fps, size)
    event_start_time = time.time()
    logger.warning(f"🚨 事件录制: {event_path}")
    return event_path

def process_frame(frame, model, frame_id):
    """核心检测+跟踪逻辑"""
    # 使用内置跟踪功能
    results = model.track(
        source=frame,
        persist=True,  # 保持Track ID
        conf=conf_threshold,
        classes=classes_to_detect,
        verbose=False,
        tracker="botsort.yaml",  # 内置BOTSORT
        project="temp",  # 临时项目，避免保存
        name="temp"      # 临时名称
    )
    
    detections = []
    result = results[0]  # 单帧结果
    
    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        track_ids = result.boxes.id
        
        # 处理跟踪ID
        if track_ids is not None:
            track_ids = track_ids.int().cpu().numpy()
        else:
            track_ids = np.arange(len(boxes))  # 备用ID
            
        for i in range(len(boxes)):
            box = boxes[i]
            conf = confidences[i]
            cls_id = class_ids[i]
            track_id = int(track_ids[i])
            
            x1, y1, x2, y2 = map(int, box)
            
            # 分类和行为
            obj_type = classify_object(int(cls_id))
            behavior = analyze_behavior([x1, y1, x2, y2], frame.shape)
            
            detection = {
                "track_id": track_id,
                "type": obj_type,
                "class_name": model.names[int(cls_id)],
                "confidence": float(conf),
                "bbox": [x1, y1, x2, y2],
                "behavior": behavior
            }
            detections.append(detection)
    
    return frame, detections

def save_worker():
    """异步保存"""
    from queue import Empty

    while True:
        try:
            task = save_queue.get(timeout=1)
            if task is None:
                break
            frame_id, frame, detections = task
            image_path = save_detection_image(frame.copy(), detections, frame_id)
            if image_path:
                save_detection_data(detections, image_path, frame_id)
            save_queue.task_done()
        except Empty:
            # 队列为空，继续等待
            continue
        except Exception as e:
            logger.error(f"保存错误: {e}", exc_info=True)

# 主程序初始化
cap = cv2.VideoCapture(RTSP_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    logger.error("无法打开RTSP流")
    exit(1)

# 启动保存线程
save_thread = threading.Thread(target=save_worker, daemon=True)
save_thread.start()

# 预热
logger.info("预热模型...")
for i in range(10):
    ret, frame = cap.read()
    if ret:
        frame, _ = process_frame(frame, model, i)

logger.info("🚀 开始监控...")
prev_time = time.time()
frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("RTSP重连...")
            cap.release()
            cap = cv2.VideoCapture(RTSP_URL)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue
        
        frame_id = frame_count
        annotated_frame, detections = process_frame(frame, model, frame_id)
        
        # 输出和保存
        if detections:
            console_output(detections, frame_id)
            if not save_queue.full():
                save_queue.put((frame_id, annotated_frame.copy(), detections))
        
        # 事件录制
        if SAVE_EVENTS and detections:
            if event_writer is None:
                start_event_recording(annotated_frame)
            event_writer.write(annotated_frame)
            if time.time() - event_start_time > EVENT_DURATION:
                event_writer.release()
                event_writer = None
        elif event_writer:
            event_writer.release()
            event_writer = None
        
        # FPS显示
        frame_count += 1
        if frame_count % 30 == 0:
            curr_time = time.time()
            fps = 30 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(annotated_frame, f'FPS:{fps:.1f}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('智能监控', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    logger.info("用户中断")

finally:
    save_queue.put(None)
    save_thread.join(timeout=5)
    if event_writer:
        event_writer.release()
    cap.release()
    cv2.destroyAllWindows()
    
    logger.info(f"📈 统计: 帧={frame_count}, 轨迹={len(unique_tracks)}, "
                f"图像={len(list(IMAGES_DIR.glob('*.jpg')))}")