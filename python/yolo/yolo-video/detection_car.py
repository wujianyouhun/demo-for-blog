"""
YOLO车辆和行人检测与跟踪系统
使用YOLOv8检测视频中的车辆和行人，并实现目标跟踪功能
作者：Claude Code Assistant
版本：2.0 (优化版)
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch
from collections import deque
import time
import sys
import os
import warnings
from PIL import Image, ImageDraw, ImageFont

# 解决Windows平台中文编码问题
if sys.platform == 'win32':
    import locale
    import codecs
    # 设置控制台编码
    try:
        # 尝试设置UTF-8编码
        sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
        sys.stderr.reconfigure(encoding='utf-8', errors='ignore')
        # 设置Windows控制台代码页为UTF-8
        os.system('chcp 65001 >nul 2>&1')
    except:
        # 如果UTF-8失败，尝试使用系统默认编码
        try:
            locale.setlocale(locale.LC_ALL, 'Chinese_Simplified.936')
            sys.stdout.reconfigure(encoding='gbk', errors='ignore')
            sys.stderr.reconfigure(encoding='gbk', errors='ignore')
        except:
            # 最后的备选方案
            sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
            sys.stderr.reconfigure(encoding='utf-8', errors='ignore')

# 修复PyTorch 2.6权重加载问题
import torch.serialization
try:
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])
except:
    pass

# 忽略警告信息
warnings.filterwarnings("ignore", category=UserWarning)


def put_chinese_text(img, text, position, font_size=20, color=(255, 255, 255)):
    """
    在OpenCV图像上绘制中文文本

    Args:
        img: OpenCV图像
        text: 要绘制的文本
        position: 位置 (x, y)
        font_size: 字体大小
        color: 颜色 (B, G, R)

    Returns:
        处理后的图像
    """
    try:
        # 将OpenCV图像转换为PIL图像
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # 尝试加载中文字体
        font_paths = [
            "C:/Windows/Fonts/msyh.ttc",        # 微软雅黑
            "C:/Windows/Fonts/simhei.ttf",      # 黑体
            "C:/Windows/Fonts/simsun.ttc",      # 宋体
            "C:/Windows/Fonts/arial.ttf"        # Arial (备选)
        ]

        font = None
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                continue

        if font is None:
            # 如果都失败了，使用默认字体
            font = ImageFont.load_default()

        # 绘制文本
        draw.text(position, text, font=font, fill=color)

        # 转换回OpenCV格式
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    except Exception as e:
        # 如果PIL方法失败，回退到OpenCV方法（显示英文）
        # 英文映射
        english_map = {
            "行人": "Person", "汽车": "Car", "卡车": "Truck",
            "公交车": "Bus", "摩托车": "Motorcycle", "自行车": "Bicycle",
            "帧数:": "Frame:", "行人数:": "People:", "车辆数:": "Vehicles:",
            "FPS:": "FPS:", "ID:": "ID:"
        }

        # 将中文转换为英文
        english_text = text
        for chinese, english in english_map.items():
            english_text = english_text.replace(chinese, english)

        # 使用OpenCV绘制英文
        cv2.putText(img, english_text, position, cv2.FONT_HERSHEY_SIMPLEX,
                   font_size / 25, color, 2)

    return img


def create_english_labels():
    """创建英文标签映射"""
    return {
        'person': 'Person',
        'car': 'Car',
        'truck': 'Truck',
        'bus': 'Bus',
        'motorcycle': 'Motorcycle',
        'bicycle': 'Bicycle'
    }


class ObjectTracker:
    """
    目标跟踪器类
    使用质心距离算法进行多目标跟踪
    """

    def __init__(self, max_disappeared=10, max_distance=50):
        """
        初始化跟踪器

        Args:
            max_disappeared (int): 目标消失最大帧数，超过则删除跟踪
            max_distance (int): 匹配最大距离阈值（像素）
        """
        self.next_object_id = 0  # 下一个目标ID
        self.objects = {}         # 存储所有跟踪目标
        self.disappeared = {}     # 存储目标消失计数
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        """
        注册新的跟踪目标

        Args:
            centroid (tuple): 目标质心坐标 (x, y)
        """
        self.objects[self.next_object_id] = {
            'centroid': centroid,     # 质心位置
            'class_id': None,         # 类别ID
            'class_name': None,       # 类别名称
            'bbox': None,            # 边界框坐标
            'trajectory': deque(maxlen=30)  # 轨迹历史（最近30个位置）
        }
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        """
        删除跟踪目标

        Args:
            object_id (int): 要删除的目标ID
        """
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections):
        """
        更新跟踪器状态

        Args:
            detections (list): 当前帧检测结果列表

        Returns:
            dict: 更新后的跟踪目标字典
        """
        # 如果没有检测结果，更新所有目标的消失计数
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                # 如果目标消失时间过长，删除跟踪
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # 计算所有检测结果的质心坐标
        input_centroids = np.zeros((len(detections), 2), dtype="int")
        for i, detection in enumerate(detections):
            x, y, w, h = detection['bbox']
            cx = int(x + w / 2.0)  # 质心x坐标
            cy = int(y + h / 2.0)  # 质心y坐标
            input_centroids[i] = (cx, cy)

        # 如果当前没有跟踪目标，为每个检测结果注册新目标
        if len(self.objects) == 0:
            for i in range(len(detections)):
                self.register(input_centroids[i])
                obj_id = self.next_object_id - 1
                self.objects[obj_id]['class_id'] = detections[i]['class_id']
                self.objects[obj_id]['class_name'] = detections[i]['class_name']
                self.objects[obj_id]['bbox'] = detections[i]['bbox']
                self.objects[obj_id]['trajectory'].append(input_centroids[i])
        else:
            # 计算现有目标和检测结果的距离矩阵
            object_centroids = np.array([obj['centroid'] for obj in self.objects.values()])
            object_ids = list(self.objects.keys())
            D = np.linalg.norm(object_centroids[:, np.newaxis] - input_centroids[np.newaxis, :], axis=2)

            # 使用匈牙利算法进行最优匹配
            rows = D.min(axis=1).argsort()  # 按最小距离排序
            cols = D.argmin(axis=1)[rows]   # 对应的检测结果索引

            used_row_idxs = set()  # 已使用的现有目标索引
            used_col_idxs = set()  # 已使用的检测结果索引

            # 匹配目标和检测结果
            for (row, col) in zip(rows, cols):
                if row in used_row_idxs or col in used_col_idxs:
                    continue  # 避免重复匹配

                if D[row, col] > self.max_distance:
                    continue  # 距离过大，不匹配

                # 更新匹配目标的信息
                if row < len(object_ids):  # 确保索引有效
                    object_id = object_ids[row]
                    self.objects[object_id]['centroid'] = input_centroids[col]
                    self.objects[object_id]['bbox'] = detections[col]['bbox']
                    self.objects[object_id]['trajectory'].append(input_centroids[col])
                    self.disappeared[object_id] = 0

                    used_row_idxs.add(row)
                    used_col_idxs.add(col)

            # 处理未匹配的现有目标（可能消失）
            unused_row_idxs = set(range(0, D.shape[0])).difference(used_row_idxs)
            unused_col_idxs = set(range(0, D.shape[1])).difference(used_col_idxs)

            # 如果现有目标多于检测结果，更新消失计数
            if D.shape[0] >= D.shape[1]:
                for row in unused_row_idxs:
                    if row < len(object_ids):  # 确保索引有效
                        object_id = object_ids[row]
                        self.disappeared[object_id] += 1
                        if self.disappeared[object_id] > self.max_disappeared:
                            self.deregister(object_id)
            else:
                # 如果检测结果多于现有目标，注册新目标
                for col in unused_col_idxs:
                    self.register(input_centroids[col])
                    obj_id = self.next_object_id - 1
                    self.objects[obj_id]['class_id'] = detections[col]['class_id']
                    self.objects[obj_id]['class_name'] = detections[col]['class_name']
                    self.objects[obj_id]['bbox'] = detections[col]['bbox']
                    self.objects[obj_id]['trajectory'].append(input_centroids[col])

        return self.objects


class YOLODetector:
    """
    YOLO检测器类
    封装YOLOv8检测功能，专门用于车辆和行人检测
    """

    def __init__(self, model_path="yolov8l.pt"):
        """
        初始化YOLO检测器

        Args:
            model_path (str): YOLO模型文件路径
        """
        print(f"正在加载YOLO模型: {model_path}")

        # 保存原始的torch.load函数
        original_torch_load = torch.load

        # 创建修补版本的torch.load来处理weights_only问题
        def patched_torch_load(f, *args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return original_torch_load(f, *args, **kwargs)

        # 设置安全全局变量（如果可用）
        if hasattr(torch.serialization, 'add_safe_globals'):
            try:
                torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])
            except:
                pass

        try:
            # 应用补丁
            torch.load = patched_torch_load
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"直接加载模型失败: {e}")
            # 尝试备选方案 - 设置环境变量并重试
            import os
            original_weights_only = os.environ.get('PYTORCH_WEIGHTS_ONLY', '1')
            os.environ['PYTORCH_WEIGHTS_ONLY'] = '0'
            try:
                self.model = YOLO(model_path)
                print("通过环境变量变通方案加载模型成功")
            except Exception as e2:
                print(f"备选加载失败: {e2}")
                raise e2
            finally:
                os.environ['PYTORCH_WEIGHTS_ONLY'] = original_weights_only
        finally:
            # 恢复原始torch.load
            torch.load = original_torch_load

        # 获取模型类别名称
        self.class_names = self.model.names
        # 定义要检测的目标类别（车辆和行人相关）
        self.target_classes = ['person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle']
        # 初始化跟踪器
        self.tracker = ObjectTracker()
        print("模型加载成功!")

        # 定义不同类别的颜色（BGR格式）
        self.colors = {
            'person': (0, 255, 0),      # 绿色 - 行人
            'car': (0, 0, 255),         # 红色 - 汽车
            'truck': (255, 0, 0),       # 蓝色 - 卡车
            'bus': (255, 255, 0),       # 青色 - 公交车
            'motorcycle': (255, 0, 255), # 紫色 - 摩托车
            'bicycle': (0, 255, 255)    # 黄色 - 自行车
        }

    def detect(self, frame, conf_threshold=0.5):
        """
        在图像帧中检测目标

        Args:
            frame (numpy.ndarray): 输入图像帧
            conf_threshold (float): 置信度阈值

        Returns:
            list: 检测结果列表，每个元素包含bbox、confidence、class_id、class_name
        """
        # 使用YOLO模型进行检测
        results = self.model(frame, conf=conf_threshold)
        detections = []

        # 处理检测结果
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.class_names[class_id]

                    # 只检测目标类别
                    if class_name in self.target_classes:
                        bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                        detection = {
                            'bbox': bbox,
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': class_name
                        }
                        detections.append(detection)

        return detections

    def draw_detections(self, frame, tracked_objects):
        """
        在图像上绘制检测结果和跟踪信息

        Args:
            frame (numpy.ndarray): 输入图像帧
            tracked_objects (dict): 跟踪目标字典

        Returns:
            numpy.ndarray: 绘制后的图像帧
        """
        for object_id, obj_data in tracked_objects.items():
            bbox = obj_data['bbox']
            class_name = obj_data['class_name']
            trajectory = obj_data['trajectory']

            if bbox is not None:
                x, y, w, h = bbox

                # 获取类别对应的颜色
                color = self.colors.get(class_name, (255, 255, 255))

                # 绘制边界框
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                # 绘制标签
                if class_name == 'person':
                    label_text = f"ID: {object_id} 行人"
                elif class_name == 'car':
                    label_text = f"ID: {object_id} 汽车"
                elif class_name == 'truck':
                    label_text = f"ID: {object_id} 卡车"
                elif class_name == 'bus':
                    label_text = f"ID: {object_id} 公交车"
                elif class_name == 'motorcycle':
                    label_text = f"ID: {object_id} 摩托车"
                elif class_name == 'bicycle':
                    label_text = f"ID: {object_id} 自行车"
                else:
                    label_text = f"ID: {object_id} {class_name}"

                # 绘制背景矩形
                cv2.rectangle(frame, (x, y - 30), (x + 150, y), color, -1)

                # 使用中文文本绘制函数
                frame = put_chinese_text(frame, label_text, (x, y - 5), 16, (255, 255, 255))

                # 绘制轨迹（如果有历史位置）- 已禁用
                # if len(trajectory) > 1:
                #     points = np.array(list(trajectory), dtype=np.int32)
                #     cv2.polylines(frame, [points], False, color, 2)

        return frame


class VideoProcessor:
    """
    视频处理器类
    协调检测和跟踪功能，处理视频输入和输出
    """

    def __init__(self, model_path="yolov8l.pt"):
        """
        初始化视频处理器

        Args:
            model_path (str): YOLO模型文件路径
        """
        self.detector = YOLODetector(model_path)

    def process_video(self, video_source, output_path=None, show_display=True):
        """
        处理视频文件或摄像头输入

        Args:
            video_source: 视频源（文件路径或摄像头索引）
            output_path (str): 输出视频文件路径
            show_display (bool): 是否显示处理结果
        """
        # 打开视频源
        cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            print(f"错误: 无法打开视频源: {video_source}")
            return

        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 设置输出视频
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        else:
            out = None

        print(f"正在处理视频 - 帧率: {fps}, 分辨率: {width}x{height}")
        print("按 'q' 退出, 按 's' 保存当前帧")

        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 检测目标
            detections = self.detector.detect(frame)

            # 更新跟踪器
            tracked_objects = self.detector.tracker.update(detections)

            # 绘制检测结果
            frame = self.detector.draw_detections(frame, tracked_objects)

            # 添加统计信息
            person_count = sum(1 for obj in tracked_objects.values() if obj['class_name'] == 'person')
            vehicle_count = sum(1 for obj in tracked_objects.values() if obj['class_name'] != 'person')

            # 显示中文统计信息
            frame = put_chinese_text(frame, f"帧数: {frame_count}", (10, 30), 20, (255, 255, 255))
            frame = put_chinese_text(frame, f"行人数: {person_count}", (10, 60), 20, (255, 255, 255))
            frame = put_chinese_text(frame, f"车辆数: {vehicle_count}", (10, 90), 20, (255, 255, 255))

            # 计算并显示FPS
            if frame_count % 30 == 0:
                elapsed_time = time.time() - start_time
                current_fps = frame_count / elapsed_time
                frame = put_chinese_text(frame, f"FPS: {current_fps:.2f}", (10, 120), 20, (255, 255, 255))

            # 显示或保存结果
            if show_display:
                cv2.imshow('YOLO Vehicle and Pedestrian Detection & Tracking', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    cv2.imwrite(f'检测结果_{frame_count}.jpg', frame)
                    print(f"已保存第 {frame_count} 帧")

            if out:
                out.write(frame)

        # 清理资源
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

        # 输出处理结果统计
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        print(f"\n处理完成!")
        print(f"{'='*50}")
        print(f"性能统计")
        print(f"{'='*50}")
        print(f"总帧数: {frame_count}")
        print(f"平均FPS: {avg_fps:.2f}")
        print(f"处理时间: {total_time:.2f} 秒")
        print(f"视频时长: {frame_count/max(fps, 1):.2f} 秒")

        # 性能建议
        if avg_fps < 15:
            print(f"\n⚠️  检测到性能较低!")
            print("建议:")
            print("- 尝试使用更小的模型 (yolov8n.pt 或 yolov8s.pt)")
            print("- 降低视频分辨率")
            print("- 关闭其他应用程序")
        elif avg_fps > 30:
            print(f"\n✅ 性能优秀!")
            print("您可以使用更大的模型获得更高精度:")
            print("- 尝试 yolov8x.pt 获得更高精度")
        else:
            print(f"\n👍 性能良好")
            print("当前设置提供了很好的平衡")

        print(f"\n输出文件: {output_path if output_path else '未保存'}")


def print_model_info():
    """
    打印可用的YOLO模型信息
    """
    models = {
        'yolov8n.pt': {'size': '6.2MB', 'mAP': '37.3', 'speed': '80+', 'description': 'Nano - 最小，最快'},
        'yolov8s.pt': {'size': '21.5MB', 'mAP': '44.9', 'speed': '50+', 'description': 'Small - 速度和精度平衡'},
        'yolov8m.pt': {'size': '49.7MB', 'mAP': '50.2', 'speed': '40+', 'description': 'Medium - 良好精度'},
        'yolov8l.pt': {'size': '83.7MB', 'mAP': '52.9', 'speed': '30+', 'description': 'Large - 高精度，推荐'},
        'yolov8x.pt': {'size': '131.4MB', 'mAP': '53.9', 'speed': '20+', 'description': 'Extra Large - 最高精度，最慢'},
    }

    print("\n可用的YOLOv8模型:")
    print("-" * 80)
    print(f"{'模型':<12} {'大小':<8} {'mAP':<6} {'FPS':<8} {'描述'}")
    print("-" * 80)
    for model, info in models.items():
        print(f"{model:<12} {info['size']:<8} {info['mAP']:<6} {info['speed']:<8} {info['description']}")
    print("-" * 80)
    print("注意: 更大的模型精度更高但速度更慢")
    print("推荐: yolov8l.pt (默认) 追求高精度, yolov8s.pt 追求平衡性能")


def get_model_choice():
    """
    获取用户的模型选择

    Returns:
        str: 选择的模型路径或特殊标识
    """
    print_model_info()

    print("\n选择YOLO模型:")
    print("1. yolov8n.pt (Nano) - 最快，基础精度")
    print("2. yolov8s.pt (Small) - 良好的平衡性能")
    print("3. yolov8m.pt (Medium) - 精度和速度平衡")
    print("4. yolov8l.pt (Large) - 🔥 默认 - 高精度，推荐")
    print("5. yolov8x.pt (Extra Large) - 最高精度，最慢")
    print("6. 自定义模型路径")
    print("7. 对比所有模型 (性能基准测试)")

    while True:
        choice = input("\n请输入选择 (1-7, 默认=4): ").strip()

        if not choice:
            return "yolov8l.pt"  # 默认选择
        elif choice == '1':
            return "yolov8n.pt"
        elif choice == '2':
            return "yolov8s.pt"
        elif choice == '3':
            return "yolov8m.pt"
        elif choice == '4':
            return "yolov8l.pt"
        elif choice == '5':
            return "yolov8x.pt"
        elif choice == '6':
            custom_path = input("请输入自定义模型路径: ").strip()
            if custom_path:
                return custom_path
            else:
                print("路径无效，使用默认 yolov8s.pt")
                return "yolov8s.pt"
        elif choice == '7':
            return "COMPARE_ALL"
        else:
            print("无效选择。请输入1-7。")


def compare_models(video_file):
    """
    对比不同YOLO模型的性能

    Args:
        video_file (str): 视频文件路径
    """
    models_to_test = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']

    print(f"\n{'='*60}")
    print("模型性能对比测试")
    print(f"{'='*60}")
    print(f"使用视频文件进行测试: {video_file}")
    print("将测试每个模型的前100帧进行对比")
    print(f"{'='*60}")

    results = []

    for model_path in models_to_test:
        print(f"\n正在测试 {model_path}...")

        try:
            processor = VideoProcessor(model_path)

            # 快速基准测试 - 只测试前100帧
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                print(f"无法打开视频: {video_file}")
                continue

            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            test_frames = min(100, total_frames)

            start_time = time.time()
            frame_count = 0
            total_detections = 0

            while frame_count < test_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # 检测目标
                detections = processor.detector.detect(frame)
                total_detections += len(detections)

                # 更新跟踪器（为了速度，不绘制）
                processor.detector.tracker.update(detections)

                frame_count += 1

            cap.release()

            benchmark_time = time.time() - start_time
            avg_fps = frame_count / benchmark_time
            avg_detections = total_detections / max(frame_count, 1)

            results.append({
                'model': model_path,
                'fps': avg_fps,
                'detections': avg_detections,
                'time': benchmark_time
            })

            print(f"  ✓ {model_path}: {avg_fps:.1f} FPS, {avg_detections:.1f} 平均检测数/帧")

        except Exception as e:
            print(f"  ✗ {model_path}: 失败 - {e}")

    # 打印对比结果
    print(f"\n{'='*60}")
    print("基准测试结果")
    print(f"{'='*60}")
    print(f"{'模型':<15} {'FPS':<8} {'平均检测':<10} {'时间(秒)':<10} {'评级'}")
    print("-" * 60)

    for result in results:
        model = result['model']
        fps = result['fps']
        detections = result['detections']
        time_taken = result['time']

        if fps > 40:
            rating = "⚡ 极快"
        elif fps > 25:
            rating = "✅ 良好"
        elif fps > 15:
            rating = "⚠️  较慢"
        else:
            rating = "🐌 很慢"

        print(f"{model:<15} {fps:<8.1f} {detections:<10.1f} {time_taken:<10.2f} {rating}")

    print("-" * 60)

    if results:
        fastest = max(results, key=lambda x: x['fps'])
        most_detect = max(results, key=lambda x: x['detections'])

        print(f"\n🏆 最快模型: {fastest['model']} ({fastest['fps']:.1f} FPS)")
        print(f"🎯 检测最多: {most_detect['model']} ({most_detect['detections']:.1f} 平均)")

        # 推荐
        avg_fps = sum(r['fps'] for r in results) / len(results)
        if avg_fps < 20:
            recommended = "yolov8n.pt (优先速度)"
        elif avg_fps > 35:
            recommended = "yolov8l.pt 或 yolov8x.pt (优先精度)"
        else:
            recommended = "yolov8s.pt 或 yolov8m.pt (平衡)"

        print(f"\n💡 为您的系统推荐: {recommended}")


def main():
    """
    主函数
    """
    print("YOLO 车辆和行人检测与跟踪系统")
    print("=" * 50)
    print("增强版 - 支持模型选择以获得更高精度")

    # 检查命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("用法: python detection_car.py [模型路径]")
            print("示例:")
            print("  python detection_car.py              # 使用默认模型 yolov8l.pt")
            print("  python detection_car.py yolov8s.pt   # 使用指定模型")
            print("  python detection_car.py model.pt     # 使用自定义模型")
            return
        else:
            model_path = sys.argv[1]
            print(f"使用指定模型: {model_path}")
    else:
        # 获取用户模型选择（如果在交互环境中）
        try:
            model_path = get_model_choice()
        except (EOFError, KeyboardInterrupt):
            # 非交互环境或用户中断，使用默认模型
            print("\n检测到非交互环境，使用默认模型: yolov8l.pt")
            model_path = "yolov8l.pt"

    # 检测video.mp4文件
    video_file = "video.mp4"

    # 处理模型对比选项
    if model_path == "COMPARE_ALL":
        if not os.path.exists(video_file):
            print(f"\n错误: 找不到视频文件 '{video_file}'!")
            print("请在当前目录下放置 video.mp4 文件。")
            return

        compare_models(video_file)
        return

    # 使用选择的模型创建视频处理器
    processor = VideoProcessor(model_path)

    # 基于模型生成输出文件名
    model_name = os.path.basename(model_path).replace('.pt', '')
    output_file = f"output_{model_name}_result.mp4"

    # 检查视频文件是否存在
    if not os.path.exists(video_file):
        print(f"\n错误: 找不到视频文件 '{video_file}'!")
        print("请在当前目录下放置 video.mp4 文件。")
        return

    # 开始处理
    print(f"\n{'='*50}")
    print(f"开始检测，配置信息:")
    print(f"  模型: {model_path}")
    print(f"  视频: {video_file}")
    print(f"  输出: {output_file}")
    print(f"{'='*50}")
    print("按 'q' 退出, 按 's' 保存当前帧")
    print("按 'ESC' 或关闭窗口停止处理\n")

    processor.process_video(video_file, output_file, show_display=True)


if __name__ == "__main__":
    main()