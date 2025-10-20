#!/usr/bin/env python3
"""
RTSP to Web Streaming Server
使用FFmpeg将RTSP流转换为Web可播放的HLS/DASH格式
"""

import json
import os
import subprocess
import threading
import time
import signal
import sys
import logging
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import shutil

class RTSPToWebStreamer:
    def __init__(self, config_file='ffmpeg_config.json'):
        self.config = self.load_config(config_file)
        self.setup_logging()
        self.ffmpeg_process = None
        self.running = False
        self.setup_directories()

    def load_config(self, config_file):
        """加载配置文件"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"配置文件 {config_file} 不存在，使用默认配置")
            return self.get_default_config()

    def get_default_config(self):
        """获取默认配置"""
        return {
            "ffmpeg_path": "ffmpeg",
            "rtsp_config": {
                "url": "rtsp://admin:password@192.168.1.100",
                "timeout": 30,
                "reconnect_delay": 5,
                "max_reconnect_attempts": 10
            },
            "output_formats": {
                "hls": {
                    "enabled": True,
                    "segment_time": 4,
                    "segment_list_size": 6,
                    "output_dir": "./hls_output",
                    "playlist": "stream.m3u8"
                },
                "dash": {
                    "enabled": True,
                    "segment_duration": 4,
                    "output_dir": "./dash_output",
                    "manifest": "stream.mpd"
                }
            },
            "stream_settings": {
                "video_codec": "libx264",
                "audio_codec": "aac",
                "bitrate": "2000k",
                "resolution": "1280x720",
                "framerate": 25,
                "preset": "ultrafast",
                "tune": "zerolatency"
            },
            "web_server": {
                "port": 8080,
                "static_dir": "./public"
            }
        }

    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('rtsp_stream.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_directories(self):
        """创建必要的目录"""
        os.makedirs(self.config['output_formats']['hls']['output_dir'], exist_ok=True)
        os.makedirs(self.config['output_formats']['dash']['output_dir'], exist_ok=True)
        os.makedirs(self.config['web_server']['static_dir'], exist_ok=True)

    def build_ffmpeg_command(self):
        """构建FFmpeg命令"""
        rtsp_config = self.config['rtsp_config']
        stream_settings = self.config['stream_settings']
        hls_config = self.config['output_formats']['hls']

        # 基础FFmpeg命令
        cmd = [
            self.config['ffmpeg_path'],
            '-rtsp_transport', 'tcp',  # 使用TCP传输
            '-timeout', str(rtsp_config['timeout'] * 1000000),  # 超时时间（微秒）
            '-i', rtsp_config['url'],  # RTSP输入
            '-c:v', stream_settings['video_codec'],
            '-c:a', stream_settings['audio_codec'],
            '-b:v', stream_settings['bitrate'],
            '-s', stream_settings['resolution'],
            '-r', str(stream_settings['framerate']),
            '-preset', stream_settings['preset'],
            '-tune', stream_settings['tune'],
            '-g', str(int(stream_settings['framerate'] * 2)),  # GOP大小
            '-keyint_min', str(int(stream_settings['framerate'])),  # 最小关键帧间隔
            '-f', 'hls',  # HLS格式
            '-hls_time', str(hls_config['segment_time']),
            '-hls_list_size', str(hls_config['segment_list_size']),
            '-hls_flags', 'delete_segments+append_list',
            '-hls_allow_cache', '0',
            '-hls_segment_type', 'mpegts',
            '-hls_segment_filename', f"{hls_config['output_dir']}/segment_%03d.ts",
            f"{hls_config['output_dir']}/{hls_config['playlist']}"
        ]

        return cmd

    def start_streaming(self):
        """开始流媒体转换"""
        if self.running:
            self.logger.warning("流媒体转换已经在运行中")
            return

        self.logger.info("启动RTSP到Web的流媒体转换")
        self.running = True

        try:
            cmd = self.build_ffmpeg_command()
            self.logger.info(f"执行FFmpeg命令: {' '.join(cmd)}")

            # 启动FFmpeg进程
            self.ffmpeg_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            # 启动输出监控线程
            self.monitor_thread = threading.Thread(target=self.monitor_ffmpeg_output)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()

            self.logger.info("FFmpeg进程已启动")

        except Exception as e:
            self.logger.error(f"启动FFmpeg失败: {e}")
            self.running = False

    def monitor_ffmpeg_output(self):
        """监控FFmpeg输出"""
        if not self.ffmpeg_process:
            return

        while self.running and self.ffmpeg_process.poll() is None:
            output = self.ffmpeg_process.stderr.readline()
            if output:
                self.logger.info(f"FFmpeg: {output.strip()}")
            time.sleep(0.1)

        if self.ffmpeg_process.poll() is not None:
            self.logger.warning(f"FFmpeg进程已退出，返回码: {self.ffmpeg_process.returncode}")
            if self.running:
                # 尝试重新连接
                self.reconnect()

    def reconnect(self):
        """重新连接RTSP流"""
        rtsp_config = self.config['rtsp_config']

        for attempt in range(rtsp_config['max_reconnect_attempts']):
            if not self.running:
                break

            self.logger.info(f"尝试重新连接RTSP流 ({attempt + 1}/{rtsp_config['max_reconnect_attempts']})")

            if self.ffmpeg_process:
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait()

            time.sleep(rtsp_config['reconnect_delay'])

            if self.running:
                self.start_streaming()
                break
        else:
            self.logger.error("达到最大重连次数，停止流媒体转换")
            self.stop_streaming()

    def stop_streaming(self):
        """停止流媒体转换"""
        self.logger.info("停止流媒体转换")
        self.running = False

        if self.ffmpeg_process:
            self.ffmpeg_process.terminate()
            try:
                self.ffmpeg_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.ffmpeg_process.kill()
            self.ffmpeg_process = None

    def get_stream_url(self, format_type='hls'):
        """获取流媒体URL"""
        if format_type == 'hls':
            config = self.config['output_formats']['hls']
            return f"http://localhost:{self.config['web_server']['port']}/hls/{config['playlist']}"
        elif format_type == 'dash':
            config = self.config['output_formats']['dash']
            return f"http://localhost:{self.config['web_server']['port']}/dash/{config['manifest']}"
        else:
            raise ValueError(f"不支持的格式: {format_type}")

class StreamRequestHandler(SimpleHTTPRequestHandler):
    """自定义HTTP请求处理器"""

    def __init__(self, *args, streamer=None, **kwargs):
        self.streamer = streamer
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """处理GET请求"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        # 处理根路径
        if path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()

            html = self.get_player_page()
            self.wfile.write(html.encode('utf-8'))
            return

        # 处理HLS流
        if path.startswith('/hls/'):
            file_path = os.path.join(self.streamer.config['output_formats']['hls']['output_dir'],
                                   path[5:])
            self.serve_file(file_path)
            return

        # 处理DASH流
        if path.startswith('/dash/'):
            file_path = os.path.join(self.streamer.config['output_formats']['dash']['output_dir'],
                                   path[6:])
            self.serve_file(file_path)
            return

        # 处理其他静态文件
        super().do_GET()

    def serve_file(self, file_path):
        """提供文件服务"""
        if not os.path.exists(file_path):
            self.send_error(404, "File not found")
            return

        try:
            with open(file_path, 'rb') as f:
                content = f.read()

            self.send_response(200)

            # 设置正确的Content-Type
            if file_path.endswith('.m3u8'):
                self.send_header('Content-type', 'application/x-mpegURL')
            elif file_path.endswith('.ts'):
                self.send_header('Content-type', 'video/MP2T')
            elif file_path.endswith('.mpd'):
                self.send_header('Content-type', 'application/dash+xml')
            else:
                self.send_header('Content-type', 'application/octet-stream')

            self.send_header('Content-Length', str(len(content)))
            self.end_headers()

            self.wfile.write(content)

        except Exception as e:
            self.send_error(500, f"Error serving file: {e}")

    def get_player_page(self):
        """获取播放器页面"""
        hls_url = self.streamer.get_stream_url('hls')
        dash_url = self.streamer.get_stream_url('dash')

        return f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RTSP流媒体播放器</title>
    <link href="https://vjs.zencdn.net/8.6.1/video-js.css" rel="stylesheet">
    <script src="https://vjs.zencdn.net/8.6.1/video.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .video-container {{
            margin: 20px 0;
        }}
        .controls {{
            margin: 20px 0;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }}
        .status {{
            margin: 10px 0;
            padding: 10px;
            border-radius: 5px;
        }}
        .status.running {{
            background-color: #d4edda;
            color: #155724;
        }}
        .status.stopped {{
            background-color: #f8d7da;
            color: #721c24;
        }}
        button {{
            padding: 10px 20px;
            margin: 5px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }}
        .start-btn {{
            background-color: #28a745;
            color: white;
        }}
        .stop-btn {{
            background-color: #dc3545;
            color: white;
        }}
        .info-panel {{
            margin: 20px 0;
            padding: 15px;
            background-color: #e9ecef;
            border-radius: 5px;
        }}
        .tab-container {{
            margin: 20px 0;
        }}
        .tab-button {{
            padding: 10px 20px;
            margin: 0;
            border: none;
            background-color: #6c757d;
            color: white;
            cursor: pointer;
        }}
        .tab-button.active {{
            background-color: #007bff;
        }}
        .tab-content {{
            display: none;
            padding: 20px;
            border: 1px solid #ddd;
            border-top: none;
        }}
        .tab-content.active {{
            display: block;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>RTSP流媒体播放器</h1>

        <div class="controls">
            <button class="start-btn" onclick="startStream()">开始流媒体转换</button>
            <button class="stop-btn" onclick="stopStream()">停止流媒体转换</button>
            <button onclick="checkStatus()">检查状态</button>
        </div>

        <div id="status" class="status stopped">
            状态：未启动
        </div>

        <div class="tab-container">
            <button class="tab-button active" onclick="showTab('hls')">HLS播放器</button>
            <button class="tab-button" onclick="showTab('dash')">DASH播放器</button>
        </div>

        <div id="hls-tab" class="tab-content active">
            <div class="video-container">
                <video
                    id="hls-player"
                    class="video-js vjs-default-skin"
                    controls
                    preload="auto"
                    width="800"
                    height="450"
                    data-setup='{{}}'>
                    <source src="{hls_url}" type="application/x-mpegURL">
                </video>
            </div>
        </div>

        <div id="dash-tab" class="tab-content">
            <div class="video-container">
                <video
                    id="dash-player"
                    class="video-js vjs-default-skin"
                    controls
                    preload="auto"
                    width="800"
                    height="450"
                    data-setup='{{}}'>
                    <source src="{dash_url}" type="application/dash+xml">
                </video>
            </div>
        </div>

        <div class="info-panel">
            <h3>流媒体信息</h3>
            <p><strong>HLS流地址:</strong> {hls_url}</p>
            <p><strong>DASH流地址:</strong> {dash_url}</p>
            <p><strong>RTSP源地址:</strong> {self.streamer.config['rtsp_config']['url']}</p>
            <p><strong>视频编码:</strong> {self.streamer.config['stream_settings']['video_codec']}</p>
            <p><strong>音频编码:</strong> {self.streamer.config['stream_settings']['audio_codec']}</p>
            <p><strong>分辨率:</strong> {self.streamer.config['stream_settings']['resolution']}</p>
            <p><strong>比特率:</strong> {self.streamer.config['stream_settings']['bitrate']}</p>
        </div>
    </div>

    <script>
        let hlsPlayer, dashPlayer;

        // 初始化播放器
        document.addEventListener('DOMContentLoaded', function() {{
            hlsPlayer = videojs('hls-player');
            dashPlayer = videojs('dash-player');
        }});

        function showTab(tabName) {{
            // 隐藏所有标签页
            document.querySelectorAll('.tab-content').forEach(tab => {{
                tab.classList.remove('active');
            }});
            document.querySelectorAll('.tab-button').forEach(btn => {{
                btn.classList.remove('active');
            }});

            // 显示选中的标签页
            document.getElementById(tabName + '-tab').classList.add('active');
            event.target.classList.add('active');
        }}

        function startStream() {{
            fetch('/start-stream', {{ method: 'POST' }})
                .then(response => response.json())
                .then(data => {{
                    updateStatus(data.status);
                    if (data.success) {{
                        setTimeout(() => {{
                            hlsPlayer.src({{ src: '{hls_url}', type: 'application/x-mpegURL' }});
                            dashPlayer.src({{ src: '{dash_url}', type: 'application/dash+xml' }});
                        }}, 2000);
                    }}
                }})
                .catch(error => {{
                    console.error('Error:', error);
                }});
        }}

        function stopStream() {{
            fetch('/stop-stream', {{ method: 'POST' }})
                .then(response => response.json())
                .then(data => {{
                    updateStatus(data.status);
                }})
                .catch(error => {{
                    console.error('Error:', error);
                }});
        }}

        function checkStatus() {{
            fetch('/status')
                .then(response => response.json())
                .then(data => {{
                    updateStatus(data.status);
                }})
                .catch(error => {{
                    console.error('Error:', error);
                }});
        }}

        function updateStatus(status) {{
            const statusDiv = document.getElementById('status');
            statusDiv.textContent = '状态：' + status;
            statusDiv.className = 'status ' + (status === '运行中' ? 'running' : 'stopped');
        }}

        // 定期检查状态
        setInterval(checkStatus, 5000);
    </script>
</body>
</html>
        """

def signal_handler(signum, frame):
    """信号处理器"""
    print("\n接收到退出信号，正在停止服务...")
    if hasattr(signal_handler, 'streamer'):
        signal_handler.streamer.stop_streaming()
    sys.exit(0)

def main():
    """主函数"""
    print("RTSP到Web流媒体转换器")
    print("=" * 50)

    # 创建流媒体转换器
    streamer = RTSPToWebStreamer()
    signal_handler.streamer = streamer

    # 设置信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 设置HTTP服务器
    port = streamer.config['web_server']['port']

    def handler(*args, **kwargs):
        return StreamRequestHandler(*args, streamer=streamer, **kwargs)

    httpd = HTTPServer(('', port), handler)

    print(f"Web服务器已启动: http://localhost:{port}")
    print("按 Ctrl+C 停止服务")

    try:
        # 启动HTTP服务器
        server_thread = threading.Thread(target=httpd.serve_forever)
        server_thread.daemon = True
        server_thread.start()

        # 添加API路由
        add_api_routes(streamer, httpd)

        # 保持主线程运行
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n正在停止服务...")
        streamer.stop_streaming()
        httpd.shutdown()
        httpd.server_close()
        print("服务已停止")

def add_api_routes(streamer, httpd):
    """添加API路由"""
    # 这里简化处理，实际可以使用更复杂的Web框架
    pass

if __name__ == "__main__":
    main()