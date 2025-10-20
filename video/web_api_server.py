#!/usr/bin/env python3
"""
Web API服务器 - 提供RESTful API和WebSocket服务
"""

import json
import os
import threading
import time
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any, Optional
import subprocess
import asyncio
import websockets
from stream_manager import StreamManager

class APIRequestHandler(BaseHTTPRequestHandler):
    """API请求处理器"""

    def __init__(self, *args, stream_manager=None, **kwargs):
        self.stream_manager = stream_manager
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """处理GET请求"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        query_params = parse_qs(parsed_path.query)

        # 路由
        if path == '/api/streams':
            self.handle_get_streams()
        elif path == '/api/stats':
            self.handle_get_stats()
        elif path.startswith('/api/streams/'):
            stream_id = path.split('/')[3]
            self.handle_get_stream(stream_id)
        elif path == '/':
            self.serve_index()
        elif path.startswith('/hls/') or path.startswith('/dash/'):
            self.serve_stream_file(path)
        else:
            self.send_error(404, 'Not Found')

    def do_POST(self):
        """处理POST请求"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        if path == '/api/streams':
            self.handle_create_stream()
        elif path.startswith('/api/streams/'):
            parts = path.split('/')
            if len(parts) >= 5 and parts[4] == 'start':
                self.handle_start_stream(parts[3])
            elif len(parts) >= 5 and parts[4] == 'stop':
                self.handle_stop_stream(parts[3])
            else:
                self.send_error(404, 'Not Found')
        else:
            self.send_error(404, 'Not Found')

    def do_DELETE(self):
        """处理DELETE请求"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        if path.startswith('/api/streams/'):
            stream_id = path.split('/')[3]
            self.handle_delete_stream(stream_id)
        else:
            self.send_error(404, 'Not Found')

    def handle_get_streams(self):
        """获取所有流列表"""
        try:
            streams = self.stream_manager.get_all_streams_status()
            self.send_json_response({
                'success': True,
                'streams': streams
            })
        except Exception as e:
            self.send_error_response(f'获取流列表失败: {e}')

    def handle_get_stream(self, stream_id: str):
        """获取单个流信息"""
        try:
            stream = self.stream_manager.get_stream_status(stream_id)
            self.send_json_response({
                'success': True,
                'stream': stream
            })
        except ValueError as e:
            self.send_error_response(str(e), 404)
        except Exception as e:
            self.send_error_response(f'获取流信息失败: {e}')

    def handle_get_stats(self):
        """获取系统统计信息"""
        try:
            stats = self.stream_manager.get_system_stats()
            self.send_json_response({
                'success': True,
                'stats': stats
            })
        except Exception as e:
            self.send_error_response(f'获取统计信息失败: {e}')

    def handle_create_stream(self):
        """创建新流"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))

            stream_id = data.get('stream_id')
            rtsp_url = data.get('rtsp_url')
            config = data.get('config', {})

            if not stream_id or not rtsp_url:
                self.send_error_response('缺少必要参数: stream_id, rtsp_url', 400)
                return

            self.stream_manager.add_stream(stream_id, rtsp_url, config)
            self.send_json_response({
                'success': True,
                'message': f'流 {stream_id} 创建成功'
            })

        except json.JSONDecodeError:
            self.send_error_response('无效的JSON数据', 400)
        except ValueError as e:
            self.send_error_response(str(e), 400)
        except Exception as e:
            self.send_error_response(f'创建流失败: {e}')

    def handle_start_stream(self, stream_id: str):
        """启动流"""
        try:
            self.stream_manager.start_stream(stream_id)
            self.send_json_response({
                'success': True,
                'message': f'流 {stream_id} 启动成功'
            })
        except ValueError as e:
            self.send_error_response(str(e), 404)
        except Exception as e:
            self.send_error_response(f'启动流失败: {e}')

    def handle_stop_stream(self, stream_id: str):
        """停止流"""
        try:
            self.stream_manager.stop_stream(stream_id)
            self.send_json_response({
                'success': True,
                'message': f'流 {stream_id} 停止成功'
            })
        except ValueError as e:
            self.send_error_response(str(e), 404)
        except Exception as e:
            self.send_error_response(f'停止流失败: {e}')

    def handle_delete_stream(self, stream_id: str):
        """删除流"""
        try:
            self.stream_manager.remove_stream(stream_id)
            self.send_json_response({
                'success': True,
                'message': f'流 {stream_id} 删除成功'
            })
        except ValueError as e:
            self.send_error_response(str(e), 404)
        except Exception as e:
            self.send_error_response(f'删除流失败: {e}')

    def serve_index(self):
        """提供主页"""
        try:
            index_path = os.path.join(os.path.dirname(__file__), 'public', 'index.html')
            with open(index_path, 'r', encoding='utf-8') as f:
                content = f.read()

            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(content.encode('utf-8'))

        except FileNotFoundError:
            self.send_error(404, 'Index page not found')
        except Exception as e:
            self.send_error_response(f'服务主页失败: {e}')

    def serve_stream_file(self, path: str):
        """提供流媒体文件服务"""
        try:
            # 解析路径
            if path.startswith('/hls/'):
                file_path = path[5:]  # 移除 '/hls/' 前缀
                base_dir = './hls_output'
            elif path.startswith('/dash/'):
                file_path = path[6:]  # 移除 '/dash/' 前缀
                base_dir = './dash_output'
            else:
                self.send_error(404, 'Not Found')
                return

            full_path = os.path.join(base_dir, file_path)

            if not os.path.exists(full_path):
                self.send_error(404, 'File not found')
                return

            # 读取文件
            with open(full_path, 'rb') as f:
                content = f.read()

            # 设置正确的Content-Type
            if file_path.endswith('.m3u8'):
                content_type = 'application/x-mpegURL'
            elif file_path.endswith('.ts'):
                content_type = 'video/MP2T'
            elif file_path.endswith('.mpd'):
                content_type = 'application/dash+xml'
            else:
                content_type = 'application/octet-stream'

            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(content)

        except Exception as e:
            self.send_error_response(f'服务文件失败: {e}')

    def send_json_response(self, data: Dict[str, Any]):
        """发送JSON响应"""
        response = json.dumps(data, ensure_ascii=False)
        self.send_response(200)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(response.encode('utf-8'))

    def send_error_response(self, message: str, status_code: int = 500):
        """发送错误响应"""
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps({
            'success': False,
            'error': message
        }, ensure_ascii=False).encode('utf-8'))

    def log_message(self, format, *args):
        """自定义日志格式"""
        logging.info(f"{self.address_string()} - {format % args}")


class WebAPIServer:
    """Web API服务器"""

    def __init__(self, config_file='ffmpeg_config.json'):
        self.stream_manager = StreamManager(config_file)
        self.http_port = 8080
        self.ws_port = 8081
        self.running = False

        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('WebAPIServer')

    def start(self):
        """启动服务器"""
        self.logger.info("启动Web API服务器")

        # 创建必要的目录
        os.makedirs('./hls_output', exist_ok=True)
        os.makedirs('./dash_output', exist_ok=True)
        os.makedirs('./public', exist_ok=True)

        # 启动HTTP服务器
        def run_http_server():
            def handler(*args, **kwargs):
                return APIRequestHandler(*args, stream_manager=self.stream_manager, **kwargs)

            httpd = HTTPServer(('', self.http_port), handler)
            self.logger.info(f"HTTP服务器启动: http://localhost:{self.http_port}")
            httpd.serve_forever()

        http_thread = threading.Thread(target=run_http_server)
        http_thread.daemon = True
        http_thread.start()

        # 启动WebSocket服务器
        def run_websocket_server():
            asyncio.run(self.start_websocket_server())

        ws_thread = threading.Thread(target=run_websocket_server)
        ws_thread.daemon = True
        ws_thread.start()

        self.running = True
        self.logger.info("Web API服务器启动完成")

    async def start_websocket_server(self):
        """启动WebSocket服务器"""
        self.logger.info(f"启动WebSocket服务器: ws://localhost:{self.ws_port}")

        async with websockets.serve(self.stream_manager.websocket_handler, "localhost", self.ws_port):
            await asyncio.Future()  # 永远运行

    def stop(self):
        """停止服务器"""
        self.logger.info("停止Web API服务器")
        self.running = False
        self.stream_manager.cleanup()

    def run(self):
        """运行服务器（阻塞）"""
        self.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("接收到停止信号")
            self.stop()


def main():
    """主函数"""
    print("RTSP到Web流媒体转换器 - API服务器")
    print("=" * 50)

    # 检查FFmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("警告: FFmpeg未安装或不在PATH中")
        else:
            print("FFmpeg版本检查通过")
    except FileNotFoundError:
        print("警告: 未找到FFmpeg，请先安装FFmpeg")

    # 启动服务器
    server = WebAPIServer()
    server.run()


if __name__ == "__main__":
    main()