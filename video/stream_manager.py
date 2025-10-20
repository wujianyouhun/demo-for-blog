#!/usr/bin/env python3
"""
流媒体管理器 - 提供高级流媒体转换功能
"""

import json
import os
import subprocess
import threading
import time
import queue
import logging
from datetime import datetime
import asyncio
import websockets
import aiohttp
from typing import Dict, List, Optional

class StreamManager:
    """高级流媒体管理器"""

    def __init__(self, config_file='ffmpeg_config.json'):
        self.config = self.load_config(config_file)
        self.setup_logging()
        self.streams: Dict[str, dict] = {}
        self.message_queue = queue.Queue()
        self.stats = {
            'uptime': 0,
            'bytes_processed': 0,
            'errors': 0,
            'reconnections': 0
        }
        self.start_time = time.time()

    def load_config(self, config_file):
        """加载配置文件"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logging.error(f"配置文件 {config_file} 不存在")
            return self.get_default_config()

    def get_default_config(self):
        """获取默认配置"""
        return {
            "ffmpeg_path": "ffmpeg",
            "streams": {},
            "global_settings": {
                "timeout": 30,
                "reconnect_delay": 5,
                "max_reconnect_attempts": 10
            },
            "output_formats": {
                "hls": {
                    "enabled": True,
                    "segment_time": 4,
                    "segment_list_size": 6
                },
                "dash": {
                    "enabled": True,
                    "segment_duration": 4
                },
                "webm": {
                    "enabled": False,
                    "chunk_duration": 1
                }
            },
            "web_server": {
                "port": 8080,
                "ws_port": 8081,
                "enable_ssl": False
            }
        }

    def setup_logging(self):
        """设置日志"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler('stream_manager.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('StreamManager')

    def add_stream(self, stream_id: str, rtsp_url: str, config: Optional[dict] = None):
        """添加新的流媒体"""
        if stream_id in self.streams:
            raise ValueError(f"流 {stream_id} 已存在")

        stream_config = {
            'id': stream_id,
            'rtsp_url': rtsp_url,
            'status': 'stopped',
            'pid': None,
            'start_time': None,
            'config': config or self.get_default_stream_config(),
            'stats': {
                'uptime': 0,
                'bytes_processed': 0,
                'errors': 0,
                'reconnections': 0
            }
        }

        self.streams[stream_id] = stream_config
        self.logger.info(f"添加流: {stream_id}")
        return stream_id

    def remove_stream(self, stream_id: str):
        """移除流媒体"""
        if stream_id not in self.streams:
            raise ValueError(f"流 {stream_id} 不存在")

        self.stop_stream(stream_id)
        del self.streams[stream_id]
        self.logger.info(f"移除流: {stream_id}")

    def get_default_stream_config(self):
        """获取默认流配置"""
        return {
            'video_codec': 'libx264',
            'audio_codec': 'aac',
            'bitrate': '2000k',
            'resolution': '1280x720',
            'framerate': 25,
            'preset': 'ultrafast',
            'tune': 'zerolatency',
            'output_formats': ['hls', 'dash']
        }

    def build_ffmpeg_command(self, stream_id: str) -> List[str]:
        """构建FFmpeg命令"""
        stream = self.streams[stream_id]
        config = stream['config']

        cmd = [
            self.config['ffmpeg_path'],
            '-rtsp_transport', 'tcp',
            '-timeout', str(self.config['global_settings']['timeout'] * 1000000),
            '-i', stream['rtsp_url'],
            '-c:v', config['video_codec'],
            '-c:a', config['audio_codec'],
            '-b:v', config['bitrate'],
            '-s', config['resolution'],
            '-r', str(config['framerate']),
            '-preset', config['preset'],
            '-tune', config['tune'],
            '-g', str(int(config['framerate'] * 2)),
            '-keyint_min', str(int(config['framerate'])),
            '-movflags', '+faststart'
        ]

        # 添加输出格式
        outputs = []
        if 'hls' in config['output_formats']:
            hls_config = self.config['output_formats']['hls']
            hls_dir = f"./hls_output/{stream_id}"
            os.makedirs(hls_dir, exist_ok=True)

            outputs.extend([
                '-f', 'hls',
                '-hls_time', str(hls_config['segment_time']),
                '-hls_list_size', str(hls_config['segment_list_size']),
                '-hls_flags', 'delete_segments+append_list',
                '-hls_allow_cache', '0',
                '-hls_segment_type', 'mpegts',
                '-hls_segment_filename', f"{hls_dir}/segment_%03d.ts",
                f"{hls_dir}/stream.m3u8"
            ])

        if 'dash' in config['output_formats']:
            dash_config = self.config['output_formats']['dash']
            dash_dir = f"./dash_output/{stream_id}"
            os.makedirs(dash_dir, exist_ok=True)

            outputs.extend([
                '-f', 'dash',
                '-seg_duration', str(dash_config['segment_duration']),
                '-use_timeline', '1',
                '-use_template', '1',
                '-window_size', '6',
                '-extra_window_size', '0',
                '-remove_at_exit', '1',
                f"{dash_dir}/stream.mpd"
            ])

        cmd.extend(outputs)
        return cmd

    def start_stream(self, stream_id: str):
        """启动指定流"""
        if stream_id not in self.streams:
            raise ValueError(f"流 {stream_id} 不存在")

        stream = self.streams[stream_id]
        if stream['status'] == 'running':
            self.logger.warning(f"流 {stream_id} 已在运行中")
            return

        self.logger.info(f"启动流: {stream_id}")

        try:
            cmd = self.build_ffmpeg_command(stream_id)
            self.logger.info(f"FFmpeg命令: {' '.join(cmd)}")

            # 启动FFmpeg进程
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            stream['pid'] = process.pid
            stream['status'] = 'running'
            stream['start_time'] = time.time()
            stream['process'] = process

            # 启动监控线程
            monitor_thread = threading.Thread(
                target=self.monitor_stream,
                args=(stream_id,)
            )
            monitor_thread.daemon = True
            monitor_thread.start()

            self.message_queue.put({
                'type': 'stream_started',
                'stream_id': stream_id,
                'timestamp': datetime.now().isoformat()
            })

        except Exception as e:
            self.logger.error(f"启动流 {stream_id} 失败: {e}")
            stream['status'] = 'error'
            stream['stats']['errors'] += 1
            raise

    def stop_stream(self, stream_id: str):
        """停止指定流"""
        if stream_id not in self.streams:
            raise ValueError(f"流 {stream_id} 不存在")

        stream = self.streams[stream_id]
        if stream['status'] != 'running':
            return

        self.logger.info(f"停止流: {stream_id}")

        try:
            if 'process' in stream and stream['process']:
                stream['process'].terminate()
                try:
                    stream['process'].wait(timeout=5)
                except subprocess.TimeoutExpired:
                    stream['process'].kill()

            stream['status'] = 'stopped'
            stream['pid'] = None

            # 更新运行时间
            if stream['start_time']:
                stream['stats']['uptime'] += time.time() - stream['start_time']
                stream['start_time'] = None

            self.message_queue.put({
                'type': 'stream_stopped',
                'stream_id': stream_id,
                'timestamp': datetime.now().isoformat()
            })

        except Exception as e:
            self.logger.error(f"停止流 {stream_id} 失败: {e}")

    def monitor_stream(self, stream_id: str):
        """监控流状态"""
        stream = self.streams[stream_id]

        while stream['status'] == 'running':
            try:
                if 'process' not in stream or not stream['process']:
                    break

                process = stream['process']
                return_code = process.poll()

                if return_code is not None:
                    self.logger.warning(f"流 {stream_id} 进程退出，返回码: {return_code}")
                    stream['status'] = 'stopped'
                    stream['stats']['errors'] += 1

                    # 尝试重新连接
                    if self.config['global_settings']['max_reconnect_attempts'] > 0:
                        self.reconnect_stream(stream_id)
                    break

                # 读取输出
                output = process.stderr.readline()
                if output:
                    self.logger.debug(f"FFmpeg [{stream_id}]: {output.strip()}")

                    # 更新统计信息
                    if 'bytes' in output.lower():
                        stream['stats']['bytes_processed'] += 1024  # 估算值

                time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"监控流 {stream_id} 时发生错误: {e}")
                break

    def reconnect_stream(self, stream_id: str):
        """重新连接流"""
        stream = self.streams[stream_id]
        max_attempts = self.config['global_settings']['max_reconnect_attempts']
        delay = self.config['global_settings']['reconnect_delay']

        for attempt in range(max_attempts):
            self.logger.info(f"尝试重新连接流 {stream_id} ({attempt + 1}/{max_attempts})")

            time.sleep(delay)

            try:
                self.start_stream(stream_id)
                stream['stats']['reconnections'] += 1
                return True

            except Exception as e:
                self.logger.error(f"重新连接流 {stream_id} 失败: {e}")

        self.logger.error(f"流 {stream_id} 达到最大重连次数")
        return False

    def get_stream_status(self, stream_id: str) -> dict:
        """获取流状态"""
        if stream_id not in self.streams:
            raise ValueError(f"流 {stream_id} 不存在")

        stream = self.streams[stream_id].copy()

        # 计算实时运行时间
        if stream['status'] == 'running' and stream['start_time']:
            stream['current_uptime'] = time.time() - stream['start_time']
        else:
            stream['current_uptime'] = 0

        # 移除敏感信息
        if 'process' in stream:
            del stream['process']

        return stream

    def get_all_streams_status(self) -> dict:
        """获取所有流状态"""
        return {
            stream_id: self.get_stream_status(stream_id)
            for stream_id in self.streams
        }

    def get_system_stats(self) -> dict:
        """获取系统统计信息"""
        uptime = time.time() - self.start_time
        total_streams = len(self.streams)
        active_streams = sum(1 for s in self.streams.values() if s['status'] == 'running')

        return {
            'uptime': uptime,
            'total_streams': total_streams,
            'active_streams': active_streams,
            'total_bytes_processed': sum(s['stats']['bytes_processed'] for s in self.streams.values()),
            'total_errors': sum(s['stats']['errors'] for s in self.streams.values()),
            'total_reconnections': sum(s['stats']['reconnections'] for s in self.streams.values())
        }

    def get_stream_urls(self, stream_id: str) -> dict:
        """获取流的URL"""
        if stream_id not in self.streams:
            raise ValueError(f"流 {stream_id} 不存在")

        stream = self.streams[stream_id]
        config = stream['config']
        port = self.config['web_server']['port']

        urls = {}

        if 'hls' in config['output_formats']:
            urls['hls'] = f"http://localhost:{port}/hls/{stream_id}/stream.m3u8"

        if 'dash' in config['output_formats']:
            urls['dash'] = f"http://localhost:{port}/dash/{stream_id}/stream.mpd"

        return urls

    async def websocket_handler(self, websocket, path):
        """WebSocket处理器"""
        self.logger.info("WebSocket客户端连接")

        try:
            # 发送初始状态
            await websocket.send(json.dumps({
                'type': 'initial_status',
                'streams': self.get_all_streams_status(),
                'system_stats': self.get_system_stats()
            }))

            # 监听消息队列
            while True:
                try:
                    message = self.message_queue.get(timeout=1)
                    await websocket.send(json.dumps(message))
                except queue.Empty:
                    # 发送心跳
                    await websocket.send(json.dumps({
                        'type': 'heartbeat',
                        'timestamp': datetime.now().isoformat()
                    }))

        except websockets.exceptions.ConnectionClosed:
            self.logger.info("WebSocket客户端断开连接")
        except Exception as e:
            self.logger.error(f"WebSocket错误: {e}")

    async def start_websocket_server(self):
        """启动WebSocket服务器"""
        ws_port = self.config['web_server']['ws_port']

        self.logger.info(f"启动WebSocket服务器: ws://localhost:{ws_port}")

        async with websockets.serve(self.websocket_handler, "localhost", ws_port):
            await asyncio.Future()  # 永远运行

    def start_websocket_server_thread(self):
        """在单独的线程中启动WebSocket服务器"""
        def run_server():
            asyncio.run(self.start_websocket_server())

        thread = threading.Thread(target=run_server)
        thread.daemon = True
        thread.start()

    def cleanup(self):
        """清理资源"""
        self.logger.info("清理资源")

        # 停止所有流
        for stream_id in list(self.streams.keys()):
            self.stop_stream(stream_id)

        self.logger.info("资源清理完成")