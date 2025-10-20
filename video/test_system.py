#!/usr/bin/env python3
"""
系统测试脚本
"""

import os
import sys
import subprocess
import time
import json
import requests
import threading
from urllib.parse import urlparse

def test_ffmpeg():
    """测试FFmpeg是否可用"""
    print("测试FFmpeg...")
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ FFmpeg可用")
            return True
        else:
            print("✗ FFmpeg不可用")
            return False
    except FileNotFoundError:
        print("✗ 未找到FFmpeg，请先安装FFmpeg")
        return False

def test_dependencies():
    """测试Python依赖"""
    print("测试Python依赖...")
    required_modules = ['websockets']
    missing_modules = []

    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module}可用")
        except ImportError:
            print(f"✗ {module}不可用")
            missing_modules.append(module)

    if missing_modules:
        print(f"请安装缺少的模块: pip install {' '.join(missing_modules)}")
        return False
    return True

def test_config_file():
    """测试配置文件"""
    print("测试配置文件...")
    config_file = 'ffmpeg_config.json'

    if not os.path.exists(config_file):
        print(f"✗ 配置文件 {config_file} 不存在")
        return False

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 检查必要的配置项
        required_keys = ['ffmpeg_path', 'rtsp_config', 'output_formats', 'stream_settings', 'web_server']
        for key in required_keys:
            if key not in config:
                print(f"✗ 配置文件缺少必要的键: {key}")
                return False

        print("✓ 配置文件格式正确")
        return True
    except json.JSONDecodeError as e:
        print(f"✗ 配置文件JSON格式错误: {e}")
        return False
    except Exception as e:
        print(f"✗ 配置文件读取失败: {e}")
        return False

def test_directories():
    """测试必要目录"""
    print("测试必要目录...")
    directories = ['./hls_output', './dash_output', './public']

    for directory in directories:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"✓ 创建目录: {directory}")
            except Exception as e:
                print(f"✗ 创建目录失败 {directory}: {e}")
                return False
        else:
            print(f"✓ 目录存在: {directory}")

    return True

def test_web_server():
    """测试Web服务器启动"""
    print("测试Web服务器...")

    # 由于Web服务器需要单独启动，这里只检查端口是否被占用
    import socket

    def is_port_in_use(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return False
            except:
                return True

    port = 8080
    if is_port_in_use(port):
        print(f"✗ 端口 {port} 已被占用")
        return False
    else:
        print(f"✓ 端口 {port} 可用")
        return True

def test_rtsp_connection():
    """测试RTSP连接"""
    print("测试RTSP连接...")

    try:
        with open('ffmpeg_config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)

        rtsp_url = config['rtsp_config']['url']

        # 使用FFmpeg测试RTSP连接
        cmd = [
            'ffmpeg',
            '-rtsp_transport', 'tcp',
            '-timeout', '10000000',  # 10秒超时
            '-i', rtsp_url,
            '-t', '1',  # 只测试1秒
            '-f', 'null',
            '-'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("✓ RTSP连接成功")
            return True
        else:
            print(f"✗ RTSP连接失败: {result.stderr}")
            return False

    except Exception as e:
        print(f"✗ RTSP连接测试失败: {e}")
        return False

def run_all_tests():
    """运行所有测试"""
    print("开始系统测试...")
    print("=" * 50)

    tests = [
        ("FFmpeg测试", test_ffmpeg),
        ("Python依赖测试", test_dependencies),
        ("配置文件测试", test_config_file),
        ("目录测试", test_directories),
        ("Web服务器测试", test_web_server),
        ("RTSP连接测试", test_rtsp_connection)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 20)

        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ 测试异常: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"测试完成: 通过 {passed} 个, 失败 {failed} 个")

    if failed == 0:
        print("✓ 所有测试通过，系统准备就绪")
        return True
    else:
        print("✗ 部分测试失败，请修复问题后再试")
        return False

def main():
    """主函数"""
    if len(sys.argv) > 1 and sys.argv[1] == '--fix':
        print("尝试修复常见问题...")

        # 安装依赖
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'websockets'])

        # 创建目录
        os.makedirs('./hls_output', exist_ok=True)
        os.makedirs('./dash_output', exist_ok=True)
        os.makedirs('./public', exist_ok=True)

        print("修复完成，请重新运行测试")
        return

    success = run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()