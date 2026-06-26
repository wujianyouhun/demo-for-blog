#!/usr/bin/env python3
"""
GeoAI 要素提取 —— 一键启动脚本
================================
核心依赖: geoai-py (BuildingFootprintExtractor, GroundedSAM)

1. 安装 Python 依赖 (含 geoai-py)
2. 安装前端依赖
3. 启动后端 (port 8003)
4. 启动前端 (port 5176)
"""

import os
import sys
import subprocess
import time
import threading

ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(ROOT, "backend")
FRONTEND_DIR = os.path.join(ROOT, "frontend")
TIF_PATH = os.environ.get("TIF_PATH", os.path.join(ROOT, "data", "sample.tif"))


def run_cmd(cmd, cwd=None, shell=False):
    result = subprocess.run(cmd, cwd=cwd, shell=shell, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def step(msg):
    print(f"\n{'='*50}")
    print(f"  {msg}")
    print(f"{'='*50}")


def install_python_deps():
    step("Step 1/3: 安装 Python 依赖")
    req = os.path.join(BACKEND_DIR, "requirements.txt")
    rc, out, err = run_cmd(
        [sys.executable, "-m", "pip", "install", "-r", req, "-q"],
        cwd=BACKEND_DIR,
    )
    print("  Python 依赖安装完成" if rc == 0 else f"  警告: {err[:300]}")
    return rc == 0


def install_frontend_deps():
    step("Step 2/3: 安装前端依赖")
    node_modules = os.path.join(FRONTEND_DIR, "node_modules")
    if os.path.isdir(node_modules):
        print("  node_modules 已存在, 跳过")
        return True
    rc, out, err = run_cmd(["npm", "install"], cwd=FRONTEND_DIR, shell=True)
    print("  前端依赖安装完成" if rc == 0 else f"  警告: {err[:300]}")
    return rc == 0


def start_services():
    step("Step 3/3: 启动服务")

    env = os.environ.copy()
    env["TIF_PATH"] = TIF_PATH

    backend_proc = subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=BACKEND_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    def tail_backend():
        for line in iter(backend_proc.stdout.readline, b""):
            print(f"[后端] {line.decode('utf-8', errors='replace').rstrip()}")

    threading.Thread(target=tail_backend, daemon=True).start()

    print("  后端启动中... http://127.0.0.1:8003")
    time.sleep(2)

    frontend_proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=FRONTEND_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
    )

    def tail_frontend():
        for line in iter(frontend_proc.stdout.readline, b""):
            print(f"[前端] {line.decode('utf-8', errors='replace').rstrip()}")

    threading.Thread(target=tail_frontend, daemon=True).start()

    print("  前端启动中... http://127.0.0.1:5176")
    print(f"\n{'='*50}")
    print(f"  TIF 文件: {TIF_PATH}")
    print(f"  后端 API:  http://127.0.0.1:8003")
    print(f"  前端页面:  http://127.0.0.1:5176")
    print(f"{'='*50}\n")
    print("  按 Ctrl+C 停止所有服务\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在关闭服务...")
        backend_proc.terminate()
        frontend_proc.terminate()
        print("已关闭")


def main():
    print("╔══════════════════════════════════════════════╗")
    print("║   GeoAI 要素提取 (geoai-py) —— 项目启动器   ║")
    print("╚══════════════════════════════════════════════╝")

    if not os.path.exists(TIF_PATH):
        print(f"\n错误: TIF 文件不存在: {TIF_PATH}")
        print("请设置环境变量 TIF_PATH 或将文件放到 D:\\西安19级.tif")
        sys.exit(1)

    install_python_deps()
    install_frontend_deps()
    start_services()


if __name__ == "__main__":
    main()
