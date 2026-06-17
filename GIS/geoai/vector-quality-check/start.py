#!/usr/bin/env python3
"""
矢量数据质量自动检查 —— 一键启动脚本
=====================================
功能：
  1. 检查并安装 Python 依赖
  2. 生成演示数据（如果不存在）
  3. 安装前端依赖
  4. 启动后端 (port 8002)
  5. 启动前端 (port 5175)
"""

import os
import sys
import subprocess
import time
import threading

ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(ROOT, "backend")
FRONTEND_DIR = os.path.join(ROOT, "frontend")
DATA_DIR = os.path.join(ROOT, "data")


def run_cmd(cmd, cwd=None, shell=False):
    """运行命令并返回结果"""
    result = subprocess.run(
        cmd, cwd=cwd, shell=shell,
        capture_output=True, text=True
    )
    return result.returncode, result.stdout, result.stderr


def step(msg):
    print(f"\n{'='*50}")
    print(f"  {msg}")
    print(f"{'='*50}")


def install_python_deps():
    step("Step 1/4: 安装 Python 依赖")
    req = os.path.join(BACKEND_DIR, "requirements.txt")
    rc, out, err = run_cmd(
        [sys.executable, "-m", "pip", "install", "-r", req, "-q"],
        cwd=BACKEND_DIR,
    )
    if rc == 0:
        print("  Python 依赖安装完成")
    else:
        print(f"  警告: 安装可能有问题\n  {err[:300]}")
    return rc == 0


def generate_demo_data():
    step("Step 2/4: 生成演示数据")
    if os.path.isdir(DATA_DIR) and len([f for f in os.listdir(DATA_DIR) if f.endswith(".geojson")]) >= 3:
        print("  演示数据已存在, 跳过")
        return True

    gen_script = os.path.join(ROOT, "generate_demo_data.py")
    rc, out, err = run_cmd([sys.executable, gen_script])
    print(out)
    if rc != 0:
        print(f"  生成失败: {err[:300]}")
    return rc == 0


def install_frontend_deps():
    step("Step 3/4: 安装前端依赖")
    node_modules = os.path.join(FRONTEND_DIR, "node_modules")
    if os.path.isdir(node_modules):
        print("  node_modules 已存在, 跳过")
        return True

    rc, out, err = run_cmd(["npm", "install"], cwd=FRONTEND_DIR, shell=True)
    if rc == 0:
        print("  前端依赖安装完成")
    else:
        print(f"  警告: {err[:300]}")
    return rc == 0


def start_services():
    step("Step 4/4: 启动服务")

    # 启动后端
    backend_proc = subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=BACKEND_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    def tail_backend():
        for line in iter(backend_proc.stdout.readline, b""):
            print(f"[后端] {line.decode('utf-8', errors='replace').rstrip()}")

    t1 = threading.Thread(target=tail_backend, daemon=True)
    t1.start()

    print("  后端启动中... http://127.0.0.1:8002")
    time.sleep(2)

    # 启动前端
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

    t2 = threading.Thread(target=tail_frontend, daemon=True)
    t2.start()

    print("  前端启动中... http://127.0.0.1:5175")
    print(f"\n{'='*50}")
    print("  服务已启动!")
    print("  后端 API:  http://127.0.0.1:8002")
    print("  前端页面:  http://127.0.0.1:5175")
    print(f"{'='*50}")
    print("\n  按 Ctrl+C 停止所有服务\n")

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
    print("║   矢量数据质量自动检查 —— 项目启动器         ║")
    print("╚══════════════════════════════════════════════╝")

    install_python_deps()
    generate_demo_data()
    install_frontend_deps()
    start_services()


if __name__ == "__main__":
    main()
