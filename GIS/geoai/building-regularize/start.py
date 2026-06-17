"""
一键启动建筑物轮廓正则化工具。
同时启动后端 (FastAPI:8001) 和前端 (Vite:5174)。
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    backend_dir = os.path.join(ROOT, "backend")
    frontend_dir = os.path.join(ROOT, "frontend")

    # 检查后端依赖
    try:
        import fastapi, shapely, geopandas
    except ImportError:
        print("[!] 后端依赖缺失，正在安装...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r",
             os.path.join(backend_dir, "requirements.txt")],
            check=True,
        )

    # 检查演示数据
    data_dir = os.path.join(ROOT, "data")
    if not os.path.isdir(data_dir) or not os.listdir(data_dir):
        print("[!] 演示数据缺失，正在生成...")
        subprocess.run(
            [sys.executable, os.path.join(ROOT, "generate_demo_data.py")],
            check=True,
        )

    # 检查前端依赖
    node_modules = os.path.join(frontend_dir, "node_modules")
    if not os.path.isdir(node_modules):
        print("[!] 前端依赖缺失，正在安装...")
        subprocess.run(["npm", "install"], cwd=frontend_dir, check=True, shell=True)

    # 启动后端
    print("[*] 启动后端 FastAPI (端口 8001)...")
    backend_proc = subprocess.Popen(
        [sys.executable, os.path.join(backend_dir, "main.py")],
        cwd=backend_dir,
    )
    time.sleep(2)

    # 启动前端
    print("[*] 启动前端 Vite (端口 5174)...")
    frontend_proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=frontend_dir,
        shell=True,
    )

    print()
    print("=" * 50)
    print("  建筑物轮廓正则化工具已启动")
    print("  前端: http://localhost:5174")
    print("  后端: http://localhost:8001")
    print("  API文档: http://localhost:8001/docs")
    print("=" * 50)
    print()
    print("按 Ctrl+C 停止服务...")

    try:
        backend_proc.wait()
    except KeyboardInterrupt:
        print("\n[*] 正在停止服务...")
        backend_proc.terminate()
        frontend_proc.terminate()
        backend_proc.wait(timeout=5)
        frontend_proc.wait(timeout=5)
        print("[*] 服务已停止")


if __name__ == "__main__":
    main()
