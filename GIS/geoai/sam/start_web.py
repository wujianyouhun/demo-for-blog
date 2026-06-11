"""
SAM GeoAI 标注平台 — 一键启动脚本

同时启动 FastAPI 后端和 Vue3 前端开发服务器。

用法:
    python start_web.py
"""

import os
import sys
import subprocess
import time
import signal
from pathlib import Path

_project_root = Path(__file__).resolve().parent
_backend_dir = _project_root / "backend"
_frontend_dir = _project_root / "frontend"

BACKEND_PORT = 8000
FRONTEND_PORT = 5173


def main():
    print("=" * 50)
    print("  SAM GeoAI 标注平台 v2.0")
    print("=" * 50)
    print()

    processes = []

    def cleanup(sig=None, frame=None):
        print("\n正在关闭服务...")
        for p in processes:
            try:
                p.terminate()
                p.wait(timeout=5)
            except Exception:
                p.kill()
        print("已关闭。")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, cleanup)

    # 1. 启动后端
    print(f"[1/2] 启动 FastAPI 后端 (port {BACKEND_PORT})...")
    backend_cmd = [
        sys.executable, "-m", "uvicorn",
        "main:app",
        "--host", "0.0.0.0",
        "--port", str(BACKEND_PORT),
        "--reload",
    ]
    backend_proc = subprocess.Popen(
        backend_cmd,
        cwd=str(_backend_dir),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    processes.append(backend_proc)

    # 等待后端启动
    time.sleep(2)

    # 2. 启动前端
    print(f"[2/2] 启动 Vue3 前端 (port {FRONTEND_PORT})...")
    frontend_cmd = ["npm", "run", "dev"]
    frontend_proc = subprocess.Popen(
        frontend_cmd,
        cwd=str(_frontend_dir),
        shell=True if os.name == "nt" else False,
    )
    processes.append(frontend_proc)

    print()
    print(f"  后端 API:  http://127.0.0.1:{BACKEND_PORT}/docs")
    print(f"  前端页面:  http://127.0.0.1:{FRONTEND_PORT}")
    print()
    print("  按 Ctrl+C 停止所有服务")
    print("=" * 50)

    try:
        # 等待任一进程退出
        while True:
            for p in processes:
                ret = p.poll()
                if ret is not None:
                    print(f"\n进程退出 (code={ret})，正在关闭...")
                    cleanup()
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()
