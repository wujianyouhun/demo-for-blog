#!/usr/bin/env python3
"""
ChangeDetection 项目一键启动脚本
- 自动检测并安装 Python / 前端依赖
- 同时启动后端 (FastAPI) 和前端 (Vite)
- Ctrl+C 优雅退出
"""
import subprocess
import sys
import os
import time
import signal
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
REPO_ROOT = PROJECT_ROOT.parent
SHARED_MODELS_DIR = Path(os.getenv("GEOAI_MODELS_DIR", str(REPO_ROOT / "models"))).expanduser()
if not SHARED_MODELS_DIR.is_absolute():
    SHARED_MODELS_DIR = (REPO_ROOT / SHARED_MODELS_DIR).resolve()
FRONTEND_DIR = PROJECT_ROOT / "frontend"
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"

BACKEND_HOST = "0.0.0.0"
BACKEND_PORT = 8000
FRONTEND_PORT = 5273

# ─── 颜色输出 ────────────────────────────────────────────────
RESET = "\033[0m"
BOLD  = "\033[1m"
RED   = "\033[91m"
GREEN = "\033[92m"
CYAN  = "\033[96m"
YELLOW = "\033[93m"

def info(msg):
    print(f"{CYAN}[INFO]{RESET} {msg}")

def ok(msg):
    print(f"{GREEN}[OK]{RESET}   {msg}")

def warn(msg):
    print(f"{YELLOW}[WARN]{RESET} {msg}")

def fail(msg):
    print(f"{RED}[FAIL]{RESET} {msg}")

# ─── 环境检查 ────────────────────────────────────────────────
def check_python():
    """检查 Python 版本 >= 3.10"""
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 10):
        fail(f"需要 Python >= 3.10，当前版本: {sys.version}")
        sys.exit(1)
    ok(f"Python {v.major}.{v.minor}.{v.micro}")

def check_node():
    """检查 Node.js 是否可用"""
    if not shutil.which("node"):
        fail("未检测到 Node.js，请先安装 Node.js 18+  (https://nodejs.org)")
        sys.exit(1)
    result = subprocess.run(["node", "--version"], capture_output=True, text=True)
    ok(f"Node.js {result.stdout.strip()}")

# ─── 依赖安装 ────────────────────────────────────────────────
def ensure_python_deps():
    """检测核心 Python 包，缺失则自动安装"""
    # 用几个关键包做探针，避免逐个 pip show
    probe_modules = [
        "fastapi", "uvicorn", "torch", "rasterio",
        "geopandas", "albumentations", "rich",
    ]
    missing = []
    for mod in probe_modules:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)

    if not missing:
        ok("Python 依赖已就绪")
        return

    info(f"检测到缺失依赖: {', '.join(missing)}，正在安装...")
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS), "-q"]
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if proc.returncode != 0:
        fail("Python 依赖安装失败，请手动运行: pip install -r requirements.txt")
        sys.exit(1)
    ok("Python 依赖安装完成")

def ensure_frontend_deps():
    """检测前端 node_modules，缺失则自动 npm install"""
    node_modules = FRONTEND_DIR / "node_modules"
    if node_modules.is_dir():
        ok("前端依赖已就绪")
        return

    info("正在安装前端依赖 (npm install)...")
    proc = subprocess.run(["npm", "install"], cwd=str(FRONTEND_DIR))
    if proc.returncode != 0:
        fail("前端依赖安装失败，请手动运行: cd frontend && npm install")
        sys.exit(1)
    ok("前端依赖安装完成")

# ─── 启动服务 ────────────────────────────────────────────────
processes = []

def start_backend():
    """启动 FastAPI 后端"""
    cmd = [
        sys.executable, "-m", "uvicorn",
        "backend.main:app",
        "--host", BACKEND_HOST,
        "--port", str(BACKEND_PORT),
        "--reload",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["TORCH_HOME"] = str(SHARED_MODELS_DIR)
    env["HF_HOME"] = str(SHARED_MODELS_DIR / "huggingface")
    env["HF_HUB_CACHE"] = str(SHARED_MODELS_DIR / "huggingface" / "hub")
    env["HUGGINGFACE_HUB_CACHE"] = str(SHARED_MODELS_DIR / "huggingface" / "hub")
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    info(f"启动后端  →  http://127.0.0.1:{BACKEND_PORT}  (API 文档: /docs)")
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
    )
    processes.append(("后端", proc))
    return proc

def start_frontend():
    """启动 Vite 前端开发服务器"""
    # Windows 下 npm 实际是 .cmd，需要 shell=True
    shell = os.name == "nt"
    cmd = "npm run dev" if shell else ["npm", "run", "dev"]

    info(f"启动前端  →  http://localhost:{FRONTEND_PORT}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(FRONTEND_DIR),
        shell=shell,
    )
    processes.append(("前端", proc))
    return proc

def shutdown(*_):
    """优雅关闭所有子进程"""
    print()
    info("正在关闭服务...")
    for name, proc in processes:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
                ok(f"{name} 已停止")
            except subprocess.TimeoutExpired:
                proc.kill()
                warn(f"{name} 强制终止")
    sys.exit(0)

# ─── 主流程 ──────────────────────────────────────────────────
def main():
    print(f"\n{BOLD}{'=' * 46}{RESET}")
    print(f"{BOLD}  ChangeDetection 项目启动器{RESET}")
    print(f"{BOLD}{'=' * 46}{RESET}\n")

    # 1. 环境检查
    check_python()
    check_node()
    print()

    # 2. 依赖检查 / 安装
    ensure_python_deps()
    ensure_frontend_deps()
    print()

    # 3. 创建数据目录
    for d in ["data/raw/time_a", "data/raw/time_b", "data/samples", "data/models", "data/output"]:
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)

    # 4. 注册信号处理
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    if os.name == "nt":
        signal.signal(signal.SIGBREAK, shutdown)

    # 5. 启动服务
    start_backend()
    time.sleep(2)  # 等后端就绪
    start_frontend()

    print(f"\n{BOLD}{'─' * 46}{RESET}")
    ok(f"后端 API 文档:  http://127.0.0.1:{BACKEND_PORT}/docs")
    ok(f"前端界面:       http://localhost:{FRONTEND_PORT}")
    print(f"{BOLD}{'─' * 46}{RESET}")
    info("按 Ctrl+C 停止所有服务\n")

    # 6. 等待，任一子进程退出则通知
    try:
        while True:
            for name, proc in processes:
                if proc.poll() is not None:
                    warn(f"{name} 已退出 (code={proc.returncode})")
                    shutdown()
            time.sleep(2)
    except KeyboardInterrupt:
        shutdown()

if __name__ == "__main__":
    main()
