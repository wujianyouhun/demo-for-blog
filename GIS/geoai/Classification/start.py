"""
GeoAI 遥感图像分类系统 — 一键启动脚本
=====================================
用法:
    conda activate geoai
    python start.py

功能:
  1. 检查并安装缺失依赖
  2. 检查模型文件是否存在（不存在时给出提示）
  3. 自动后台启动 FastAPI 后端服务 (localhost:8000)
  4. 自动打开浏览器前端页面
  5. 实时转发后端日志，Ctrl+C 优雅退出
"""

import os
import sys
import time
import signal
import threading
import subprocess
import webbrowser
from pathlib import Path

# ─── 彩色终端输出 ─────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BLUE   = "\033[94m"
DIM    = "\033[2m"

def c(text, color=RESET):      return f"{color}{text}{RESET}"
def ok(msg):    print(c(f"  ✅ {msg}", GREEN))
def warn(msg):  print(c(f"  ⚠️  {msg}", YELLOW))
def err(msg):   print(c(f"  ❌ {msg}", RED))
def info(msg):  print(c(f"  ℹ️  {msg}", CYAN))
def dim(msg):   print(c(f"     {msg}", DIM))
def step(msg):  print(c(f"\n{msg}", BOLD))

def banner():
    print(c(r"""
╔═══════════════════════════════════════════════════════╗
║      🛰️  GeoAI 遥感图像分类系统  v1.0.0               ║
║      EuroSAT · ResNet50 / ViT · FastAPI + Web UI      ║
╚═══════════════════════════════════════════════════════╝""", BOLD + BLUE))


# ─── 项目路径 ─────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent
FRONTEND_HTML = ROOT / "frontend" / "index.html"
CHECKPOINT    = ROOT / "checkpoints" / "best_model.pth"
ENV_FILE      = ROOT / ".env"

# ─── 默认参数（可被 .env 覆盖）────────────────────────────────────────────────
API_HOST = "127.0.0.1"
API_PORT = 8000

if ENV_FILE.exists():
    for raw in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().split("#")[0].strip()
        if key == "API_PORT":
            try:
                API_PORT = int(val)
            except ValueError:
                pass
        elif key == "API_HOST":
            if val:
                API_HOST = val

API_URL = f"http://{API_HOST}:{API_PORT}"


# ─── 依赖检查 ─────────────────────────────────────────────────────────────────
REQUIRED_PKGS = [
    ("fastapi",     "fastapi==0.110.0"),
    ("uvicorn",     "uvicorn[standard]==0.29.0"),
    ("torch",       "torch"),
    ("torchvision", "torchvision"),
    ("PIL",         "pillow"),
    ("dotenv",      "python-dotenv"),
    ("pydantic",    "pydantic"),
    ("multipart",   "python-multipart"),
]

def check_dependencies():
    step("📦 检查依赖包...")
    missing = []
    for mod, pkg in REQUIRED_PKGS:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
            warn(f"{mod} 未安装  →  {pkg}")

    if not missing:
        ok("所有依赖已就绪")
        return

    print(c(f"\n  🔧 自动安装缺失依赖: {', '.join(missing)}", YELLOW))
    ret = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet"] + missing,
        capture_output=True, text=True,
    )
    if ret.returncode == 0:
        ok("依赖安装成功")
    else:
        err("依赖安装失败，请手动执行:")
        print(f"     pip install {' '.join(missing)}")
        if ret.stderr:
            print(c(ret.stderr[:500], DIM))
        sys.exit(1)


# ─── 端口占用检测 ─────────────────────────────────────────────────────────────
def is_port_in_use(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


# ─── 等待后端健康就绪 ─────────────────────────────────────────────────────────
def wait_for_backend(timeout: int = 30) -> bool:
    import urllib.request
    deadline = time.time() + timeout
    dots = 0
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{API_URL}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.8)
        print(".", end="", flush=True)
        dots += 1
    return False


# ─── 后端日志转发（后台守护线程）─────────────────────────────────────────────
def _forward(stream, prefix: str, color: str):
    for raw in iter(stream.readline, b""):
        line = raw.decode("utf-8", errors="replace").rstrip()
        if line:
            print(c(f"  {prefix} {line}", color))


def attach_log_threads(proc: subprocess.Popen):
    for stream, prefix, color in [
        (proc.stdout, "[API]", RESET),
        (proc.stderr, "[ERR]", YELLOW),
    ]:
        threading.Thread(
            target=_forward, args=(stream, prefix, color), daemon=True
        ).start()


# ─── 启动后端子进程 ───────────────────────────────────────────────────────────
def start_backend():
    """返回 Popen 对象；端口已占用时返回 None（复用已有服务）"""
    if is_port_in_use(API_PORT):
        warn(f"端口 {API_PORT} 已被占用，复用已有服务")
        return None

    cmd = [
        sys.executable, "-m", "uvicorn",
        "backend.main:app",
        "--host", API_HOST,
        "--port", str(API_PORT),
        "--reload",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    attach_log_threads(proc)
    return proc


# ─── 打开浏览器 ───────────────────────────────────────────────────────────────
def open_browser():
    # 优先访问后端根路径（FastAPI 已挂载 frontend/index.html）
    try:
        webbrowser.open(API_URL)
        ok(f"浏览器已打开: {API_URL}")
        return
    except Exception:
        pass
    # 回退：直接打开本地文件
    if FRONTEND_HTML.exists():
        webbrowser.open(FRONTEND_HTML.as_uri())
        ok(f"浏览器已打开: {FRONTEND_HTML}")
    else:
        warn(f"前端文件不存在，请手动访问: {API_URL}")


# ─── 主流程 ───────────────────────────────────────────────────────────────────
def main():
    banner()

    info(f"Python  : {sys.version.split()[0]}")
    info(f"项目根  : {ROOT}")
    info(f"后端地址: {API_URL}")

    # Step 1 ── 依赖检查
    check_dependencies()

    # Step 2 ── 模型文件检查
    step("🧠 检查模型文件...")
    if CHECKPOINT.exists():
        size_mb = CHECKPOINT.stat().st_size / 1024 / 1024
        ok(f"模型已存在: {CHECKPOINT.name}  ({size_mb:.1f} MB)")
    else:
        warn(f"模型文件不存在: {CHECKPOINT}")
        warn("后端将以「模型未加载」状态运行，/predict 接口暂不可用")
        dim("训练命令: python scripts/train.py")

    # Step 3 ── 启动后端
    step(f"🚀 启动后端服务 ({API_URL}) ...")
    proc = start_backend()

    # Step 4 ── 等待就绪
    print(c("  ⏳ 等待后端就绪 ", CYAN), end="", flush=True)
    ready = wait_for_backend(timeout=30)
    print()   # 换行（结束点号输出）

    if ready:
        ok(f"后端运行中   : {API_URL}")
        ok(f"Swagger 文档 : {API_URL}/docs")
        ok(f"健康检查     : {API_URL}/health")
    else:
        if proc and proc.poll() is not None:
            err(f"后端进程异常退出 (code={proc.returncode})，请检查上方日志")
            sys.exit(1)
        else:
            warn("等待超时，服务可能仍在初始化，继续打开浏览器...")

    # Step 5 ── 打开前端
    step("🌐 打开前端界面...")
    time.sleep(0.4)
    open_browser()

    # 运行状态摘要
    print(c(f"""
  ┌───────────────────────────────────────────────────┐
  │  🟢 GeoAI 服务正在运行                             │
  │                                                   │
  │  前端界面  {API_URL:<39}│
  │  API 文档  {API_URL + "/docs":<39}│
  │  健康检查  {API_URL + "/health":<39}│
  │                                                   │
  │  按  Ctrl+C  可停止所有服务                         │
  └───────────────────────────────────────────────────┘""", CYAN))

    # Step 6 ── 信号处理 + 阻塞等待
    def _shutdown(sig, frame):
        print(c("\n\n  🛑 正在关闭服务...", YELLOW))
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        ok("所有服务已停止，再见！👋")
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    if proc:
        proc.wait()          # 阻塞直到后端子进程退出
    else:
        while True:          # 复用已有端口时，保持脚本存活以响应 Ctrl+C
            time.sleep(1)


if __name__ == "__main__":
    main()
