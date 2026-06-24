"""
SAM GeoAI 标注平台 — 一键启动脚本

同时启动 FastAPI 后端和 Vue3 前端开发服务器。

用法:
    python start_web.py
"""

# 导入标准库模块
import os          # 操作系统接口，用于环境变量和进程管理
import sys         # 系统相关的参数和函数
import subprocess  # 子进程管理，用于启动后端/前端服务
import time        # 时间相关函数，用于延时等待
import signal      # 信号处理，用于捕获 Ctrl+C 等中断信号
from pathlib import Path  # 面向对象的路径操作

# =============================================================================
# 路径与端口配置
# =============================================================================

# 获取脚本所在目录作为项目根目录
_project_root = Path(__file__).resolve().parent
# 后端代码目录（FastAPI）
_backend_dir = _project_root / "backend"
# 前端代码目录（Vue3）
_frontend_dir = _project_root / "frontend"

# 后端服务监听端口
BACKEND_PORT = 8000
# 前端开发服务器端口
FRONTEND_PORT = 5217


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主入口函数：依次启动后端和前端服务，并监控进程状态。"""

    # 打印启动横幅
    print("=" * 50)
    print("  SAM GeoAI 标注平台 v2.0")
    print("=" * 50)
    print()

    # 用于保存所有子进程引用，以便统一关闭
    processes = []

    # -------------------------------------------------------------------------
    # 进程清理函数
    # -------------------------------------------------------------------------
    def cleanup(sig=None, frame=None):
        """
        关闭所有已启动的子进程。

        先尝试优雅终止（terminate），等待最多 5 秒；
        若超时则强制结束（kill）。
        """
        print("\n正在关闭服务...")
        for p in processes:
            try:
                p.terminate()          # 发送终止信号
                p.wait(timeout=5)      # 等待进程退出
            except Exception:
                p.kill()               # 强制结束
        print("已关闭。")
        sys.exit(0)

    # 注册信号处理器：捕获 Ctrl+C（SIGINT）和终止信号（SIGTERM）
    signal.signal(signal.SIGINT, cleanup)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, cleanup)

    # -------------------------------------------------------------------------
    # 1. 启动 FastAPI 后端
    # -------------------------------------------------------------------------
    print(f"[1/2] 启动 FastAPI 后端 (port {BACKEND_PORT})...")

    # 构造 uvicorn 启动命令
    # --reload 表示代码变更时自动重载（仅开发环境使用）
    backend_cmd = [
        sys.executable, "-m", "uvicorn",   # 使用当前 Python 解释器运行 uvicorn
        "main:app",                         # 入口模块:FastAPI 应用实例
        "--host", "0.0.0.0",               # 监听所有网络接口
        "--port", str(BACKEND_PORT),       # 指定监听端口
        "--reload",                         # 启用热重载
    ]

    # 以后台子进程方式启动后端
    backend_proc = subprocess.Popen(
        backend_cmd,
        cwd=str(_backend_dir),                        # 在后端目录下执行
        env={**os.environ, "PYTHONUNBUFFERED": "1"},  # 设置无缓冲输出，确保日志实时打印
    )
    processes.append(backend_proc)

    # 等待后端初始化完成，避免前端请求时后端尚未就绪
    time.sleep(2)

    # -------------------------------------------------------------------------
    # 2. 启动 Vue3 前端
    # -------------------------------------------------------------------------
    print(f"[2/2] 启动 Vue3 前端 (port {FRONTEND_PORT})...")

    frontend_cmd = ["npm", "run", "dev"]

    # 在 Windows 上使用 shell=True，确保 npm 命令能正确解析
    frontend_proc = subprocess.Popen(
        frontend_cmd,
        cwd=str(_frontend_dir),                          # 在前端目录下执行
        shell=True if os.name == "nt" else False,        # Windows 需要 shell 模式
    )
    processes.append(frontend_proc)

    # -------------------------------------------------------------------------
    # 打印访问地址
    # -------------------------------------------------------------------------
    print()
    print(f"  后端 API:  http://127.0.0.1:{BACKEND_PORT}/docs")
    print(f"  前端页面:  http://127.0.0.1:{FRONTEND_PORT}")
    print()
    print("  按 Ctrl+C 停止所有服务")
    print("=" * 50)

    # -------------------------------------------------------------------------
    # 守护循环：监控子进程状态
    # -------------------------------------------------------------------------
    try:
        # 无限循环，直到任一进程异常退出
        while True:
            for p in processes:
                ret = p.poll()          # 检查进程是否已结束
                if ret is not None:     # 如果进程已退出（返回码不为 None）
                    print(f"\n进程退出 (code={ret})，正在关闭...")
                    cleanup()
            time.sleep(1)               # 每秒轮询一次
    except KeyboardInterrupt:
        # 捕获键盘中断（Ctrl+C），执行清理
        cleanup()


# =============================================================================
# 脚本入口
# =============================================================================

if __name__ == "__main__":
    main()
