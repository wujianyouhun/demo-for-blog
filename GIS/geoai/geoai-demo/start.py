#!/usr/bin/env python3
"""
start.py - GeoAI Demo 统一启动脚本

自动检查环境、安装依赖、启动前后端服务。

Usage:
    python start.py
    python start.py --skip-check
    python start.py --backend-only
    python start.py --frontend-only
"""

import argparse
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

# ── ANSI Colors ──────────────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"


def info(msg):
    print(f"{BLUE}[INFO]{RESET} {msg}")


def ok(msg):
    print(f"{GREEN}[ OK ]{RESET} {msg}")


def warn(msg):
    print(f"{YELLOW}[WARN]{RESET} {msg}")


def fail(msg):
    print(f"{RED}[FAIL]{RESET} {msg}")


def header():
    print(f"\n{BOLD}{CYAN}{'='*60}")
    print("  GeoAI Demo - 土地覆盖分类系统")
    print(f"{'='*60}{RESET}\n")


# ── Checks ───────────────────────────────────────────────────────────────────

def check_python_version():
    """Check Python >= 3.10."""
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 10):
        fail(f"Python >= 3.10 required, found {major}.{minor}")
        return False
    ok(f"Python {major}.{minor}")
    return True


def check_nodejs():
    """Check Node.js availability."""
    try:
        result = subprocess.run(
            ["node", "-v"], capture_output=True, text=True, timeout=10
        )
        version = result.stdout.strip()
        ok(f"Node.js {version}")
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        fail("Node.js not found. Please install Node.js >= 18")
        return False


def check_command(cmd, label):
    """Check if a command exists."""
    if shutil.which(cmd):
        ok(f"{label} ({cmd})")
        return True
    else:
        fail(f"{label} not found ({cmd})")
        return False


def check_python_packages():
    """Check key Python packages."""
    packages = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "numpy": "numpy",
    }

    # Optional packages (warn but don't fail)
    optional_packages = {
        "torch": "torch",
        "rasterio": "rasterio",
        "segmentation_models_pytorch": "segmentation-models-pytorch",
        "albumentations": "albumentations",
        "shapely": "shapely",
    }

    all_ok = True
    missing = []

    for import_name, pip_name in packages.items():
        try:
            __import__(import_name)
            ok(f"  {pip_name}")
        except ImportError:
            fail(f"  {pip_name} (required)")
            missing.append(pip_name)
            all_ok = False

    info("Optional packages:")
    for import_name, pip_name in optional_packages.items():
        try:
            __import__(import_name)
            ok(f"  {pip_name}")
        except ImportError:
            warn(f"  {pip_name} (optional, demo mode available)")

    return all_ok, missing


def check_frontend_deps(project_root):
    """Check frontend node_modules."""
    nm_dir = project_root / "frontend" / "node_modules"
    if nm_dir.exists():
        ok("Frontend node_modules")
        return True
    else:
        warn("Frontend node_modules not found")
        return False


# ── Setup ────────────────────────────────────────────────────────────────────

def install_python_deps(project_root):
    """Install Python dependencies from requirements.txt."""
    req_file = project_root / "requirements.txt"
    if not req_file.exists():
        warn("requirements.txt not found, skipping pip install")
        return False

    info("Installing Python dependencies...")
    try:
        subprocess.run(
            [sys.executable, "pip", "install", "-r", str(req_file), "-q"],
            check=True,
            timeout=300,
        )
        ok("Python dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        fail(f"pip install failed: {e}")
        return False


def install_frontend_deps(project_root):
    """Install frontend npm dependencies."""
    frontend_dir = project_root / "frontend"
    if not (frontend_dir / "package.json").exists():
        warn("frontend/package.json not found")
        return False

    info("Installing frontend dependencies (npm install)...")
    try:
        subprocess.run(
            ["npm", "install"],
            cwd=str(frontend_dir),
            check=True,
            timeout=300,
            shell=(platform.system() == "Windows"),
        )
        ok("Frontend dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        fail(f"npm install failed: {e}")
        return False


def create_directories(project_root):
    """Create required data directories."""
    dirs = ["data", "output", "output/models", "output/results"]
    for d in dirs:
        path = project_root / d
        path.mkdir(parents=True, exist_ok=True)
    ok("Data directories created")


# ── Service Management ───────────────────────────────────────────────────────

def start_backend(project_root):
    """Start FastAPI backend with uvicorn."""
    backend_dir = project_root / "backend"
    main_module = "backend.main:app"

    # Check if backend exists
    if not (backend_dir / "main.py").exists():
        warn("backend/main.py not found, backend will not start")
        warn("The API endpoints are defined in the backend module")
        return None

    info("Starting backend (uvicorn on port 8000)...")

    cmd = [
        sys.executable, "-m", "uvicorn",
        main_module,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload",
    ]

    proc = subprocess.Popen(
        cmd,
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    time.sleep(1)
    if proc.poll() is None:
        ok("Backend started (PID: {})".format(proc.pid))
    else:
        fail("Backend failed to start")
        return None

    return proc


def start_frontend(project_root):
    """Start Vue frontend with Vite."""
    frontend_dir = project_root / "frontend"

    if not (frontend_dir / "package.json").exists():
        warn("frontend/package.json not found")
        return None

    info("Starting frontend (Vite on port 5173)...")

    shell = platform.system() == "Windows"
    cmd = ["npm", "run", "dev"]

    proc = subprocess.Popen(
        cmd,
        cwd=str(frontend_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=shell,
    )

    time.sleep(2)
    if proc.poll() is None:
        ok("Frontend started (PID: {})".format(proc.pid))
    else:
        fail("Frontend failed to start")
        return None

    return proc


def show_urls():
    """Display service URLs."""
    print(f"\n{BOLD}{GREEN}{'─'*60}")
    print("  Services running:")
    print(f"{'─'*60}{RESET}")
    print(f"  {CYAN}Backend  (API):{RESET}   http://localhost:8000")
    print(f"  {CYAN}Frontend (UI):{RESET}    http://localhost:5173")
    print(f"  {CYAN}API Docs:{RESET}         http://localhost:8000/docs")
    print(f"{BOLD}{GREEN}{'─'*60}{RESET}")
    print(f"\n  Press {BOLD}Ctrl+C{RESET} to stop all services\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GeoAI Demo - 统一启动脚本")
    parser.add_argument("--skip-check", action="store_true", help="跳过环境检查")
    parser.add_argument("--backend-only", action="store_true", help="仅启动后端")
    parser.add_argument("--frontend-only", action="store_true", help="仅启动前端")
    parser.add_argument("--no-auto-install", action="store_true", help="不自动安装依赖")

    args = parser.parse_args()

    project_root = Path(__file__).parent.resolve()
    header()

    processes = []

    # ── Environment checks ──────────────────────────────────────────────────

    if not args.skip_check:
        info("Checking environment...")
        print()

        checks_passed = True

        if not check_python_version():
            checks_passed = False

        if not args.frontend_only:
            pkgs_ok, missing = check_python_packages()
            if not pkgs_ok and not args.no_auto_install:
                if install_python_deps(project_root):
                    pkgs_ok, _ = check_python_packages()

        if not args.backend_only:
            node_ok = check_nodejs()
            if node_ok:
                if not check_frontend_deps(project_root) and not args.no_auto_install:
                    install_frontend_deps(project_root)

        print()

    # ── Create directories ──────────────────────────────────────────────────

    create_directories(project_root)
    print()

    # ── Start services ──────────────────────────────────────────────────────

    backend_proc = None
    frontend_proc = None

    if not args.frontend_only:
        backend_proc = start_backend(project_root)
        if backend_proc:
            processes.append(backend_proc)

    if not args.backend_only:
        frontend_proc = start_frontend(project_root)
        if frontend_proc:
            processes.append(frontend_proc)

    if not processes:
        fail("No services started")
        sys.exit(1)

    show_urls()

    # ── Wait and handle Ctrl+C ──────────────────────────────────────────────

    try:
        while True:
            # Check if processes are still running
            for proc in processes[:]:
                if proc.poll() is not None:
                    warn(f"Process {proc.pid} exited with code {proc.returncode}")
                    processes.remove(proc)

            if not processes:
                warn("All services stopped")
                break

            time.sleep(2)

    except KeyboardInterrupt:
        print(f"\n{YELLOW}Shutting down...{RESET}")

        for proc in processes:
            try:
                if platform.system() == "Windows":
                    proc.terminate()
                else:
                    os.kill(proc.pid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass

        # Wait for graceful shutdown
        for proc in processes:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

        ok("All services stopped")


if __name__ == "__main__":
    main()
