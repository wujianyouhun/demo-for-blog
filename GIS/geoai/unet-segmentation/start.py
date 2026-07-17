from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="启动 U-Net GeoAI Web 项目")
    result.add_argument("--host", default="127.0.0.1")
    result.add_argument("--backend-port", type=int, default=8028)
    result.add_argument("--frontend-port", type=int, default=5188)
    result.add_argument("--no-install", action="store_true")
    result.add_argument("--no-browser", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    frontend = ROOT / "frontend"
    if not args.no_install and not (frontend / "node_modules").exists():
        subprocess.run(["npm", "install"], cwd=frontend, check=True, shell=os.name == "nt")
    backend = subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.main:app", "--host", args.host, "--port", str(args.backend_port)], cwd=ROOT)
    front = subprocess.Popen(["npm", "run", "dev", "--", "--host", args.host, "--port", str(args.frontend_port)], cwd=frontend, shell=os.name == "nt")
    url = f"http://{args.host}:{args.frontend_port}"
    print(f"U-Net Web: {url}\nAPI docs: http://{args.host}:{args.backend_port}/docs")
    if not args.no_browser:
        time.sleep(2)
        webbrowser.open(url)
    try:
        return backend.wait()
    except KeyboardInterrupt:
        for process in (backend, front):
            process.terminate()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
