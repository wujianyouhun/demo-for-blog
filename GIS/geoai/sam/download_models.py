"""
GeoAI SAM 项目 — 模型一键下载脚本

下载所有 SAM 模型权重到项目 models/ 目录:
    - SAM1: vit_b / vit_l / vit_h
    - SAM2 (sam2.1): hiera_base_plus / hiera_large / hiera_huge
    - SAM3: 与 SAM2 共享权重（sam2.1 系列）

用法:
    python download_models.py              # 下载全部
    python download_models.py --sam1       # 仅下载 SAM1
    python download_models.py --sam2       # 仅下载 SAM2/SAM3
    python download_models.py --vit_b      # 仅下载 vit_b 级别
    python download_models.py --vit_l      # 仅下载 vit_l 级别
    python download_models.py --vit_h      # 仅下载 vit_h 级别
    python download_models.py --list       # 列出所有模型
"""

import os
import sys
import time
import argparse
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
MODEL_DIR = Path(os.getenv("GEOAI_MODELS_DIR", str(REPO_ROOT / "models"))).expanduser()
if not MODEL_DIR.is_absolute():
    MODEL_DIR = (REPO_ROOT / MODEL_DIR).resolve()
MODEL_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("TORCH_HOME", str(MODEL_DIR / "torch"))
os.environ.setdefault("HF_HOME", str(MODEL_DIR / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(MODEL_DIR / "huggingface" / "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(MODEL_DIR / "huggingface" / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(MODEL_DIR / "huggingface" / "transformers"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(MODEL_DIR / "sentence_transformers"))
os.environ.setdefault("CLIP_CACHE", str(MODEL_DIR / "clip"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ================================================================
# 模型定义
# ================================================================

MODELS = {
    # --- SAM1 ---
    "sam1_vit_b": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "filename": "sam_vit_b_01ec64.pth",
        "size_mb": 375,
        "version": "SAM1",
        "type": "vit_b",
        "description": "SAM1 ViT-B 轻量模型 (91M 参数)",
    },
    "sam1_vit_l": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "filename": "sam_vit_l_0b3195.pth",
        "size_mb": 1190,
        "version": "SAM1",
        "type": "vit_l",
        "description": "SAM1 ViT-L 中等模型 (308M 参数, 推荐)",
    },
    "sam1_vit_h": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "filename": "sam_vit_h_4b8939.pth",
        "size_mb": 2445,
        "version": "SAM1",
        "type": "vit_h",
        "description": "SAM1 ViT-H 大型模型 (636M 参数, 精度最高)",
    },

    # --- SAM2 (sam2.1) ---
    # 注意: SAM2.1 没有 huge 变体
    # vit_b → hiera_base_plus, vit_l → hiera_large, vit_h → 复用 hiera_large
    "sam2_vit_b": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        "filename": "sam2.1_hiera_base_plus.pt",
        "size_mb": 308,
        "version": "SAM2",
        "type": "vit_b",
        "description": "SAM2.1 Hiera Base+ (80.8M 参数)",
    },
    "sam2_vit_l": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "filename": "sam2.1_hiera_large.pt",
        "size_mb": 856,
        "version": "SAM2",
        "type": "vit_l",
        "description": "SAM2.1 Hiera Large (224.4M 参数)",
    },
    "sam2_vit_h": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "filename": "sam2.1_hiera_large.pt",  # SAM2.1 无 huge, 复用 large
        "size_mb": 856,
        "version": "SAM2",
        "type": "vit_h",
        "description": "SAM2.1 Hiera Large (复用, SAM2.1 无 huge 变体)",
        "shared_with": "sam2_vit_l",
    },

    # --- SAM3 ---
    # SAM3 使用与 SAM2 相同的 sam2.1 系列权重
    "sam3_vit_b": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        "filename": "sam2.1_hiera_base_plus.pt",
        "size_mb": 308,
        "version": "SAM3",
        "type": "vit_b",
        "description": "SAM3 (复用 SAM2.1 Hiera Base+ 权重)",
        "shared_with": "sam2_vit_b",
    },
    "sam3_vit_l": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "filename": "sam2.1_hiera_large.pt",
        "size_mb": 856,
        "version": "SAM3",
        "type": "vit_l",
        "description": "SAM3 (复用 SAM2.1 Hiera Large 权重)",
        "shared_with": "sam2_vit_l",
    },
    "sam3_vit_h": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "filename": "sam2.1_hiera_large.pt",
        "size_mb": 856,
        "version": "SAM3",
        "type": "vit_h",
        "description": "SAM3 (复用 SAM2.1 Hiera Large 权重)",
        "shared_with": "sam2_vit_l",
    },
}


def format_size(size_mb: int) -> str:
    """格式化文件大小。"""
    if size_mb >= 1024:
        return f"{size_mb / 1024:.1f}GB"
    return f"{size_mb}MB"


def download_with_progress(url: str, local_path: str, expected_size_mb: int = 0) -> bool:
    """下载文件并显示进度，支持断点续传。"""
    filename = os.path.basename(local_path)

    # 检查文件是否已存在且完整
    if os.path.exists(local_path):
        actual_size = os.path.getsize(local_path)
        actual_mb = actual_size / (1024 * 1024)
        if expected_size_mb > 0 and actual_mb >= expected_size_mb * 0.95:
            print(f"  [跳过] 文件已存在且完整: {filename} ({actual_mb:.1f}MB)")
            return True
        else:
            print(f"  [续传] 文件不完整: {actual_mb:.1f}MB / 期望 ~{expected_size_mb}MB")

    print(f"  下载: {url}")
    print(f"  保存: {local_path}")
    print(f"  大小: ~{format_size(expected_size_mb)}")

    # 优先使用 curl.exe (支持断点续传和自动重试)
    try:
        import shutil
        curl_path = shutil.which("curl.exe") or shutil.which("curl")
        if curl_path:
            print(f"  方法: curl (断点续传 + 自动重试)")
            import subprocess
            result = subprocess.run(
                [
                    curl_path, "-L", "-C", "-",
                    "--retry", "5",
                    "--retry-delay", "10",
                    "--retry-max-time", "7200",
                    "-o", local_path,
                    url,
                ],
                check=False,
            )
            if result.returncode == 0 and os.path.exists(local_path):
                actual_mb = os.path.getsize(local_path) / (1024 * 1024)
                print(f"  完成: {actual_mb:.1f}MB")
                return True
            print(f"  curl 返回码: {result.returncode}")
    except Exception as e:
        print(f"  curl 失败: {e}")

    # 方法 2: torch.hub (有进度条)
    try:
        import torch
        print(f"  方法: torch.hub.download_url_to_file")
        torch.hub.download_url_to_file(url, local_path, progress=True)
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            actual_mb = os.path.getsize(local_path) / (1024 * 1024)
            print(f"  完成: {actual_mb:.1f}MB")
            return True
    except Exception as e:
        print(f"  torch.hub 失败: {e}")
        if os.path.exists(local_path):
            os.remove(local_path)

    # 方法 3: requests (流式下载)
    try:
        import requests
        from tqdm import tqdm

        print(f"  方法: requests stream")
        session = requests.Session()
        response = session.get(url, stream=True, timeout=(30, 300))
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(local_path, "wb") as f:
            with tqdm(total=total, unit="B", unit_scale=True, desc=f"  {filename}") as pbar:
                for chunk in response.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        pbar.update(len(chunk))

        actual_mb = os.path.getsize(local_path) / (1024 * 1024)
        if total > 0 and downloaded < total * 0.95:
            print(f"  下载不完整: {downloaded / (1024*1024):.1f}MB / {total / (1024*1024):.1f}MB")
            os.remove(local_path)
            return False

        print(f"  完成: {actual_mb:.1f}MB")
        return True

    except Exception as e:
        print(f"  requests 失败: {e}")
        if os.path.exists(local_path):
            os.remove(local_path)

    return False


def download_model(key: str) -> bool:
    """下载单个模型。"""
    info = MODELS[key]
    local_path = str(MODEL_DIR / info["filename"])

    # 如果是共享文件，检查是否已通过其他模型下载
    shared_with = info.get("shared_with")
    if shared_with and os.path.exists(local_path):
        actual_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"  [跳过] 共享文件已存在: {info['filename']} ({actual_mb:.1f}MB)")
        return True

    print(f"\n{'='*60}")
    print(f"  下载: {key}")
    print(f"  {info['description']}")
    print(f"{'='*60}")

    success = download_with_progress(
        url=info["url"],
        local_path=local_path,
        expected_size_mb=info["size_mb"],
    )

    if success:
        print(f"  [成功] {key}")
    else:
        print(f"  [失败] {key}")
        print(f"  手动下载: {info['url']}")
        print(f"  保存到: {local_path}")

    return success


def list_models():
    """列出所有模型信息。"""
    print("=" * 80)
    print("  GeoAI SAM 项目 — 模型列表")
    print("=" * 80)
    print()

    current_version = None
    for key, info in MODELS.items():
        if info["version"] != current_version:
            current_version = info["version"]
            print(f"--- {current_version} ---")

        local_path = MODEL_DIR / info["filename"]
        status = "已下载" if local_path.exists() else "未下载"
        size_str = format_size(info["size_mb"])

        shared = f" (共享: {info['shared_with']})" if "shared_with" in info else ""

        print(f"  {key:20s} | {info['type']:6s} | {size_str:>8s} | {status:6s} | {info['description']}{shared}")

    print()
    total_size = sum(info["size_mb"] for info in MODELS.values() if "shared_with" not in info)
    print(f"  总大小 (去重): ~{format_size(total_size)}")
    print(f"  模型目录: {MODEL_DIR}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="GeoAI SAM 模型下载工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python download_models.py              # 下载全部模型
    python download_models.py --sam1       # 仅下载 SAM1
    python download_models.py --sam2       # 仅下载 SAM2/SAM3
    python download_models.py --vit_b      # 仅下载 vit_b 级别
    python download_models.py --vit_l      # 仅下载 vit_l 级别
    python download_models.py --vit_h      # 仅下载 vit_h 级别
    python download_models.py --list       # 列出所有模型
        """,
    )
    parser.add_argument("--sam1", action="store_true", help="仅下载 SAM1 模型")
    parser.add_argument("--sam2", action="store_true", help="仅下载 SAM2/SAM3 模型")
    parser.add_argument("--vit_b", action="store_true", help="仅下载 vit_b 级别")
    parser.add_argument("--vit_l", action="store_true", help="仅下载 vit_l 级别")
    parser.add_argument("--vit_h", action="store_true", help="仅下载 vit_h 级别")
    parser.add_argument("--list", action="store_true", help="列出所有模型信息")
    args = parser.parse_args()

    if args.list:
        list_models()
        return

    # 确定要下载的模型
    keys_to_download = []

    for key, info in MODELS.items():
        # 版本过滤
        if args.sam1 and info["version"] != "SAM1":
            continue
        if args.sam2 and info["version"] not in ("SAM2", "SAM3"):
            continue

        # 类型过滤
        if args.vit_b and info["type"] != "vit_b":
            continue
        if args.vit_l and info["type"] != "vit_l":
            continue
        if args.vit_h and info["type"] != "vit_h":
            continue

        keys_to_download.append(key)

    # 如果没有指定过滤，下载全部
    if not any([args.sam1, args.sam2, args.vit_b, args.vit_l, args.vit_h]):
        keys_to_download = list(MODELS.keys())

    if not keys_to_download:
        print("没有匹配的模型，请检查参数。")
        return

    print("=" * 60)
    print("  GeoAI SAM 模型下载工具")
    print("=" * 60)
    print(f"  模型目录: {MODEL_DIR}")
    print(f"  待下载: {len(keys_to_download)} 个模型")

    # 统计总大小 (去重)
    seen_files = set()
    total_mb = 0
    for key in keys_to_download:
        filename = MODELS[key]["filename"]
        if filename not in seen_files:
            seen_files.add(filename)
            total_mb += MODELS[key]["size_mb"]
    print(f"  总大小: ~{format_size(total_mb)}")
    print()

    # 逐个下载
    success_count = 0
    fail_count = 0
    for i, key in enumerate(keys_to_download, 1):
        print(f"\n[{i}/{len(keys_to_download)}] 开始下载 {key}...")
        if download_model(key):
            success_count += 1
        else:
            fail_count += 1

    # 汇总
    print("\n" + "=" * 60)
    print("  下载完成汇总")
    print("=" * 60)
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    print(f"  模型目录: {MODEL_DIR}")
    print()

    # 列出已下载文件
    print("  已下载文件:")
    for f in sorted(MODEL_DIR.iterdir()):
        if f.is_file() and f.suffix in (".pth", ".pt"):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"    {f.name:40s} {size_mb:>10.1f}MB")
    print()


if __name__ == "__main__":
    main()
