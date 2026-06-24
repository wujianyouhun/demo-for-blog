"""Download pretrained models used by the change-detection demo."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRETRAINED_DIR = PROJECT_ROOT / "data" / "pretrained"
HF_HOME_DIR = PRETRAINED_DIR / "huggingface"
HF_HUB_CACHE_DIR = HF_HOME_DIR / "hub"
MODEL_DIR = PRETRAINED_DIR / "models"

CHANGESTAR_REPO = "EVER-Z/Changen2-ChangeStar1x256"
CHANGESTAR_KEYS = {
    "s0_s1c1_vitb": "s0_changestar_vitb_1x256",
    "s0_s1c5_vitb": "s0_changestar_vitb_1x256",
    "s0_s9c1_vitb": "s0_changestar_vitb_1x256",
    "s0_xview2_s1c5_vitb": "s0_xView2_changestar_vitb_1x256",
    "s1_s1c1_vitb": "s1_changestar_vitb_1x256",
    "s1_s1c1_vitl": "s1_changestar_vitl_1x256",
    "s9_s9c1_vitb": "s9_changestar_vitb_1x256",
}


def retry_call(label: str, fn, retries: int = 20, delay: int = 5):
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:  # network downloads can fail mid-stream
            last_error = exc
            print(f"{label} failed ({attempt}/{retries}): {type(exc).__name__}: {exc}", flush=True)
            if attempt < retries:
                time.sleep(delay)
    raise RuntimeError(f"{label} failed after {retries} attempts") from last_error


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == source.stat().st_size:
        return
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def range_download(url: str, destination: Path, expected_size: int | None = None) -> Path:
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)

    if expected_size is None:
        with requests.get(
            url,
            headers={"Range": "bytes=0-0"},
            stream=True,
            timeout=(20, 60),
            allow_redirects=True,
        ) as response:
            response.raise_for_status()
            content_range = response.headers.get("Content-Range", "")
            if "/" in content_range:
                expected_size = int(content_range.rsplit("/", 1)[1])
            else:
                expected_size = int(response.headers.get("Content-Length", "0"))

    attempts = 0
    last_printed_mb = -1
    while True:
        current_size = destination.stat().st_size if destination.exists() else 0
        if expected_size and current_size >= expected_size:
            print(f"Range download complete: {destination} ({current_size / 1024 / 1024:.1f} MB)", flush=True)
            return destination

        attempts += 1
        headers = {"Range": f"bytes={current_size}-"} if current_size else {}
        try:
            with requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=(20, 120),
                allow_redirects=True,
            ) as response:
                if response.status_code not in (200, 206):
                    response.raise_for_status()
                if response.status_code == 200 and current_size:
                    raise RuntimeError("Server ignored Range header during resume")

                with destination.open("ab") as fp:
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        if not chunk:
                            continue
                        fp.write(chunk)
                        current_size += len(chunk)
                        mb = int(current_size / 1024 / 1024)
                        if mb >= last_printed_mb + 20:
                            last_printed_mb = mb
                            total = f"/{expected_size / 1024 / 1024:.1f} MB" if expected_size else ""
                            print(f"  downloaded {current_size / 1024 / 1024:.1f} MB{total}", flush=True)
        except Exception as exc:
            current_size = destination.stat().st_size if destination.exists() else 0
            print(
                f"Range download retry {attempts}: {type(exc).__name__}: {exc}; "
                f"saved {current_size / 1024 / 1024:.1f} MB",
                flush=True,
            )
            time.sleep(min(30, 2 + attempts // 5))


def populate_hf_cache(filename: str, source: Path, etag: str | None = None) -> None:
    repo_cache = HF_HUB_CACHE_DIR / "models--EVER-Z--Changen2-ChangeStar1x256"
    refs_main = repo_cache / "refs" / "main"
    commit = refs_main.read_text(encoding="utf-8").strip() if refs_main.exists() else "a51540d0abaca8f2b96ccd97139ccd4cfe55bd67"

    snapshot_file = repo_cache / "snapshots" / commit / filename
    link_or_copy(source, snapshot_file)

    if etag:
        blob_file = repo_cache / "blobs" / etag.strip('"')
        link_or_copy(source, blob_file)


def configure_project_cache() -> None:
    for directory in [PRETRAINED_DIR, HF_HOME_DIR, HF_HUB_CACHE_DIR, MODEL_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(PRETRAINED_DIR)
    os.environ["HF_HOME"] = str(HF_HOME_DIR)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_HUB_CACHE_DIR)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def download_changestar(model_names: list[str]) -> list[Path]:
    from huggingface_hub import hf_hub_download

    config_path = retry_call(
        "Download ChangeStar config",
        lambda: hf_hub_download(
            CHANGESTAR_REPO,
            "config.json",
            cache_dir=str(HF_HUB_CACHE_DIR),
        ),
    )
    with open(config_path, "r", encoding="utf-8") as fp:
        available = json.load(fp)

    downloaded: list[Path] = []
    for model_name in model_names:
        key = CHANGESTAR_KEYS[model_name]
        if key not in available:
            raise KeyError(f"{key} not found in {CHANGESTAR_REPO}/config.json")

        filename = available[key]
        url = f"https://huggingface.co/{CHANGESTAR_REPO}/resolve/main/{filename}"
        target = MODEL_DIR / f"{model_name}__{Path(filename).name}"
        cache_path = range_download(
            url,
            target,
            expected_size=399863295 if filename == "s1c1_cstar_vitb_1x256.pth" else None,
        )
        populate_hf_cache(
            filename,
            cache_path,
            etag="a435f255fa60748fff6ab19050fbe67dd87d9aa24c74ff089c0ec51cdda138a9"
            if filename == "s1c1_cstar_vitb_1x256.pth"
            else None,
        )
        downloaded.append(target)
        print(f"GeoAI ChangeStar {model_name}: {target} ({target.stat().st_size / 1024 / 1024:.1f} MB)")
    return downloaded


def download_torchvision_backbones() -> list[Path]:
    import torch
    from torchvision.models import MobileNet_V2_Weights, ResNet34_Weights, ResNet50_Weights
    from torchvision.models import mobilenet_v2, resnet34, resnet50

    checkpoints = PRETRAINED_DIR / "hub" / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)

    resnet34(weights=ResNet34_Weights.DEFAULT)
    resnet50(weights=ResNet50_Weights.DEFAULT)
    mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

    paths = sorted(checkpoints.glob("*.pth"))
    for path in paths:
        print(f"Torchvision backbone: {path} ({path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="下载项目所需的预训练模型到 data/pretrained")
    parser.add_argument(
        "--changestar-model",
        action="append",
        default=None,
        choices=sorted(CHANGESTAR_KEYS),
        help="GeoAI ChangeStar 模型名；可重复传入。默认只下载 s1_s1c1_vitb。",
    )
    parser.add_argument(
        "--all-changestar",
        action="store_true",
        help="下载 GeoAI 0.40.0 列出的全部 ChangeStar 权重，体积较大。",
    )
    parser.add_argument(
        "--skip-torchvision",
        action="store_true",
        help="不下载自训练路径使用的 torchvision 编码器权重。",
    )
    args = parser.parse_args()

    configure_project_cache()
    changestar_models = sorted(CHANGESTAR_KEYS) if args.all_changestar else (args.changestar_model or ["s1_s1c1_vitb"])

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Pretrained dir: {PRETRAINED_DIR}")
    download_changestar(changestar_models)
    if not args.skip_torchvision:
        download_torchvision_backbones()


if __name__ == "__main__":
    main()
