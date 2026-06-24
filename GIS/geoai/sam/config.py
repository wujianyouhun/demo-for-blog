"""
GeoAI SAM 项目配置文件

通过 .env 文件或环境变量加载配置，
也支持在代码中直接修改默认值。
"""

import os
from pathlib import Path
from typing import Optional


# 项目根目录
PROJECT_ROOT = Path(__file__).parent


def load_dotenv():
    """尝试加载 .env 文件。"""
    try:
        from dotenv import load_dotenv
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            return True
    except ImportError:
        pass
    return False


# 尝试加载 .env
load_dotenv()


# ================================================================
# 模型目录与缓存重定向
# ================================================================
# 将所有模型缓存重定向到项目 models/ 目录，
# 避免下载到 C:\Users\...\*.cache 等系统目录。
MODEL_DIR: str = os.getenv("MODEL_DIR", str(PROJECT_ROOT / "models"))
os.makedirs(MODEL_DIR, exist_ok=True)

# PyTorch Hub (SAM 模型下载)
os.environ.setdefault("TORCH_HOME", os.path.join(MODEL_DIR, "torch"))

# HuggingFace (GroundingDINO, CLIP, transformers)
os.environ.setdefault("HF_HOME", os.path.join(MODEL_DIR, "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(MODEL_DIR, "huggingface", "hub"))

# SentenceTransformers
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", os.path.join(MODEL_DIR, "sentence_transformers"))

# CLIP
os.environ.setdefault("CLIP_CACHE", os.path.join(MODEL_DIR, "clip"))

# samgeo
os.environ.setdefault("SAMGEO_CACHE", os.path.join(MODEL_DIR, "samgeo"))


# ================================================================
# SAM 模型配置
# ================================================================
# 默认使用 vit_l: 需要 4~8GB 显存，适合 6GB 显卡
SAM_VERSION: str = os.getenv("SAM_VERSION", "sam1")
SAM_MODEL_TYPE: str = os.getenv("SAM_MODEL_TYPE", "vit_l")
SAM_CHECKPOINT_PATH: Optional[str] = os.getenv("SAM_CHECKPOINT_PATH") or None


# ================================================================
# GroundingDINO 配置
# ================================================================
GROUNDINGDINO_MODEL: str = os.getenv("GROUNDINGDINO_MODEL", "GroundingDINO_SwinB")
BOX_THRESHOLD: float = float(os.getenv("BOX_THRESHOLD", "0.25"))
TEXT_THRESHOLD: float = float(os.getenv("TEXT_THRESHOLD", "0.25"))


# ================================================================
# CLIP 配置
# ================================================================
CLIP_MODEL: str = os.getenv("CLIP_MODEL", "ViT-B-32")


# ================================================================
# 运行设备
# ================================================================
DEVICE: Optional[str] = os.getenv("DEVICE") or None


# ================================================================
# 输出配置
# ================================================================
OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", str(PROJECT_ROOT / "output"))
OUTPUT_FORMAT: str = os.getenv("OUTPUT_FORMAT", "geojson")


# ================================================================
# 后处理默认参数
# ================================================================
POSTPROCESS_MIN_SIZE: int = int(os.getenv("POSTPROCESS_MIN_SIZE", "200"))
POSTPROCESS_FILL_HOLES: bool = os.getenv("POSTPROCESS_FILL_HOLES", "true").lower() == "true"
POSTPROCESS_SMOOTH_SIGMA: float = float(os.getenv("POSTPROCESS_SMOOTH_SIGMA", "1.5"))
POSTPROCESS_OPENING_RADIUS: int = int(os.getenv("POSTPROCESS_OPENING_RADIUS", "2"))
POSTPROCESS_CLOSING_RADIUS: int = int(os.getenv("POSTPROCESS_CLOSING_RADIUS", "3"))


# ================================================================
# 数据路径
# ================================================================
DEFAULT_IMAGE: str = os.getenv("DEFAULT_IMAGE", r"E:\data\baoji\宝鸡市\I48E006018\I48E006018.tif")


# ================================================================
# 模型信息
# ================================================================
MODEL_INFO = {
    "vit_b": {
        "vram": "2~4GB",
        "params": "91M",
        "url_sam1": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "url_sam2": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
    },
    "vit_l": {
        "vram": "4~8GB",
        "params": "308M",
        "url_sam1": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "url_sam2": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
    },
    "vit_h": {
        "vram": "8GB+",
        "params": "636M",
        "url_sam1": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        # SAM2.1 没有 huge 变体，vit_h 复用 large
        "url_sam2": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
    },
}


def print_config():
    """打印当前配置。"""
    print("=" * 55)
    print("  GeoAI SAM 项目配置")
    print("=" * 55)
    print(f"  SAM 版本:      {SAM_VERSION}")
    print(f"  模型类型:       {SAM_MODEL_TYPE}")
    print(f"  检查点:         {SAM_CHECKPOINT_PATH or '自动下载到 models/'}")
    print(f"  模型目录:       {MODEL_DIR}")
    print(f"  GroundingDINO:  {GROUNDINGDINO_MODEL}")
    print(f"  CLIP:           {CLIP_MODEL}")
    print(f"  设备:           {DEVICE or '自动检测'}")
    print(f"  输出目录:       {OUTPUT_DIR}")
    print(f"  检测阈值:       box={BOX_THRESHOLD}, text={TEXT_THRESHOLD}")
    print(f"  后处理:         min_size={POSTPROCESS_MIN_SIZE}")
    print(f"  默认影像:       {DEFAULT_IMAGE}")
    print("-" * 55)
    print("  模型缓存目录 (已重定向到项目内):")
    print(f"    TORCH_HOME:       {os.environ.get('TORCH_HOME', '未设置')}")
    print(f"    HF_HOME:          {os.environ.get('HF_HOME', '未设置')}")
    print(f"    TRANSFORMERS:     {os.environ.get('TRANSFORMERS_CACHE', '未设置')}")
    print(f"    CLIP_CACHE:       {os.environ.get('CLIP_CACHE', '未设置')}")
    print("=" * 55)


if __name__ == "__main__":
    print_config()
