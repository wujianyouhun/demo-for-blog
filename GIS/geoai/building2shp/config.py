"""
建筑物提取项目 - 全局配置
"""
import os
from pathlib import Path

# ── 项目路径 ──
PROJECT_ROOT = Path(__file__).parent
REPO_ROOT = PROJECT_ROOT.parent
SHARED_MODELS_DIR = Path(os.getenv("GEOAI_MODELS_DIR", str(REPO_ROOT / "models"))).expanduser()
if not SHARED_MODELS_DIR.is_absolute():
    SHARED_MODELS_DIR = (REPO_ROOT / SHARED_MODELS_DIR).resolve()
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
MODELS_DIR = SHARED_MODELS_DIR / "building2shp" / "trained"
OUTPUT_DIR = DATA_DIR / "output"

# 预训练模型缓存到工作区根 models/，供多个子项目共享。
PRETRAINED_DIR = SHARED_MODELS_DIR
for d in [INPUT_DIR, MODELS_DIR, OUTPUT_DIR, PRETRAINED_DIR]:
    d.mkdir(parents=True, exist_ok=True)
os.environ["TORCH_HOME"] = str(PRETRAINED_DIR)
os.environ["HF_HOME"] = str(PRETRAINED_DIR / "huggingface")
os.environ["HF_HUB_CACHE"] = str(PRETRAINED_DIR / "huggingface" / "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(PRETRAINED_DIR / "huggingface" / "hub")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ── 类别定义 ──
CLASS_NAMES = ["background", "building", "road", "water", "vegetation", "barren"]
CLASS_COLORS = {
    0: [0, 0, 0],
    1: [255, 0, 0],
    2: [255, 255, 0],
    3: [0, 0, 255],
    4: [0, 255, 0],
    5: [139, 69, 19],
}
NUM_CLASSES = len(CLASS_NAMES)
BUILDING_CLASS_ID = 1

# ── 模型配置 ──
MODEL_CONFIG = {
    "deeplabv3p_resnet50":  {"backbone": "resnet50",  "in_channels": 3},
    "deeplabv3p_resnet101": {"backbone": "resnet101", "in_channels": 3},
    "deeplabv3p_mobilenet": {"backbone": "mobilenet_v2", "in_channels": 3},
}
DEFAULT_MODEL = "deeplabv3p_resnet50"

# ── 推理参数 ──
INFERENCE_CONFIG = {
    "tile_size": 256,
    "overlap": 32,
    "batch_size": 4,
    "threshold": 0.5,
    "smoothing_sigma": 1.0,
    "min_area_px": 30,       # 矢量化时最小面积（像素）
}

# ── 正则化参数 ──
REGULARIZE_CONFIG = {
    "simplify_tolerance": 2.0,
    "smooth_iterations": 3,
    "min_area": 50.0,        # 坐标单位（如平方米）
    "orthogonalize": True,
}
