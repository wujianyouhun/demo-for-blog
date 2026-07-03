"""GeoAI Demo - 全局配置"""
import os as _os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
REPO_ROOT = PROJECT_ROOT.parent
SHARED_MODELS_DIR = Path(_os.getenv("GEOAI_MODELS_DIR", str(REPO_ROOT / "models"))).expanduser()
if not SHARED_MODELS_DIR.is_absolute():
    SHARED_MODELS_DIR = (REPO_ROOT / SHARED_MODELS_DIR).resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SAMPLES_DIR = DATA_DIR / "samples"
LABEL_DIR = DATA_DIR / "labels"
MODELS_DIR = SHARED_MODELS_DIR / "geoai-demo" / "trained"
OUTPUT_DIR = DATA_DIR / "output"

for d in [RAW_DIR, SAMPLES_DIR, LABEL_DIR, MODELS_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── 预训练模型缓存目录（工作区根 models/，供多个子项目共享） ──
PRETRAINED_DIR = SHARED_MODELS_DIR
PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)

_os.environ["TORCH_HOME"] = str(PRETRAINED_DIR)
_os.environ["HF_HOME"] = str(PRETRAINED_DIR / "huggingface")
_os.environ["HF_HUB_CACHE"] = str(PRETRAINED_DIR / "huggingface" / "hub")
_os.environ["HUGGINGFACE_HUB_CACHE"] = str(PRETRAINED_DIR / "huggingface" / "hub")
_os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 类别定义
CLASS_NAMES = ["background", "building", "road", "water", "vegetation", "barren"]
CLASS_COLORS = {
    0: [0, 0, 0],        # background - 黑色
    1: [255, 0, 0],      # building - 红色
    2: [255, 255, 0],    # road - 黄色
    3: [0, 0, 255],      # water - 蓝色
    4: [0, 255, 0],      # vegetation - 绿色
    5: [139, 69, 19],    # barren - 棕色
}
NUM_CLASSES = len(CLASS_NAMES)

# STAC 数据源
STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
PRESET_REGIONS = {
    "beijing": {"name": "北京", "bbox": [116.20, 39.75, 116.60, 40.05]},
    "shanghai": {"name": "上海", "bbox": [121.40, 31.10, 121.70, 31.35]},
    "shenzhen": {"name": "深圳", "bbox": [113.85, 22.45, 114.15, 22.65]},
}

MODEL_CONFIG = {
    "deeplabv3p_resnet50": {"backbone": "resnet50", "in_channels": 3},
    "deeplabv3p_resnet101": {"backbone": "resnet101", "in_channels": 3},
    "deeplabv3p_mobilenet": {"backbone": "mobilenet_v2", "in_channels": 3},
}

TRAIN_CONFIG = {
    "epochs": 50, "batch_size": 8, "learning_rate": 1e-4,
    "weight_decay": 1e-4, "lr_scheduler": "cosine",
    "early_stopping_patience": 10, "num_workers": 4,
    "augmentation": True, "val_split": 0.15,
    "mixed_precision": True, "tile_size": 256, "stride": 128,
}

INFERENCE_CONFIG = {
    "tile_size": 256, "overlap": 32, "batch_size": 4,
    "threshold": 0.5, "smoothing_sigma": 1.0, "min_area": 20,
}

REGULARIZE_CONFIG = {
    "simplify_tolerance": 2.0, "smooth_iterations": 3,
    "min_area": 50, "orthogonalize": True,
}

API_HOST = "0.0.0.0"
API_PORT = 8000
CORS_ORIGINS = ["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"]
