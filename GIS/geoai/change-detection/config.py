"""
ChangeDetection 全局配置
"""
import os as _os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
REPO_ROOT = PROJECT_ROOT.parent
SHARED_MODELS_DIR = Path(_os.getenv("GEOAI_MODELS_DIR", str(REPO_ROOT / "models"))).expanduser()
if not SHARED_MODELS_DIR.is_absolute():
    SHARED_MODELS_DIR = (REPO_ROOT / SHARED_MODELS_DIR).resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TIME_A_DIR = RAW_DIR / "time_a"
TIME_B_DIR = RAW_DIR / "time_b"
SAMPLES_DIR = DATA_DIR / "samples"
MODELS_DIR = SHARED_MODELS_DIR / "change-detection" / "trained"
OUTPUT_DIR = DATA_DIR / "output"

for d in [TIME_A_DIR, TIME_B_DIR, SAMPLES_DIR, MODELS_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── 预训练模型缓存目录（工作区根 models/，供多个子项目共享） ──
PRETRAINED_DIR = SHARED_MODELS_DIR
PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
HF_HOME_DIR = PRETRAINED_DIR / "huggingface"
HF_HUB_CACHE_DIR = HF_HOME_DIR / "hub"
for d in [HF_HOME_DIR, HF_HUB_CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

_os.environ["TORCH_HOME"] = str(PRETRAINED_DIR)
_os.environ["HF_HOME"] = str(HF_HOME_DIR)
_os.environ["HF_HUB_CACHE"] = str(HF_HUB_CACHE_DIR)
_os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_HUB_CACHE_DIR)
_os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

PRESET_REGIONS = {
    "beijing": {"name": "北京", "bbox": [116.20, 39.75, 116.60, 40.05], "epsg": 32650},
    "shanghai": {"name": "上海浦东", "bbox": [121.40, 31.10, 121.70, 31.35], "epsg": 32651},
    "shenzhen": {"name": "深圳", "bbox": [113.85, 22.45, 114.15, 22.65], "epsg": 32650},
    "dubai": {"name": "迪拜", "bbox": [55.10, 25.05, 55.40, 25.25], "epsg": 32640},
    "san_francisco": {"name": "旧金山", "bbox": [-122.52, 37.70, -122.35, 37.83], "epsg": 32610},
}

S2_BANDS = ["B02", "B03", "B04", "B08"]
S2_RESOLUTION = 10

MODEL_CONFIG = {
    "siamese_unet": {"encoder": "resnet34", "in_channels": 3, "decoder_channels": [256, 128, 64, 32, 16]},
    "siamese_unet_light": {"encoder": "mobilenet_v2", "in_channels": 3, "decoder_channels": [128, 64, 32, 16, 8]},
    "bit": {"encoder": "resnet50", "in_channels": 3, "embed_dim": 256, "num_heads": 8},
}

TRAIN_CONFIG = {
    "epochs": 50, "batch_size": 8, "learning_rate": 1e-4, "weight_decay": 1e-4,
    "lr_scheduler": "cosine", "early_stopping_patience": 10, "num_workers": 4,
    "augmentation": True, "val_split": 0.15, "mixed_precision": True,
    "tile_size": 256, "stride": 128,
}

INFERENCE_CONFIG = {
    "tile_size": 256, "overlap": 32, "batch_size": 4, "threshold": 0.5,
    "smoothing_sigma": 1.0, "min_change_area": 30,
}

COMPARE_CONFIG = {"opacity": 0.7, "change_color": "#FF0000", "swipe_position": 0.5}

API_HOST = "0.0.0.0"
API_PORT = int(_os.getenv("CHANGE_DETECTION_BACKEND_PORT", "8005"))
CORS_ORIGINS = ["http://localhost:5178", "http://127.0.0.1:5178"]
