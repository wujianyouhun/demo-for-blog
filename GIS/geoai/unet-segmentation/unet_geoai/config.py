from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GEOAI_ROOT = PROJECT_ROOT.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = OUTPUT_DIR / "reports"
SHARED_MODELS_DIR = Path(os.getenv("GEOAI_MODELS_DIR", GEOAI_ROOT / "models")).expanduser().resolve()
CHECKPOINT_DIR = SHARED_MODELS_DIR / "unet-segmentation" / "checkpoints"
REAL_BUILDING_DIR = GEOAI_ROOT / "makelable" / "data" / "output"

for directory in (DATA_DIR, OUTPUT_DIR, REPORT_DIR, CHECKPOINT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("TORCH_HOME", str(SHARED_MODELS_DIR))
os.environ.setdefault("HF_HOME", str(SHARED_MODELS_DIR / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(SHARED_MODELS_DIR / "huggingface" / "hub"))

CLASS_NAMES = ["background", "building", "road", "water", "vegetation", "barren"]
CLASS_NAMES_ZH = ["其他背景", "建筑", "道路", "水体", "植被", "裸地"]
CLASS_COLORS = {
    0: [65, 72, 82],
    1: [235, 87, 87],
    2: [250, 201, 67],
    3: [57, 130, 214],
    4: [65, 176, 110],
    5: [174, 118, 76],
}
NUM_CLASSES = len(CLASS_NAMES)
IGNORE_INDEX = 255

PROFILES = {
    "quick": {
        "image_size": 128,
        "samples": 160,
        "splits": {"train": 112, "val": 24, "test": 24},
        "epochs": 3,
        "batch_size": 8,
        "base_channels": 16,
        "patience": 3,
    },
    "full": {
        "image_size": 256,
        "samples": 800,
        "splits": {"train": 560, "val": 120, "test": 120},
        "epochs": 25,
        "batch_size": 4,
        "base_channels": 32,
        "patience": 5,
    },
}

DEFAULT_SEED = 42
DEFAULT_LR = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-4
