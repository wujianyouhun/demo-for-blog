"""
建筑物提取项目 - 全局配置
"""
import os
from pathlib import Path

# ── 项目路径 ──
# 备用 DeepLabV3+ 流程使用的目录配置。主流程 extract_buildings.py 使用 models/
# 和 outputs/；这里保留 data/models 与 data/output，是为了兼容原始语义分割脚本。
PROJECT_ROOT = Path(__file__).parent
REPO_ROOT = PROJECT_ROOT.parent
SHARED_MODELS_DIR = Path(os.getenv("GEOAI_MODELS_DIR", str(REPO_ROOT / "models"))).expanduser()
if not SHARED_MODELS_DIR.is_absolute():
    SHARED_MODELS_DIR = (REPO_ROOT / SHARED_MODELS_DIR).resolve()
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR
MODELS_DIR = SHARED_MODELS_DIR / "building-extraction"
OUTPUT_DIR = DATA_DIR / "output"

# 预训练模型缓存到工作区根目录 models/，避免各子项目重复下载。
PRETRAINED_DIR = SHARED_MODELS_DIR / "torch"
HF_HOME_DIR = SHARED_MODELS_DIR / "huggingface"
HF_HUB_CACHE_DIR = HF_HOME_DIR / "hub"
for d in [INPUT_DIR, MODELS_DIR, OUTPUT_DIR, PRETRAINED_DIR, HF_HOME_DIR, HF_HUB_CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)
os.environ["TORCH_HOME"] = str(PRETRAINED_DIR)
os.environ["HF_HOME"] = str(HF_HOME_DIR)
os.environ["HF_HUB_CACHE"] = str(HF_HUB_CACHE_DIR)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_HUB_CACHE_DIR)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ── 类别定义 ──
# DeepLabV3+ 输出的是多类别语义分割图，后续只提取 BUILDING_CLASS_ID 对应类别。
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
# 这里的模型需要与训练权重匹配；如果没有训练权重，只使用 ImageNet 预训练编码器，
# 建筑物分割效果会比较有限。
MODEL_CONFIG = {
    "deeplabv3p_resnet50":  {"backbone": "resnet50",  "in_channels": 3},
    "deeplabv3p_resnet101": {"backbone": "resnet101", "in_channels": 3},
    "deeplabv3p_mobilenet": {"backbone": "mobilenet_v2", "in_channels": 3},
}
DEFAULT_MODEL = "deeplabv3p_resnet50"

# ── 推理参数 ──
# tile_size/overlap 控制滑动窗口推理；重叠区域会做概率平均，减轻窗口边界伪影。
INFERENCE_CONFIG = {
    "tile_size": 256,
    "overlap": 32,
    "batch_size": 4,
    "threshold": 0.5,
    "smoothing_sigma": 1.0,
    "min_area_px": 30,       # 矢量化时最小面积（像素）
}

# ── 正则化参数 ──
# 正则化用于让建筑物边界更规整，适合输出给 GIS 软件继续制图或分析。
REGULARIZE_CONFIG = {
    "simplify_tolerance": 2.0,
    "smooth_iterations": 3,
    "min_area": 50.0,        # 坐标单位（如平方米）
    "orthogonalize": True,
}
