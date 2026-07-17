"""U-Net GeoAI 多类语义分割教学与工程包。"""

from .config import CLASS_COLORS, CLASS_NAMES, NUM_CLASSES, PROFILES
from .models import MODEL_NAMES, build_model

__all__ = ["CLASS_COLORS", "CLASS_NAMES", "NUM_CLASSES", "PROFILES", "MODEL_NAMES", "build_model"]
