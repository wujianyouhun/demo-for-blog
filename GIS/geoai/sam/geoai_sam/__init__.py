"""
GeoAI SAM - 基于 SAM 的遥感半自动标注工具包

基于 Segment Anything Model (SAM) 系列模型，
支持点提示、框提示、文本提示等交互式标注方式，
用于快速生成 GeoAI 训练标签。

集成模型: SAM / SAM2 / SAM3 / GroundingDINO / CLIP / PyTorch
默认模型: vit_l (适配 4~8GB 显存，推荐 6GB)
模型缓存: 统一下载到项目 models/ 目录，不使用系统 .cache
"""

import os
from pathlib import Path

# 在所有模块导入前，确保模型缓存重定向到项目 models/ 目录
_pkg_dir = Path(__file__).resolve().parent.parent
_default_model_dir = str(_pkg_dir / "models")
os.makedirs(_default_model_dir, exist_ok=True)
os.environ.setdefault("TORCH_HOME", os.path.join(_default_model_dir, "torch"))
os.environ.setdefault("HF_HOME", os.path.join(_default_model_dir, "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_default_model_dir, "huggingface", "hub"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", os.path.join(_default_model_dir, "sentence_transformers"))
os.environ.setdefault("CLIP_CACHE", os.path.join(_default_model_dir, "clip"))

# matplotlib 中文字体配置（解决 Windows 中文乱码）
# 在所有模块导入前配置，确保后续所有绘图自动生效
def _setup_chinese_font():
    try:
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        preferred = [
            "Microsoft YaHei", "SimHei", "DengXian",
            "KaiTi", "FangSong", "Source Han Serif SC",
        ]
        available = {f.name for f in fm.fontManager.ttflist}
        for name in preferred:
            if name in available:
                plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                break
        else:
            cjk = [f.name for f in fm.fontManager.ttflist
                   if any(k in f.name.lower()
                          for k in ("cjk", "hei", "song", "ming", "gothic"))]
            if cjk:
                plt.rcParams["font.sans-serif"] = [cjk[0], "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except ImportError:
        pass

_setup_chinese_font()

from .core import SAMWrapper
from .grounded_sam import GroundedSAMWrapper
from .postprocess import MaskPostProcessor
from .vectorize import MaskVectorizer
from .quality import QualityMetrics

__version__ = "1.0.0"
__all__ = [
    "SAMWrapper",
    "GroundedSAMWrapper",
    "MaskPostProcessor",
    "MaskVectorizer",
    "QualityMetrics",
]
