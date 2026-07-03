"""
GeoAI 图像分类 — 推理引擎
==========================
单例模式加载模型，支持单张/批量图像推理
"""
from __future__ import annotations
import io, sys, time
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT.parent
SHARED_MODELS_DIR = Path(os.getenv("GEOAI_MODELS_DIR", str(REPO_ROOT / "models"))).expanduser()
if not SHARED_MODELS_DIR.is_absolute():
    SHARED_MODELS_DIR = (REPO_ROOT / SHARED_MODELS_DIR).resolve()
sys.path.insert(0, str(ROOT / "scripts"))

# EuroSAT 类别
CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake",
]
CLASS_NAMES_ZH = [
    "农田", "森林", "草本植被", "公路",
    "工业区", "牧场", "永久作物", "居民区",
    "河流", "海湖",
]
CLASS_DESCRIPTIONS = {
    "AnnualCrop":            "一年生农作物耕地，如小麦、玉米等",
    "Forest":                "自然或人工林地，树木覆盖密集",
    "HerbaceousVegetation":  "草本植被，无树木覆盖的低矮植物",
    "Highway":               "高速公路或主干道，线状交通设施",
    "Industrial":            "工业园区或工厂，建筑密集",
    "Pasture":               "放牧草地，用于畜牧业",
    "PermanentCrop":         "多年生作物，如葡萄园、果园",
    "Residential":           "城市居民区，住宅密集",
    "River":                 "河流水道，细长水体",
    "SeaLake":               "海洋或大型湖泊，大面积水体",
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


class GeoAIPredictor:
    """GeoAI 图像分类推理引擎（单例）"""

    _instance: Optional["GeoAIPredictor"] = None

    def __init__(self, checkpoint_path: str | Path, device: str = "auto"):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(
            "cuda" if (device == "auto" and torch.cuda.is_available())
            else ("cpu" if device == "auto" else device)
        )
        self.model      = None
        self.class_names: list[str] = CLASS_NAMES
        self.image_size: int        = 224
        self.model_name: str        = "unknown"
        self._load_time: float      = 0.0
        self._loaded    = False

    # ── 加载 ──────────────────────────────────────────────────────────
    def load(self):
        if self._loaded:
            return
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"模型文件不存在: {self.checkpoint_path}\n"
                "请先训练模型: python scripts/train.py"
            )
        t0   = time.time()
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)

        self.model_name  = ckpt.get("model_name",  "resnet50")
        self.image_size  = ckpt.get("image_size",  224)
        self.class_names = ckpt.get("class_names", CLASS_NAMES)

        from train import build_model
        self.model = build_model(self.model_name, len(self.class_names))
        self.model.load_state_dict(ckpt["model"])
        self.model = self.model.to(self.device).eval()

        self._load_time = time.time() - t0
        self._loaded    = True
        print(f"✓ 模型已加载: {self.model_name}  "
              f"设备: {self.device}  耗时: {self._load_time:.2f}s")

    # ── 预处理 ─────────────────────────────────────────────────────────
    def _preprocess(self, pil_image: Image.Image) -> torch.Tensor:
        tfm = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        return tfm(pil_image.convert("RGB")).unsqueeze(0).to(self.device)

    # ── 单张推理 ───────────────────────────────────────────────────────
    def predict(self, image_bytes: bytes) -> dict:
        """
        Args:
            image_bytes: 图像二进制数据
        Returns:
            {
              "class_name": str,
              "class_name_zh": str,
              "class_id": int,
              "confidence": float,
              "description": str,
              "top5": [{"class": str, "class_zh": str, "prob": float}, ...],
              "infer_time_ms": float
            }
        """
        if not self._loaded:
            self.load()

        img = Image.open(io.BytesIO(image_bytes))
        tensor = self._preprocess(img)

        t0 = time.time()
        with torch.no_grad():
            logits = self.model(tensor)
            probs  = F.softmax(logits, dim=1)[0]
        infer_ms = (time.time() - t0) * 1000

        top5_idx  = probs.topk(min(5, len(self.class_names))).indices.tolist()
        top5      = [
            {
                "class":    self.class_names[i],
                "class_zh": CLASS_NAMES_ZH[i] if i < len(CLASS_NAMES_ZH) else "",
                "prob":     round(float(probs[i]), 6),
            }
            for i in top5_idx
        ]

        pred_idx  = top5_idx[0]
        pred_name = self.class_names[pred_idx]
        return {
            "class_name":    pred_name,
            "class_name_zh": CLASS_NAMES_ZH[pred_idx] if pred_idx < len(CLASS_NAMES_ZH) else "",
            "class_id":      pred_idx,
            "confidence":    round(float(probs[pred_idx]), 6),
            "description":   CLASS_DESCRIPTIONS.get(pred_name, ""),
            "top5":          top5,
            "infer_time_ms": round(infer_ms, 2),
        }

    # ── 健康状态 ───────────────────────────────────────────────────────
    def health(self) -> dict:
        return {
            "loaded":      self._loaded,
            "model_name":  self.model_name,
            "device":      str(self.device),
            "num_classes": len(self.class_names),
            "image_size":  self.image_size,
            "load_time_s": round(self._load_time, 2),
        }


# ── 全局单例 ────────────────────────────────────────────────────────────
_predictor: Optional[GeoAIPredictor] = None

def get_predictor(checkpoint_path: str | Path | None = None,
                  device: str = "auto") -> GeoAIPredictor:
    global _predictor
    if _predictor is None:
        if checkpoint_path is None:
            checkpoint_path = SHARED_MODELS_DIR / "Classification" / "checkpoints" / "best_model.pth"
        _predictor = GeoAIPredictor(checkpoint_path, device)
    return _predictor
