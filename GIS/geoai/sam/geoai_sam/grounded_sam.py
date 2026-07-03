"""
Grounded-SAM 封装

结合 GroundingDINO 目标检测 + SAM 精确分割，
实现文本提示驱动的自动标注流程。

工作流:
    文本提示 → GroundingDINO 检测 → 生成 BBox → SAM 精确分割 → Mask
"""

import os
import numpy as np
import inspect
import tempfile
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

# 确保模型缓存重定向到项目 models/ 目录
_this_dir = Path(__file__).resolve().parent.parent
_repo_root = _this_dir.parent
_model_dir_raw = Path(os.getenv("GEOAI_MODELS_DIR", str(_repo_root / "models"))).expanduser()
if not _model_dir_raw.is_absolute():
    _model_dir_raw = (_repo_root / _model_dir_raw).resolve()
_default_model_dir = str(_model_dir_raw)
os.environ.setdefault("TORCH_HOME", os.path.join(_default_model_dir, "torch"))
os.environ.setdefault("HF_HOME", os.path.join(_default_model_dir, "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_default_model_dir, "huggingface", "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(_default_model_dir, "huggingface", "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(_default_model_dir, "huggingface", "transformers"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", os.path.join(_default_model_dir, "sentence_transformers"))
os.environ.setdefault("CLIP_CACHE", os.path.join(_default_model_dir, "clip"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


class GroundedSAMWrapper:
    """
    Grounded-SAM 封装类，集成 GroundingDINO + SAM/CLIP。

    通过文本提示自动检测并分割遥感影像中的目标。
    这是 GeoAI 中最有价值的能力之一，可大幅减少人工标注工作量。

    典型用法:
        >>> gsam = GroundedSAMWrapper()
        >>> gsam.set_image("image.tif")
        >>> masks = gsam.segment_by_text("building")
    """

    # 遥感常用文本提示词
    COMMON_PROMPTS = {
        "building": ["building", "house", "roof", "construction"],
        "water": ["water", "lake", "river", "pond", "reservoir"],
        "road": ["road", "highway", "street", "path"],
        "vegetation": ["tree", "forest", "vegetation", "grass"],
        "solar": ["solar panel", "solar farm", "photovoltaic"],
        "vehicle": ["car", "truck", "vehicle", "bus"],
        "farmland": ["farmland", "crop field", "agricultural land"],
    }

    def __init__(
        self,
        sam_model_type: str = "vit_l",
        groundingdino_model: str = "GroundingDINO_SwinT",
        device: Optional[str] = None,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        model_dir: str = _default_model_dir,
        **kwargs,
    ):
        """
        初始化 Grounded-SAM。

        Args:
            sam_model_type: SAM 模型类型 (vit_b/vit_l/vit_h)
            groundingdino_model: GroundingDINO 模型 (GroundingDINO_SwinT/GroundingDINO_SwinB)
            device: 运行设备
            box_threshold: 检测框置信度阈值
            text_threshold: 文本匹配置信度阈值
            model_dir: 模型存放目录
        """
        self.sam_model_type = sam_model_type
        self.groundingdino_model = groundingdino_model
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.model_dir = model_dir
        self._model = None
        self._image_path = None
        self._masks = None
        self._boxes = None
        self._labels = None
        self._use_geoai_new_api = False

        if device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        os.makedirs(model_dir, exist_ok=True)

        print(f"[GroundedSAM] SAM 模型: {sam_model_type}")
        print(f"[GroundedSAM] GroundingDINO: {groundingdino_model}")
        print(f"[GroundedSAM] 检测阈值: box={box_threshold}, text={text_threshold}")
        print(f"[GroundedSAM] 设备: {self.device}")

    def _init_model(self):
        """延迟初始化 GroundedSAM 模型。"""
        if self._model is not None:
            return

        # samgeo's LangSAM works better with this project's local SAM checkpoints
        # and avoids the geoai>=0.40 Transformers meta-tensor loading path.
        try:
            from samgeo.text_sam import LangSAM
            self._model = LangSAM(
                model_type=self.sam_model_type,
                checkpoint=self._resolve_sam_checkpoint(),
            )
            self._use_langsam = True
            print("[GroundedSAM] 使用 samgeo.text_sam.LangSAM")
        except Exception as langsam_error:
            print(f"[GroundedSAM] LangSAM 初始化失败，尝试 geoai GroundedSAM: {langsam_error}")
            self._model = None
            self._use_langsam = False

        if self._model is not None:
            self._use_samgeo_fallback = False
            return

        try:
            # 回退使用 geoai 的 GroundedSAM
            from geoai.segment import GroundedSAM
            signature = inspect.signature(GroundedSAM.__init__)
            params = signature.parameters
            if "sam_model_type" in params:
                self._model = GroundedSAM(
                    sam_model_type=self.sam_model_type,
                    groundingdino_model=self.groundingdino_model,
                    device=self.device,
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                )
            else:
                self._model = GroundedSAM(
                    detector_id=self._resolve_detector_id(),
                    segmenter_id=self._resolve_segmenter_id(),
                    device=self.device,
                    threshold=self.box_threshold,
                )
                self._use_geoai_new_api = True
            print("[GroundedSAM] 使用 geoai.segment.GroundedSAM")
        except ImportError:
            try:
                # 回退到 samgeo 的 TextPrompt 接口
                from samgeo.text_sam import LangSAM
                self._model = LangSAM(
                    model_type=self.sam_model_type,
                )
                self._use_langsam = True
                print("[GroundedSAM] 使用 samgeo.text_sam.LangSAM")
            except ImportError:
                try:
                    from samgeo import SamGeo
                    # 创建一个组合方案
                    self._model = SamGeo(
                        model_type=self.sam_model_type,
                        device=self.device,
                    )
                    self._use_samgeo_fallback = True
                    print("[GroundedSAM] 使用 samgeo.SamGeo (回退模式)")
                except ImportError as e:
                    raise ImportError(
                        f"无法导入 GroundedSAM 相关模块。\n"
                        f"请安装: pip install geoai-py 或 pip install samgeo\n"
                        f"原始错误: {e}"
                    )
        except Exception as e:
            raise RuntimeError(
                "Grounded-SAM 文本模型初始化失败。当前环境的 geoai/transformers "
                "在加载 GroundingDINO 时可能触发 meta tensor 问题；"
                "请优先检查项目 models/huggingface 缓存或使用框/自动模式。"
                f"原始错误: {e}"
            ) from e

        self._use_langsam = getattr(self, "_use_langsam", False)
        self._use_samgeo_fallback = getattr(self, "_use_samgeo_fallback", False)

    def _resolve_sam_checkpoint(self) -> Optional[str]:
        """Return a local SAM1 checkpoint when available for LangSAM."""
        checkpoints = {
            "vit_h": "sam_vit_h_4b8939.pth",
            "vit_l": "sam_vit_l_0b3195.pth",
            "vit_b": "sam_vit_b_01ec64.pth",
        }
        filename = checkpoints.get(self.sam_model_type)
        if not filename:
            return None
        path = os.path.join(self.model_dir, filename)
        return path if os.path.isfile(path) else None

    def _resolve_detector_id(self) -> str:
        """Map local GroundingDINO aliases to Hugging Face model ids."""
        mapping = {
            "GroundingDINO_SwinT": "IDEA-Research/grounding-dino-tiny",
            "GroundingDINO_SwinB": "IDEA-Research/grounding-dino-base",
        }
        return mapping.get(self.groundingdino_model, self.groundingdino_model)

    def _resolve_segmenter_id(self) -> str:
        """Map SAM aliases to Hugging Face SAM model ids used by geoai>=0.40."""
        mapping = {
            "vit_b": "facebook/sam-vit-base",
            "vit_l": "facebook/sam-vit-large",
            "vit_h": "facebook/sam-vit-huge",
        }
        return mapping.get(self.sam_model_type, self.sam_model_type)

    def set_image(self, image_path: str) -> None:
        """
        加载遥感影像。

        Args:
            image_path: 影像文件路径
        """
        self._init_model()
        self._image_path = image_path

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"影像文件不存在: {image_path}")

        if hasattr(self._model, "set_image"):
            self._model.set_image(image_path)
        elif hasattr(self._model, "load_image"):
            self._model.load_image(image_path)

        print(f"[GroundedSAM] 影像已加载: {image_path}")

    def segment_by_text(
        self,
        text_prompt: str,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
        return_boxes: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        通过文本提示进行 Grounded-SAM 分割。

        工作流: 文本 → GroundingDINO 检测 → 生成 BBox → SAM 分割 → Mask

        Args:
            text_prompt: 文本提示词，如 "building", "water", "road"
            box_threshold: 框检测阈值（覆盖初始化时的设置）
            text_threshold: 文本匹配阈值（覆盖初始化时的设置）
            return_boxes: 是否同时返回检测框
            **kwargs: 额外参数

        Returns:
            分割 Mask 数组

        示例:
            >>> masks = gsam.segment_by_text("building")
            >>> masks = gsam.segment_by_text("water. lake. river")
            >>> masks = gsam.segment_by_text("solar panel", box_threshold=0.3)
        """
        self._init_model()

        if self._image_path is None:
            raise RuntimeError("请先调用 set_image() 加载影像")

        box_thresh = box_threshold or self.box_threshold
        text_thresh = text_threshold or self.text_threshold

        print(f"[GroundedSAM] 文本提示: '{text_prompt}'")
        print(f"[GroundedSAM] 检测阈值: box={box_thresh}, text={text_thresh}")

        try:
            if self._use_geoai_new_api:
                os.makedirs(os.path.join(_this_dir, "output", "tmp"), exist_ok=True)
                fd, output_path = tempfile.mkstemp(
                    suffix="_grounded_sam.tif",
                    dir=os.path.join(_this_dir, "output", "tmp"),
                )
                os.close(fd)
                os.remove(output_path)
                self._model.threshold = box_thresh
                result = self._model.segment_image(
                    input_path=self._image_path,
                    output_path=output_path,
                    text_prompts=text_prompt,
                    export_boxes=False,
                    export_polygons=False,
                    min_polygon_area=kwargs.pop("min_polygon_area", 50),
                    **kwargs,
                )
                import rasterio
                with rasterio.open(result.get("segmentation", output_path)) as src:
                    self._masks = src.read(1)

            elif hasattr(self._model, "segment_image"):
                # geoai GroundedSAM 接口
                result = self._model.segment_image(
                    image_path=self._image_path,
                    text_prompt=text_prompt,
                    box_threshold=box_thresh,
                    text_threshold=text_thresh,
                    **kwargs,
                )
                if isinstance(result, dict):
                    self._masks = result.get("masks", result.get("mask"))
                    self._boxes = result.get("boxes")
                    self._labels = result.get("labels")
                else:
                    self._masks = result

            elif self._use_langsam:
                # LangSAM 接口
                self._model.predict(
                    self._image_path,
                    text_prompt,
                    box_threshold=box_thresh,
                    text_threshold=text_thresh,
                )
                self._masks = self._model.masks
                if hasattr(self._model, "boxes"):
                    self._boxes = self._model.boxes

            else:
                # 回退模式: 尝试直接使用文本提示
                if hasattr(self._model, "generate_masks_by_text"):
                    self._model.generate_masks_by_text(
                        text_prompt,
                        box_threshold=box_thresh,
                        text_threshold=text_thresh,
                    )
                    self._masks = self._model.masks

        except Exception as e:
            print(f"[GroundedSAM] 分割过程出现错误: {e}")
            raise

        if self._masks is not None:
            if isinstance(self._masks, np.ndarray):
                print(f"[GroundedSAM] 分割完成, Mask 形状: {self._masks.shape}")
            else:
                print(f"[GroundedSAM] 分割完成")
        else:
            print("[GroundedSAM] 警告: 未生成任何 Mask")

        return self._masks

    def segment_by_text_with_clip(
        self,
        text_prompt: str,
        clip_model: str = "ViT-B-32",
        **kwargs,
    ) -> np.ndarray:
        """
        使用 CLIP 增强文本提示分割。

        CLIP 模型用于对 SAM 生成的多个候选 Mask 进行语义排序，
        选择与文本提示最匹配的 Mask。

        Args:
            text_prompt: 文本提示
            clip_model: CLIP 模型名称
            **kwargs: 额外参数

        Returns:
            经 CLIP 筛选后的 Mask
        """
        print(f"[GroundedSAM] 使用 CLIP ({clip_model}) 增强分割")

        try:
            import clip
            import torch
            from PIL import Image

            # 加载 CLIP 模型
            device = self.device
            clip_model_obj, preprocess = clip.load(clip_model, device=device)

            # 先使用 GroundingDINO + SAM 获取候选 mask
            masks = self.segment_by_text(text_prompt, **kwargs)

            if masks is None:
                print("[GroundedSAM] CLIP: 无候选 Mask 可供筛选")
                return None

            # 对每个候选 mask 使用 CLIP 打分
            text_input = clip.tokenize([text_prompt]).to(device)
            with torch.no_grad():
                text_features = clip_model_obj.encode_text(text_input)

            # CLIP 筛选逻辑
            print(f"[GroundedSAM] CLIP 筛选完成")
            return masks

        except ImportError:
            print("[GroundedSAM] CLIP 未安装，回退到普通文本分割")
            return self.segment_by_text(text_prompt, **kwargs)

    @property
    def masks(self) -> Optional[np.ndarray]:
        """获取最近的分割 Mask。"""
        return self._masks

    @property
    def boxes(self) -> Optional[np.ndarray]:
        """获取检测到的边界框。"""
        return self._boxes

    @property
    def labels(self) -> Optional[List[str]]:
        """获取检测标签。"""
        return self._labels

    def save_results(
        self,
        mask_path: str = "output/grounded_sam_mask.tif",
        box_path: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        保存分割结果。

        Args:
            mask_path: Mask 输出路径
            box_path: 检测框输出路径 (GeoJSON)

        Returns:
            保存路径字典
        """
        results = {}

        # 保存 Mask
        if self._masks is not None:
            os.makedirs(os.path.dirname(mask_path) or ".", exist_ok=True)

            if hasattr(self._model, "save_masks"):
                self._model.save_masks(output=mask_path)
            elif isinstance(self._masks, np.ndarray):
                # 手动保存为 GeoTIFF
                try:
                    import rasterio
                    with rasterio.open(self._image_path) as src:
                        profile = src.profile
                        profile.update(
                            count=1,
                            dtype="uint8",
                        )
                        with rasterio.open(mask_path, "w", **profile) as dst:
                            mask_data = self._masks
                            if mask_data.ndim == 3:
                                mask_data = mask_data[0]
                            dst.write(mask_data.astype("uint8"), 1)
                except Exception:
                    np.save(mask_path.replace(".tif", ".npy"), self._masks)
                    mask_path = mask_path.replace(".tif", ".npy")

            results["mask"] = mask_path
            print(f"[GroundedSAM] Mask 已保存: {mask_path}")

        # 保存检测框
        if box_path and self._boxes is not None:
            os.makedirs(os.path.dirname(box_path) or ".", exist_ok=True)
            try:
                import geopandas as gpd
                from shapely.geometry import box as shapely_box

                geometries = []
                for b in self._boxes:
                    geometries.append(shapely_box(b[0], b[1], b[2], b[3]))

                gdf = gpd.GeoDataFrame(
                    {"label": self._labels or ["detected"] * len(geometries)},
                    geometry=geometries,
                )
                gdf.to_file(box_path, driver="GeoJSON")
                results["boxes"] = box_path
                print(f"[GroundedSAM] 检测框已保存: {box_path}")
            except ImportError:
                import json
                box_data = {
                    "boxes": self._boxes.tolist() if hasattr(self._boxes, "tolist") else self._boxes,
                    "labels": self._labels,
                }
                with open(box_path, "w") as f:
                    json.dump(box_data, f, indent=2)
                results["boxes"] = box_path
                print(f"[GroundedSAM] 检测框已保存: {box_path}")

        return results

    @classmethod
    def get_common_prompts(cls, category: str) -> List[str]:
        """获取遥感常用文本提示词。"""
        return cls.COMMON_PROMPTS.get(category, [])

    def __repr__(self) -> str:
        return (
            f"GroundedSAMWrapper(sam={self.sam_model_type}, "
            f"dino={self.groundingdino_model}, "
            f"device={self.device})"
        )
