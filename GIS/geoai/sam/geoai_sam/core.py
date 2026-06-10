"""
SAM 模型核心封装

支持 SAM (vit_b/vit_l/vit_h)、SAM2、SAM3 模型，
提供点提示、框提示、自动分割等核心功能。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Union, Dict, Any
from pathlib import Path

# 确保模型缓存重定向到项目 models/ 目录 (在 import torch 之前设置)
_this_dir = Path(__file__).resolve().parent.parent
_default_model_dir = str(_this_dir / "models")
os.environ.setdefault("TORCH_HOME", os.path.join(_default_model_dir, "torch"))
os.environ.setdefault("HF_HOME", os.path.join(_default_model_dir, "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_default_model_dir, "huggingface", "hub"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", os.path.join(_default_model_dir, "sentence_transformers"))
os.environ.setdefault("CLIP_CACHE", os.path.join(_default_model_dir, "clip"))


class SAMWrapper:
    """
    SAM 模型封装类，统一 SAM / SAM2 / SAM3 的调用接口。

    支持模型:
        - SAM (vit_b, vit_l, vit_h): 原始 Segment Anything Model
        - SAM2: Segment Anything Model 2，支持视频/时序数据
        - SAM3: Segment Anything Model 3，改进的分割精度

    模型选择建议 (6GB 显存推荐 vit_l):
        - vit_b: 显存 2~4GB，速度快，适合快速测试
        - vit_l: 显存 4~8GB，精度与速度平衡 (推荐 6GB 显存)
        - vit_h: 显存 8GB+，精度最高，适合建筑物提取等精细任务
    """

    # 模型类型与显存需求映射
    MODEL_INFO = {
        "vit_b": {"vram": "2~4GB", "params": "91M", "description": "轻量模型"},
        "vit_l": {"vram": "4~8GB", "params": "308M", "description": "中等模型"},
        "vit_h": {"vram": "8GB+", "params": "636M", "description": "大型模型"},
    }

    def __init__(
        self,
        model_type: str = "vit_l",
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        automatic: bool = False,
        sam_version: str = "sam1",
        model_dir: str = _default_model_dir,
        **kwargs,
    ):
        """
        初始化 SAM 模型。

        Args:
            model_type: 模型类型，可选 "vit_b", "vit_l", "vit_h"
            checkpoint_path: 模型权重路径，None 时自动下载到 model_dir
            device: 运行设备，如 "cuda:0", "cpu"，None 时自动选择
            automatic: 是否启用自动分割模式（生成全图 mask）
            sam_version: SAM 版本，"sam1", "sam2", "sam3"
            model_dir: 模型权重存放目录
            **kwargs: 传递给 SamGeo 的额外参数
        """
        self.model_type = model_type
        self.sam_version = sam_version
        self.automatic = automatic
        self.model_dir = model_dir
        self._sam = None
        self._image = None
        self._masks = None
        self._image_path = None

        # 自动检测设备
        if device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # 自动下载检查点到 models/ 目录
        if checkpoint_path is None:
            checkpoint_path = self._get_or_download_checkpoint()

        self.checkpoint_path = checkpoint_path
        self._extra_kwargs = kwargs

        print(f"[SAMWrapper] 模型版本: {sam_version}")
        print(f"[SAMWrapper] 模型类型: {model_type} ({self.MODEL_INFO.get(model_type, {}).get('description', '未知')})")
        print(f"[SAMWrapper] 运行设备: {self.device}")
        print(f"[SAMWrapper] 模型目录: {os.path.abspath(model_dir)}")
        print(f"[SAMWrapper] 自动分割: {automatic}")

    def _get_or_download_checkpoint(self) -> Optional[str]:
        """获取或下载模型检查点文件到 models/ 目录，带重试和校验。"""
        os.makedirs(self.model_dir, exist_ok=True)

        # SAM1 检查点 URL 与预期文件大小
        sam1_checkpoints = {
            "vit_h": ("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", 2566786440),
            "vit_l": ("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth", 1249599818),
            "vit_b": ("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",  393891747),
        }

        # SAM2 检查点 URL
        sam2_checkpoints = {
            "vit_h": ("https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_huge.pt", 0),
            "vit_l": ("https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt", 0),
            "vit_b": ("https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt", 0),
        }

        if self.sam_version == "sam1":
            urls = sam1_checkpoints
        elif self.sam_version in ("sam2", "sam3"):
            urls = sam2_checkpoints
        else:
            return None

        entry = urls.get(self.model_type)
        if entry is None:
            return None

        url, expected_size = entry
        filename = url.split("/")[-1]
        local_path = os.path.join(self.model_dir, filename)

        # 检查已有文件是否完整
        if os.path.exists(local_path):
            actual_size = os.path.getsize(local_path)
            if expected_size > 0 and actual_size < expected_size * 0.95:
                print(f"[SAMWrapper] 发现不完整文件: {local_path}")
                print(f"[SAMWrapper] 当前: {actual_size / 1024 / 1024:.1f}MB, "
                      f"期望: ~{expected_size / 1024 / 1024:.1f}MB")
                # 删除不完整文件，重新下载
                os.remove(local_path)
            else:
                print(f"[SAMWrapper] 找到本地模型: {local_path} ({actual_size / 1024 / 1024:.1f}MB)")
                return local_path

        # 下载: 最多重试 3 次
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            print(f"[SAMWrapper] 正在下载模型 (第 {attempt}/{max_retries} 次尝试)...")
            print(f"[SAMWrapper] URL: {url}")
            print(f"[SAMWrapper] 目标: {local_path}")

            try:
                success = self._download_file(url, local_path)
                if success:
                    # 校验文件大小
                    actual_size = os.path.getsize(local_path)
                    if expected_size > 0 and actual_size < expected_size * 0.95:
                        print(f"[SAMWrapper] 文件大小不匹配: "
                              f"{actual_size / 1024 / 1024:.1f}MB < "
                              f"期望 {expected_size / 1024 / 1024:.1f}MB")
                        os.remove(local_path)
                        if attempt < max_retries:
                            import time
                            wait = attempt * 5
                            print(f"[SAMWrapper] {wait} 秒后重试...")
                            time.sleep(wait)
                            continue
                    else:
                        print(f"[SAMWrapper] 下载完成: {local_path} ({actual_size / 1024 / 1024:.1f}MB)")
                        return local_path
                else:
                    # 下载函数返回 False，清理并重试
                    if os.path.exists(local_path):
                        os.remove(local_path)
                    if attempt < max_retries:
                        import time
                        wait = attempt * 5
                        print(f"[SAMWrapper] {wait} 秒后重试...")
                        time.sleep(wait)
            except KeyboardInterrupt:
                print("\n[SAMWrapper] 下载被用户中断")
                if os.path.exists(local_path):
                    print(f"[SAMWrapper] 已保留部分文件: {local_path}")
                break
            except Exception as e:
                print(f"[SAMWrapper] 下载异常: {e}")
                if os.path.exists(local_path):
                    os.remove(local_path)
                if attempt < max_retries:
                    import time
                    wait = attempt * 5
                    print(f"[SAMWrapper] {wait} 秒后重试...")
                    time.sleep(wait)

        # 所有重试失败，打印手动下载指引
        print()
        print("=" * 60)
        print("  模型自动下载失败，请手动下载:")
        print("=" * 60)
        print(f"  下载地址: {url}")
        print(f"  保存位置: {local_path}")
        print()
        print("  方法一: 浏览器打开上面的链接下载")
        print("  方法二: 使用 PowerShell:")
        print(f'    Invoke-WebRequest -Uri "{url}" -OutFile "{local_path}"')
        print("  方法三: 使用 curl:")
        print(f'    curl -L -o "{local_path}" "{url}"')
        print()
        print("  下载完成后重新运行脚本即可。")
        print("=" * 60)
        return None

    def _download_file(self, url: str, local_path: str) -> bool:
        """
        下载文件，带进度显示和超时处理。

        依次尝试: torch.hub → requests (stream) → urllib

        Returns:
            True 下载成功, False 下载失败
        """
        # 方法 1: torch.hub (有进度条)
        try:
            import torch
            torch.hub.download_url_to_file(url, local_path, progress=True)
            return os.path.exists(local_path) and os.path.getsize(local_path) > 0
        except Exception as e:
            if os.path.exists(local_path):
                os.remove(local_path)
            print(f"[SAMWrapper] torch.hub 下载失败: {e}")

        # 方法 2: requests stream (有进度条 + 超时)
        try:
            import requests
            from tqdm import tqdm

            session = requests.Session()
            # 增加超时和连接池配置
            response = session.get(url, stream=True, timeout=(30, 300))
            response.raise_for_status()

            total = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(local_path, "wb") as f:
                with tqdm(total=total, unit="B", unit_scale=True,
                          desc=os.path.basename(local_path)) as pbar:
                    for chunk in response.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            pbar.update(len(chunk))

            if total > 0 and downloaded < total * 0.95:
                print(f"[SAMWrapper] 下载不完整: {downloaded / 1024 / 1024:.1f}MB / {total / 1024 / 1024:.1f}MB")
                os.remove(local_path)
                return False
            return True

        except Exception as e:
            if os.path.exists(local_path):
                os.remove(local_path)
            print(f"[SAMWrapper] requests 下载失败: {e}")

        # 方法 3: urllib (最基础)
        try:
            import urllib.request
            print(f"[SAMWrapper] 使用 urllib 下载 (较慢)...")
            urllib.request.urlretrieve(url, local_path)
            return os.path.exists(local_path) and os.path.getsize(local_path) > 0
        except Exception as e:
            if os.path.exists(local_path):
                os.remove(local_path)
            print(f"[SAMWrapper] urllib 下载失败: {e}")

        return False

    def _init_sam(self):
        """延迟初始化 SamGeo 实例。"""
        if self._sam is not None:
            return

        # 构建传给 SamGeo 的 kwargs
        sam_kwargs = dict(self._extra_kwargs)
        if self.checkpoint_path is not None:
            # SamGeo 通过 kwargs["checkpoint"] 接收完整路径，跳过内部下载
            sam_kwargs["checkpoint"] = self.checkpoint_path
        else:
            # 下载失败时，让 SamGeo 内部下载到 models/ 目录
            sam_kwargs["checkpoint_dir"] = self.model_dir

        try:
            self._do_init_sam(sam_kwargs)
        except TypeError as e:
            # SamGeo 可能不支持某些参数，逐步移除后重试
            if "checkpoint_dir" in str(e):
                sam_kwargs.pop("checkpoint_dir", None)
                print("[SAMWrapper] SamGeo 不支持 checkpoint_dir，回退")
                self._do_init_sam(sam_kwargs)
            elif "sam_version" in str(e):
                sam_kwargs.pop("sam_version", None)
                self._do_init_sam(sam_kwargs)
            else:
                raise
        except ImportError as e:
            raise ImportError(
                f"无法导入 samgeo，请确保已安装: pip install samgeo\n"
                f"或安装 geoai: pip install geoai-py\n"
                f"原始错误: {e}"
            )

    def _do_init_sam(self, sam_kwargs: dict):
        """实际执行 SamGeo 初始化。所有参数通过 **sam_kwargs 传递。"""
        if self.sam_version == "sam1":
            from samgeo import SamGeo
            self._sam = SamGeo(
                model_type=self.model_type,
                device=self.device,
                automatic=self.automatic,
                **sam_kwargs,
            )
        elif self.sam_version == "sam2":
            try:
                from samgeo import SamGeo2
                self._sam = SamGeo2(
                    model_type=self.model_type,
                    device=self.device,
                    **sam_kwargs,
                )
            except ImportError:
                print("[SAMWrapper] SamGeo2 不可用，回退到 SamGeo")
                from samgeo import SamGeo
                self._sam = SamGeo(
                    model_type=self.model_type,
                    device=self.device,
                    automatic=self.automatic,
                    **sam_kwargs,
                )
        elif self.sam_version == "sam3":
            try:
                from samgeo import SamGeo
                sam_kwargs_copy = dict(sam_kwargs)
                sam_kwargs_copy["sam_version"] = "sam3"
                self._sam = SamGeo(
                    model_type=self.model_type,
                    device=self.device,
                    automatic=self.automatic,
                    **sam_kwargs_copy,
                )
            except TypeError:
                from samgeo import SamGeo
                self._sam = SamGeo(
                    model_type=self.model_type,
                    device=self.device,
                    automatic=self.automatic,
                    **sam_kwargs,
                )
        else:
            raise ValueError(f"不支持的 SAM 版本: {self.sam_version}")

        print(f"[SAMWrapper] 模型加载完成")

    def set_image(self, image_path: str) -> None:
        """
        加载遥感影像。

        Args:
            image_path: 影像文件路径，支持 GeoTIFF、PNG、JPG 等格式
        """
        self._init_sam()
        self._image_path = image_path

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"影像文件不存在: {image_path}")

        self._sam.set_image(image_path)
        print(f"[SAMWrapper] 影像已加载: {image_path}")

        # 获取影像信息
        try:
            import rasterio
            with rasterio.open(image_path) as src:
                print(f"[SAMWrapper] 影像尺寸: {src.width} x {src.height}")
                print(f"[SAMWrapper] 波段数: {src.count}")
                print(f"[SAMWrapper] CRS: {src.crs}")
        except Exception:
            pass

    def generate_masks_by_points(
        self,
        points: List[List[int]],
        point_labels: Optional[List[int]] = None,
        multimask_output: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        通过点提示生成分割 Mask。

        Args:
            points: 点坐标列表，格式 [[x1, y1], [x2, y2], ...]
            point_labels: 点标签列表，1=前景，0=背景。None 时默认为全前景。
            multimask_output: 是否输出多个 mask（3个候选）
            **kwargs: 额外参数

        Returns:
            分割 Mask 数组

        示例:
            # 单点标注
            >>> masks = sam.generate_masks_by_points([[750, 370]])

            # 多点标注（前景 + 背景）
            >>> masks = sam.generate_masks_by_points(
            ...     [[750, 370], [1125, 625]],
            ...     point_labels=[1, 0]  # 前景 + 背景
            ... )
        """
        self._init_sam()

        if point_labels is None:
            point_labels = [1] * len(points)

        # 坐标转 float，避免 SamGeo 内部类型分支错误
        points = [[float(v) for v in p] for p in points]

        print(f"[SAMWrapper] 点提示分割: {len(points)} 个点")
        print(f"[SAMWrapper] 前景点: {sum(1 for l in point_labels if l == 1)}")
        print(f"[SAMWrapper] 背景点: {sum(1 for l in point_labels if l == 0)}")

        self._sam.predict(
            point_coords=points,
            point_labels=point_labels,
            multimask_output=multimask_output,
            **kwargs,
        )

        self._masks = self._sam.masks
        return self._masks

    def generate_masks_by_box(
        self,
        box: List[int],
        multimask_output: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        通过单个框提示生成分割 Mask。

        Args:
            box: 边界框坐标 [xmin, ymin, xmax, ymax]
            multimask_output: 是否输出多个 mask
            **kwargs: 额外参数

        Returns:
            分割 Mask 数组

        示例:
            >>> masks = sam.generate_masks_by_box([100, 120, 500, 480])
        """
        self._init_sam()

        # SamGeo.predict() 内部通过 isinstance(boxes[0], float) 判断分支，
        # int 坐标会走错 predict_torch 路径，需要先转为 float
        box = [float(v) for v in box]

        print(f"[SAMWrapper] 框提示分割: {box}")

        self._sam.predict(
            boxes=box,
            multimask_output=multimask_output,
            **kwargs,
        )

        self._masks = self._sam.masks
        return self._masks

    def generate_masks_by_boxes(
        self,
        boxes: List[List[int]],
        multimask_output: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        通过多个框提示批量生成分割 Mask。

        Args:
            boxes: 边界框列表 [[xmin1, ymin1, xmax1, ymax1], ...]
            multimask_output: 是否输出多个 mask
            **kwargs: 额外参数

        Returns:
            分割 Mask 数组

        示例:
            >>> masks = sam.generate_masks_by_boxes([
            ...     [100, 120, 500, 480],
            ...     [600, 300, 900, 700],
            ... ])
        """
        self._init_sam()

        print(f"[SAMWrapper] 多框批量分割: {len(boxes)} 个框")

        # 多框走 predict_torch 路径，需要将 boxes 转为 torch tensor 并设置到 self._sam.boxes
        # 因为 SamGeo.predict() 对多框 + point_crs=None 的情况存在 numpy/torch 类型 bug
        import torch
        float_boxes = [[float(v) for v in b] for b in boxes]
        input_boxes = torch.tensor(float_boxes, dtype=torch.float32, device=self.device)
        input_boxes = self._sam.predictor.transform.apply_boxes_torch(
            input_boxes, self._sam.image.shape[:2]
        )
        self._sam.boxes = input_boxes

        masks, scores, logits = self._sam.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=input_boxes,
            multimask_output=True,
        )
        self._sam.masks = masks.cpu().numpy() if hasattr(masks, 'cpu') else masks
        self._sam.scores = scores.cpu().numpy() if hasattr(scores, 'cpu') else scores
        self._sam.logits = logits

        self._masks = self._sam.masks
        return self._masks

        self._masks = self._sam.masks
        return self._masks

    def generate_masks_auto(
        self,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        min_mask_region_area: int = 100,
        **kwargs,
    ) -> np.ndarray:
        """
        自动分割模式：对全图进行网格采样，自动发现所有目标。

        Args:
            points_per_side: 每边采样点数（越大越密集）
            pred_iou_thresh: IoU 阈值
            stability_score_thresh: 稳定性分数阈值
            min_mask_region_area: 最小 mask 区域面积
            **kwargs: 额外参数

        Returns:
            分割 Mask 数组
        """
        self._init_sam()

        if not self.automatic:
            print("[SAMWrapper] 警告: 模型未以自动模式初始化，尝试重新初始化...")
            from samgeo import SamGeo
            _auto_kwargs = dict(self._extra_kwargs)
            if self.checkpoint_path is not None:
                _auto_kwargs["checkpoint"] = self.checkpoint_path
            else:
                _auto_kwargs["checkpoint_dir"] = self.model_dir
            self._sam = SamGeo(
                model_type=self.model_type,
                device=self.device,
                automatic=True,
                **_auto_kwargs,
            )
            if self._image_path:
                self._sam.set_image(self._image_path)

        if self._image_path is None:
            raise RuntimeError("请先调用 set_image() 加载影像")

        print(f"[SAMWrapper] 自动分割: 每边 {points_per_side} 个采样点")

        # 尝试用自定义参数配置 mask generator，然后调用 generate()
        try:
            from segment_anything import SamAutomaticMaskGenerator
            predictor = self._sam.predictor
            self._sam.mask_generator = SamAutomaticMaskGenerator(
                model=predictor.model,
                points_per_side=points_per_side,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                min_mask_region_area=min_mask_region_area,
            )
        except (ImportError, AttributeError):
            print("[SAMWrapper] 无法自定义 mask generator 参数，使用默认配置")

        self._sam.generate(
            source=self._image_path,
            min_size=min_mask_region_area,
            **kwargs,
        )

        self._masks = self._sam.masks
        return self._masks

    @property
    def masks(self) -> Optional[np.ndarray]:
        """获取最近一次生成的 Mask。"""
        return self._masks

    def save_masks(self, output_path: str, **kwargs) -> str:
        """
        保存 Mask 到文件。

        兼容 predict() 返回的 numpy 数组和 generate() 返回的字典列表两种格式。

        Args:
            output_path: 输出文件路径

        Returns:
            实际保存路径
        """
        self._init_sam()
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        masks = self._sam.masks
        if masks is None:
            print("[SAMWrapper] 警告: 没有 Mask 可保存")
            return output_path

        # predict() 返回 numpy 数组 (N, H, W)，需要自行保存
        if isinstance(masks, np.ndarray):
            # 选取最佳 mask (index 0 = 最高得分)
            mask_data = masks[0] if masks.ndim == 3 else masks
            mask_binary = (mask_data > 0).astype(np.uint8)

            # 优先用 rasterio 保存为 GeoTIFF（保留地理信息）
            saved = False
            if self._image_path and os.path.exists(self._image_path):
                try:
                    import rasterio
                    with rasterio.open(self._image_path) as src:
                        profile = src.profile
                        profile.update(count=1, dtype="uint8")
                        # 确保 mask 与影像尺寸匹配
                        if mask_binary.shape == (src.height, src.width):
                            with rasterio.open(output_path, "w", **profile) as dst:
                                dst.write(mask_binary, 1)
                            saved = True
                except Exception:
                    pass

            if not saved:
                # 回退为 numpy 保存
                npy_path = output_path.rsplit(".", 1)[0] + ".npy"
                np.save(npy_path, mask_binary)
                output_path = npy_path

            print(f"[SAMWrapper] Mask 已保存: {output_path}")
            return output_path

        # generate() 返回字典列表，使用 SamGeo 内置保存
        try:
            self._sam.save_masks(output=output_path, **kwargs)
            print(f"[SAMWrapper] Mask 已保存: {output_path}")
        except Exception as e:
            print(f"[SAMWrapper] SamGeo save_masks 失败 ({e})，尝试手动保存...")
            # 手动从字典列表中提取 mask
            if isinstance(masks, list) and len(masks) > 0 and isinstance(masks[0], dict):
                combined = np.zeros_like(masks[0]["segmentation"], dtype=np.uint8)
                for i, m in enumerate(masks):
                    combined[m["segmentation"]] = i + 1
                np.save(output_path.rsplit(".", 1)[0] + ".npy", combined)
                output_path = output_path.rsplit(".", 1)[0] + ".npy"
                print(f"[SAMWrapper] Mask 已保存 (numpy): {output_path}")

        return output_path

    def show_masks(
        self,
        figsize: Tuple[int, int] = (12, 8),
        title: str = "SAM Segmentation Result",
        save_path: Optional[str] = None,
        cmap: str = "viridis",
    ) -> None:
        """
        可视化 Mask 结果。

        Args:
            figsize: 图像尺寸
            title: 标题
            save_path: 保存路径（None 则显示）
            cmap: 颜色映射
        """
        try:
            self._sam.show_masks(figsize=figsize)
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"[SAMWrapper] 可视化已保存: {save_path}")
        except Exception:
            # 手动可视化
            if self._masks is not None:
                fig, axes = plt.subplots(1, 2, figsize=figsize)
                # 显示原始影像
                try:
                    import rasterio
                    with rasterio.open(self._image_path) as src:
                        img = src.read()
                        if img.shape[0] >= 3:
                            img_show = np.transpose(img[:3], (1, 2, 0))
                        else:
                            img_show = img[0]
                        # 归一化
                        img_show = (img_show - img_show.min()) / (img_show.max() - img_show.min() + 1e-8)
                        axes[0].imshow(img_show)
                except Exception:
                    axes[0].text(0.5, 0.5, "Image", ha="center", va="center")

                axes[0].set_title("Original Image")
                axes[0].axis("off")

                # 显示 Mask
                if isinstance(self._masks, np.ndarray):
                    if self._masks.ndim == 3:
                        mask_show = self._masks[0]
                    else:
                        mask_show = self._masks
                    axes[1].imshow(mask_show, cmap=cmap)
                axes[1].set_title(title)
                axes[1].axis("off")

                plt.tight_layout()
                if save_path:
                    plt.savefig(save_path, dpi=150, bbox_inches="tight")
                    print(f"[SAMWrapper] 可视化已保存: {save_path}")
                else:
                    plt.show()
                plt.close()

    def get_mask_info(self) -> Dict[str, Any]:
        """获取当前 Mask 的信息。"""
        info = {
            "has_masks": self._masks is not None,
            "image_path": self._image_path,
            "model_type": self.model_type,
            "sam_version": self.sam_version,
            "device": self.device,
        }
        if self._masks is not None:
            if isinstance(self._masks, np.ndarray):
                info["mask_shape"] = self._masks.shape
                info["mask_dtype"] = str(self._masks.dtype)
                if self._masks.ndim >= 2:
                    mask_binary = self._masks[0] if self._masks.ndim == 3 else self._masks
                    info["foreground_pixels"] = int(mask_binary.sum())
                    info["total_pixels"] = int(mask_binary.size)
                    info["coverage_ratio"] = float(mask_binary.sum() / mask_binary.size)
        return info

    def __repr__(self) -> str:
        return (
            f"SAMWrapper(version={self.sam_version}, "
            f"type={self.model_type}, "
            f"device={self.device}, "
            f"automatic={self.automatic})"
        )
