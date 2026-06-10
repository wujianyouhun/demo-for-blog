"""
Mask 后处理模块

SAM 输出的 Mask 通常存在:
- 毛刺 (噪声像素)
- 小碎片 (零散区域)
- 孔洞 (内部空洞)
- 边界抖动 (锯齿状边缘)

本模块提供完整的后处理流程，将粗糙 Mask 转化为高质量标签。
"""

import os
import numpy as np
from typing import Optional, Tuple, Union, List
from pathlib import Path


class MaskPostProcessor:
    """
    Mask 后处理工具类。

    提供去噪、平滑、孔洞填充、面积过滤、边界优化等功能。
    支持链式调用，方便构建自定义后处理流程。

    典型用法:
        >>> processor = MaskPostProcessor()
        >>> clean_mask = (processor
        ...     .load(mask)
        ...     .remove_small_objects(min_size=200)
        ...     .fill_holes()
        ...     .smooth(sigma=1.5)
        ...     .get_result())
    """

    def __init__(self):
        self._mask = None
        self._original_mask = None
        self._operations_log = []

    def load(self, mask: np.ndarray) -> "MaskPostProcessor":
        """
        加载 Mask。

        Args:
            mask: 输入 Mask 数组 (2D 布尔或 uint8)

        Returns:
            self（支持链式调用）
        """
        if isinstance(mask, np.ndarray):
            if mask.ndim == 3:
                mask = mask[0]
            self._mask = mask.astype(bool).copy()
            self._original_mask = self._mask.copy()
            self._operations_log = []
            self._log("load", shape=mask.shape)
        else:
            raise TypeError(f"期望 numpy 数组，得到 {type(mask)}")
        return self

    def _log(self, operation: str, **kwargs):
        """记录操作日志。"""
        self._operations_log.append({"operation": operation, **kwargs})

    def remove_small_objects(
        self,
        min_size: int = 200,
        connectivity: int = 2,
    ) -> "MaskPostProcessor":
        """
        去除小斑块（面积过滤）。

        去除 Mask 中面积小于 min_size 的连通区域，
        可有效消除噪声和碎片。

        Args:
            min_size: 最小面积（像素数）
            connectivity: 连通性 (1=4连通, 2=8连通)

        Returns:
            self

        示例:
            >>> processor.remove_small_objects(min_size=200)
        """
        if self._mask is None:
            raise RuntimeError("请先调用 load() 加载 Mask")

        try:
            from skimage.morphology import remove_small_objects
            before = self._mask.sum()
            # 新版 skimage (>=0.26) 弃用 min_size，改用 max_size
            # max_size 移除 <= 该值的对象，min_size 移除 < 该值的对象
            try:
                self._mask = remove_small_objects(
                    self._mask,
                    max_size=min_size - 1,
                    connectivity=connectivity,
                )
            except TypeError:
                # 旧版 skimage 回退到 min_size
                self._mask = remove_small_objects(
                    self._mask,
                    min_size=min_size,
                    connectivity=connectivity,
                )
            after = self._mask.sum()
            self._log(
                "remove_small_objects",
                min_size=min_size,
                pixels_removed=int(before - after),
            )
            print(f"[后处理] 去除小斑块: min_size={min_size}, 移除 {int(before - after)} 像素")
        except ImportError:
            print("[后处理] skimage 未安装，使用简化版本")
            self._remove_small_objects_simple(min_size)

        return self

    def _remove_small_objects_simple(self, min_size: int):
        """简化版小斑块去除（不依赖 skimage）。"""
        from scipy.ndimage import label
        labeled, num_features = label(self._mask)
        for i in range(1, num_features + 1):
            component = labeled == i
            if component.sum() < min_size:
                self._mask[component] = False

    def fill_holes(self) -> "MaskPostProcessor":
        """
        孔洞填充。

        填充 Mask 内部的孔洞，使目标区域完整连续。
        适用于建筑物、水体等实心目标。

        Returns:
            self

        示例:
            >>> processor.fill_holes()
        """
        if self._mask is None:
            raise RuntimeError("请先调用 load() 加载 Mask")

        try:
            from scipy.ndimage import binary_fill_holes
            before = self._mask.sum()
            self._mask = binary_fill_holes(self._mask)
            after = self._mask.sum()
            self._log(
                "fill_holes",
                pixels_added=int(after - before),
            )
            print(f"[后处理] 孔洞填充: 添加 {int(after - before)} 像素")
        except ImportError:
            print("[后处理] scipy 未安装，跳过孔洞填充")

        return self

    def opening(self, radius: int = 2) -> "MaskPostProcessor":
        """
        开运算（先腐蚀后膨胀）。

        可去除小的突出物和毛刺，平滑边界。

        Args:
            radius: 结构元素半径

        Returns:
            self
        """
        if self._mask is None:
            raise RuntimeError("请先调用 load() 加载 Mask")

        try:
            from skimage.morphology import opening, disk
            self._mask = opening(self._mask, disk(radius))
            self._log("opening", radius=radius)
            print(f"[后处理] 开运算: radius={radius}")
        except ImportError:
            print("[后处理] skimage 未安装，跳过开运算")

        return self

    def closing(self, radius: int = 3) -> "MaskPostProcessor":
        """
        闭运算（先膨胀后腐蚀）。

        可填充小的间隙和裂缝，连接邻近区域。

        Args:
            radius: 结构元素半径

        Returns:
            self
        """
        if self._mask is None:
            raise RuntimeError("请先调用 load() 加载 Mask")

        try:
            from skimage.morphology import closing, disk
            self._mask = closing(self._mask, disk(radius))
            self._log("closing", radius=radius)
            print(f"[后处理] 闭运算: radius={radius}")
        except ImportError:
            print("[后处理] skimage 未安装，跳过闭运算")

        return self

    def smooth(self, sigma: float = 1.5, threshold: float = 0.5) -> "MaskPostProcessor":
        """
        边界平滑。

        使用高斯滤波平滑 Mask 边界，减少锯齿效应。

        Args:
            sigma: 高斯核标准差（越大越平滑）
            threshold: 二值化阈值

        Returns:
            self

        示例:
            >>> processor.smooth(sigma=1.5)
        """
        if self._mask is None:
            raise RuntimeError("请先调用 load() 加载 Mask")

        try:
            from scipy.ndimage import gaussian_filter
            smoothed = gaussian_filter(self._mask.astype(float), sigma=sigma)
            self._mask = smoothed > threshold
            self._log("smooth", sigma=sigma, threshold=threshold)
            print(f"[后处理] 边界平滑: sigma={sigma}")
        except ImportError:
            print("[后处理] scipy 未安装，跳过平滑")

        return self

    def dilate(self, radius: int = 2) -> "MaskPostProcessor":
        """
        膨胀操作。

        扩大 Mask 区域，可用于补偿分割偏紧的边界。

        Args:
            radius: 膨胀半径

        Returns:
            self
        """
        if self._mask is None:
            raise RuntimeError("请先调用 load() 加载 Mask")

        try:
            from skimage.morphology import dilation, disk
            self._mask = dilation(self._mask, disk(radius))
            self._log("dilate", radius=radius)
            print(f"[后处理] 膨胀: radius={radius}")
        except ImportError:
            from scipy.ndimage import binary_dilation
            struct = np.ones((2 * radius + 1, 2 * radius + 1))
            self._mask = binary_dilation(self._mask, structure=struct)
            self._log("dilate", radius=radius)
            print(f"[后处理] 膨胀 (scipy): radius={radius}")

        return self

    def erode(self, radius: int = 2) -> "MaskPostProcessor":
        """
        腐蚀操作。

        缩小 Mask 区域，可用于分离粘连的目标。

        Args:
            radius: 腐蚀半径

        Returns:
            self
        """
        if self._mask is None:
            raise RuntimeError("请先调用 load() 加载 Mask")

        try:
            from skimage.morphology import erosion, disk
            self._mask = erosion(self._mask, disk(radius))
            self._log("erode", radius=radius)
            print(f"[后处理] 腐蚀: radius={radius}")
        except ImportError:
            from scipy.ndimage import binary_erosion
            struct = np.ones((2 * radius + 1, 2 * radius + 1))
            self._mask = binary_erosion(self._mask, structure=struct)
            self._log("erode", radius=radius)
            print(f"[后处理] 腐蚀 (scipy): radius={radius}")

        return self

    def buffer(self, pixels: int = 3) -> "MaskPostProcessor":
        """
        缓冲区操作（先膨胀后腐蚀回去，类似 closing）。

        用于填充边界上的小缺口。

        Args:
            pixels: 缓冲区像素数

        Returns:
            self
        """
        self.dilate(radius=pixels)
        self.erode(radius=pixels)
        self._log("buffer", pixels=pixels)
        return self

    def filter_by_area(
        self,
        min_area: Optional[int] = None,
        max_area: Optional[int] = None,
    ) -> "MaskPostProcessor":
        """
        按面积范围过滤 Mask。

        只保留面积在 [min_area, max_area] 范围内的区域。

        Args:
            min_area: 最小面积（像素数）
            max_area: 最大面积（像素数）

        Returns:
            self
        """
        if self._mask is None:
            raise RuntimeError("请先调用 load() 加载 Mask")

        try:
            from scipy.ndimage import label
            labeled, num_features = label(self._mask)
            filtered_mask = np.zeros_like(self._mask)

            for i in range(1, num_features + 1):
                component = labeled == i
                area = component.sum()

                if min_area is not None and area < min_area:
                    continue
                if max_area is not None and area > max_area:
                    continue

                filtered_mask[component] = True

            removed = num_features - int(label(filtered_mask)[1])
            self._mask = filtered_mask
            self._log("filter_by_area", min_area=min_area, max_area=max_area, removed=removed)
            print(f"[后处理] 面积过滤: min={min_area}, max={max_area}, 移除 {removed} 个区域")

        except ImportError:
            print("[后处理] scipy 未安装，跳过面积过滤")

        return self

    def keep_largest(self) -> "MaskPostProcessor":
        """
        只保留最大的连通区域。

        Returns:
            self
        """
        if self._mask is None:
            raise RuntimeError("请先调用 load() 加载 Mask")

        from scipy.ndimage import label
        labeled, num_features = label(self._mask)

        if num_features == 0:
            return self

        # 找到最大的连通区域
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0  # 排除背景
        largest_label = sizes.argmax()
        self._mask = labeled == largest_label

        self._log("keep_largest", kept_area=int(self._mask.sum()))
        print(f"[后处理] 保留最大区域: {int(self._mask.sum())} 像素")

        return self

    def convex_hull(self) -> "MaskPostProcessor":
        """
        凸包处理。

        将 Mask 区域替换为其凸包，适用于形状规则的目标。

        Returns:
            self
        """
        if self._mask is None:
            raise RuntimeError("请先调用 load() 加载 Mask")

        try:
            from scipy.ndimage import label
            from skimage.morphology import convex_hull_image

            labeled, num_features = label(self._mask)
            result = np.zeros_like(self._mask)

            for i in range(1, num_features + 1):
                component = labeled == i
                bbox = np.argwhere(component)
                y_min, x_min = bbox.min(axis=0)
                y_max, x_max = bbox.max(axis=0) + 1

                sub_mask = component[y_min:y_max, x_min:x_max]
                hull = convex_hull_image(sub_mask)
                result[y_min:y_max, x_min:x_max] |= hull

            self._mask = result
            self._log("convex_hull")
            print("[后处理] 凸包处理完成")

        except ImportError:
            print("[后处理] skimage 未安装，跳过凸包处理")

        return self

    def get_result(self) -> np.ndarray:
        """
        获取处理后的 Mask。

        Returns:
            处理后的布尔数组
        """
        if self._mask is None:
            raise RuntimeError("没有可返回的 Mask，请先调用 load()")
        return self._mask.copy()

    def get_result_uint8(self) -> np.ndarray:
        """获取 uint8 类型的 Mask (0/255)。"""
        if self._mask is None:
            raise RuntimeError("没有可返回的 Mask")
        return (self._mask.astype(np.uint8)) * 255

    def get_operations_log(self) -> list:
        """获取操作日志。"""
        return self._operations_log.copy()

    def get_statistics(self) -> dict:
        """获取当前 Mask 的统计信息。"""
        if self._mask is None:
            return {}

        from scipy.ndimage import label
        labeled, num_features = label(self._mask)

        areas = []
        for i in range(1, num_features + 1):
            areas.append(int((labeled == i).sum()))

        return {
            "total_pixels": int(self._mask.size),
            "foreground_pixels": int(self._mask.sum()),
            "coverage_ratio": float(self._mask.sum() / self._mask.size),
            "num_objects": num_features,
            "object_areas": areas,
            "min_area": min(areas) if areas else 0,
            "max_area": max(areas) if areas else 0,
            "mean_area": float(np.mean(areas)) if areas else 0,
        }

    @staticmethod
    def default_pipeline(
        mask: np.ndarray,
        min_size: int = 200,
        fill_holes_flag: bool = True,
        smooth_sigma: float = 1.5,
        opening_radius: int = 2,
        closing_radius: int = 3,
    ) -> np.ndarray:
        """
        默认后处理流程（一步到位）。

        流程: 开运算 → 去小斑块 → 孔洞填充 → 闭运算 → 平滑

        Args:
            mask: 输入 Mask
            min_size: 最小面积阈值
            fill_holes_flag: 是否填充孔洞
            smooth_sigma: 平滑参数
            opening_radius: 开运算半径
            closing_radius: 闭运算半径

        Returns:
            处理后的 Mask

        示例:
            >>> clean_mask = MaskPostProcessor.default_pipeline(raw_mask)
        """
        processor = MaskPostProcessor()
        result = processor.load(mask)

        if opening_radius > 0:
            result = result.opening(radius=opening_radius)

        if min_size > 0:
            result = result.remove_small_objects(min_size=min_size)

        if fill_holes_flag:
            result = result.fill_holes()

        if closing_radius > 0:
            result = result.closing(radius=closing_radius)

        if smooth_sigma > 0:
            result = result.smooth(sigma=smooth_sigma)

        return result.get_result()

    def save(self, output_path: str, reference_image: Optional[str] = None) -> str:
        """
        保存处理后的 Mask 到文件。

        Args:
            output_path: 输出文件路径
            reference_image: 参考影像路径（用于保留地理信息）

        Returns:
            保存路径
        """
        if self._mask is None:
            raise RuntimeError("没有可保存的 Mask")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        mask_uint8 = self._mask.astype(np.uint8)

        if reference_image:
            try:
                import rasterio
                with rasterio.open(reference_image) as src:
                    profile = src.profile
                    profile.update(count=1, dtype="uint8")
                    with rasterio.open(output_path, "w", **profile) as dst:
                        dst.write(mask_uint8, 1)
                    print(f"[后处理] Mask 已保存为 GeoTIFF: {output_path}")
                    return output_path
            except ImportError:
                pass

        # 回退: 保存为 numpy 数组
        npy_path = output_path.rsplit(".", 1)[0] + ".npy"
        np.save(npy_path, mask_uint8)
        print(f"[后处理] Mask 已保存为 NumPy: {npy_path}")
        return npy_path

    def visualize(
        self,
        original_image: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 6),
    ) -> None:
        """
        可视化对比：原始 Mask vs 处理后 Mask。

        Args:
            original_image: 原始影像数组
            save_path: 保存路径
            figsize: 图像尺寸
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # 原始影像
        if original_image is not None:
            if original_image.ndim == 3 and original_image.shape[0] >= 3:
                img_show = np.transpose(original_image[:3], (1, 2, 0))
                img_show = (img_show - img_show.min()) / (img_show.max() - img_show.min() + 1e-8)
            elif original_image.ndim == 2:
                img_show = original_image
            else:
                img_show = original_image
            axes[0].imshow(img_show)
        axes[0].set_title("原始影像")
        axes[0].axis("off")

        # 原始 Mask
        if self._original_mask is not None:
            axes[1].imshow(self._original_mask, cmap="Reds")
            axes[1].set_title(f"原始 Mask ({int(self._original_mask.sum())} 像素)")
        axes[1].axis("off")

        # 处理后 Mask
        if self._mask is not None:
            axes[2].imshow(self._mask, cmap="Greens")
            axes[2].set_title(f"处理后 Mask ({int(self._mask.sum())} 像素)")
        axes[2].axis("off")

        plt.suptitle("Mask 后处理对比", fontsize=14)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[后处理] 对比图已保存: {save_path}")
        else:
            plt.show()
        plt.close()

    def __repr__(self) -> str:
        ops = len(self._operations_log)
        has_mask = self._mask is not None
        return f"MaskPostProcessor(has_mask={has_mask}, operations={ops})"
