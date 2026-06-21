"""GeoAI Core - 推理引擎 (滑动窗口推理 + 矢量化)"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.features import shapes
from scipy.ndimage import gaussian_filter
from shapely.geometry import shape
import geopandas as gpd

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    语义分割推理引擎, 支持滑动窗口、重叠概率平均、高斯平滑。
    """

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "cpu",
        tile_size: int = 256,
        overlap: int = 32,
        batch_size: int = 4,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.tile_size = tile_size
        self.overlap = overlap
        self.batch_size = batch_size

    @torch.no_grad()
    def predict(
        self,
        image_path: str | Path,
        output_path: str | Path,
        smoothing_sigma: float = 1.0,
        threshold: float = 0.5,
    ) -> Path:
        """
        对大幅影像执行滑动窗口推理, 输出分类结果 GeoTIFF。

        Args:
            image_path: 输入影像路径
            output_path: 输出分类图路径
            smoothing_sigma: 高斯平滑 sigma (0 表示不平滑)
            threshold: 概率阈值 (用于最终分类)

        Returns:
            输出文件路径
        """
        image_path = Path(image_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(image_path) as src:
            height, width = src.height, src.width
            num_bands = src.count
            profile = src.profile.copy()
            crs = src.crs
            transform = src.transform

            # 读取完整影像
            image = src.read().astype(np.float32)  # (C, H, W)
            if image.max() > 1.0:
                image = image / 255.0

        logger.info(
            "开始推理: %s (%dx%d, %d bands), tile=%d, overlap=%d",
            image_path.name, height, width, num_bands,
            self.tile_size, self.overlap,
        )

        stride = self.tile_size - self.overlap
        num_classes = self._get_num_classes()

        # 概率累积图
        prob_sum = np.zeros((num_classes, height, width), dtype=np.float64)
        count_map = np.zeros((1, height, width), dtype=np.float64)

        # 生成所有窗口位置
        windows = []
        for row in range(0, height, stride):
            for col in range(0, width, stride):
                r_end = min(row + self.tile_size, height)
                c_end = min(col + self.tile_size, width)
                r_start = max(0, r_end - self.tile_size)
                c_start = max(0, c_end - self.tile_size)
                windows.append((r_start, c_start, r_end, c_end))

        # 批量推理
        batch = []
        batch_windows = []

        for i, (r0, c0, r1, c1) in enumerate(windows):
            # 裁剪 tile
            tile = image[:, r0:r1, c0:c1]  # (C, tile_h, tile_w)

            # 如果 tile 尺寸不足 tile_size, 进行 padding
            pad_h = self.tile_size - tile.shape[1]
            pad_w = self.tile_size - tile.shape[2]
            if pad_h > 0 or pad_w > 0:
                tile = np.pad(
                    tile,
                    ((0, 0), (0, pad_h), (0, pad_w)),
                    mode="reflect",
                )

            batch.append(tile)
            batch_windows.append((r0, c0, r1, c1))

            # 达到 batch 大小或最后一个
            if len(batch) >= self.batch_size or i == len(windows) - 1:
                # 推理
                batch_tensor = torch.from_numpy(
                    np.stack(batch, axis=0)
                ).to(self.device)

                with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                    outputs = self.model(batch_tensor)

                probs = torch.softmax(outputs, dim=1).cpu().numpy()

                # 累积概率
                for j, (r0, c0, r1, c1) in enumerate(batch_windows):
                    h_actual = r1 - r0
                    w_actual = c1 - c0
                    prob_tile = probs[j, :, :h_actual, :w_actual]
                    prob_sum[:, r0:r1, c0:c1] += prob_tile
                    count_map[:, r0:r1, c0:c1] += 1

                batch = []
                batch_windows = []

        # 概率平均
        count_map = np.maximum(count_map, 1)
        prob_avg = prob_sum / count_map

        # 高斯平滑
        if smoothing_sigma > 0:
            for c in range(num_classes):
                prob_avg[c] = gaussian_filter(prob_avg[c], sigma=smoothing_sigma)

        # 最终分类
        classification = np.argmax(prob_avg, axis=0).astype(np.uint8)

        # 保存结果
        profile.update(
            count=1,
            dtype="uint8",
            compress="deflate",
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(classification, 1)

        logger.info("推理完成: %s (shape=%s)", output_path, classification.shape)
        return output_path

    def _get_num_classes(self) -> int:
        """从模型结构推断类别数。"""
        # 尝试从最后一层获取
        for module in reversed(list(self.model.modules())):
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                return module.out_channels
        return 6  # 默认

    def mask_to_vector(
        self,
        mask_path: str | Path,
        output_path: str | Path,
        min_area: int = 20,
        class_names: Optional[list] = None,
    ) -> Path:
        """
        将栅格分类图转换为矢量多边形 (GeoPackage)。

        Args:
            mask_path: 输入分类图路径
            output_path: 输出矢量文件路径 (.gpkg)
            min_area: 最小面积 (像素), 过滤掉小图斑
            class_names: 类别名称列表

        Returns:
            输出文件路径
        """
        mask_path = Path(mask_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if class_names is None:
            class_names = [
                "background", "building", "road",
                "water", "vegetation", "barren",
            ]

        with rasterio.open(mask_path) as src:
            mask = src.read(1)
            transform = src.transform
            crs = src.crs

        logger.info(
            "矢量化: %s (%dx%d, CRS=%s)",
            mask_path.name, mask.shape[1], mask.shape[0], crs,
        )

        # 提取多边形
        records = []
        for geom_value, geom_value_int in shapes(
            mask.astype(np.uint8), transform=transform
        ):
            class_id = int(geom_value_int)
            if class_id == 0:
                continue  # 跳过背景

            geom = shape(geom_value)
            if not geom.is_valid:
                geom = geom.buffer(0)

            area = geom.area
            if area < min_area:
                continue

            class_name = (
                class_names[class_id]
                if class_id < len(class_names)
                else f"class_{class_id}"
            )

            records.append({
                "geometry": geom,
                "class_id": class_id,
                "class_name": class_name,
                "area_px": area,
            })

        if not records:
            logger.warning("未提取到有效多边形")
            gdf = gpd.GeoDataFrame(
                columns=["class_id", "class_name", "area_px", "geometry"],
                geometry="geometry",
                crs=crs,
            )
        else:
            gdf = gpd.GeoDataFrame(records, crs=crs)

        gdf.to_file(output_path, driver="GPKG")
        logger.info("矢量化完成: %s (%d 个要素)", output_path, len(gdf))
        return output_path
