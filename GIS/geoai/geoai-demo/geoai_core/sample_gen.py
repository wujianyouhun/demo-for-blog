"""GeoAI Core - 滑动窗口样本生成器"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.windows import Window

logger = logging.getLogger(__name__)


class SampleGenerator:
    """
    使用滑动窗口从大幅遥感影像和对应标签中裁剪训练样本。
    """

    def __init__(
        self,
        image_dir: str | Path,
        label_dir: str | Path,
        output_dir: str | Path,
        tile_size: int = 256,
        stride: int = 128,
    ):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.output_dir = Path(output_dir)
        self.tile_size = tile_size
        self.stride = stride

        (self.output_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels").mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        image_path: str | Path,
        label_path: str | Path,
        nodata_value: Optional[float] = None,
    ) -> int:
        """
        从单对影像-标签文件中生成滑动窗口样本。

        Args:
            image_path: 输入影像路径
            label_path: 对应标签路径
            nodata_value: 忽略全为 nodata 的窗口

        Returns:
            生成的样本数量
        """
        image_path = Path(image_path)
        label_path = Path(label_path)

        if not image_path.exists():
            raise FileNotFoundError(f"影像文件不存在: {image_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"标签文件不存在: {label_path}")

        logger.info(
            "开始生成样本: 影像=%s, 标签=%s, tile=%d, stride=%d",
            image_path.name, label_path.name, self.tile_size, self.stride,
        )

        count = 0
        stem = image_path.stem

        with rasterio.open(image_path) as img_src, rasterio.open(label_path) as lbl_src:
            img_height, img_width = img_src.height, img_src.width
            lbl_height, lbl_width = lbl_src.height, lbl_src.width

            if (img_height, img_width) != (lbl_height, lbl_width):
                raise ValueError(
                    f"影像尺寸 ({img_height}x{img_width}) 与标签尺寸 "
                    f"({lbl_height}x{lbl_width}) 不一致"
                )

            img_profile = img_src.profile.copy()
            lbl_profile = lbl_src.profile.copy()

            # 更新输出 profile
            img_profile.update(
                height=self.tile_size,
                width=self.tile_size,
                compress="deflate",
            )
            lbl_profile.update(
                height=self.tile_size,
                width=self.tile_size,
                compress="deflate",
            )

            # 滑动窗口
            for row in range(0, img_height - self.tile_size + 1, self.stride):
                for col in range(0, img_width - self.tile_size + 1, self.stride):
                    window = Window(col, row, self.tile_size, self.tile_size)

                    # 读取影像和标签
                    img_tile = img_src.read(window=window)
                    lbl_tile = lbl_src.read(window=window)

                    # 跳过全 nodata 的窗口
                    if nodata_value is not None:
                        if np.all(img_tile == nodata_value):
                            continue
                        if np.all(lbl_tile == nodata_value):
                            continue

                    # 跳过全黑（无数据）窗口
                    if np.all(img_tile == 0):
                        continue

                    # 保存样本
                    img_out = (
                        self.output_dir / "images"
                        / f"{stem}_r{row}_c{col}.tif"
                    )
                    lbl_out = (
                        self.output_dir / "labels"
                        / f"{stem}_r{row}_c{col}.tif"
                    )

                    # 计算窗口对应的 transform
                    img_win_transform = rasterio.windows.transform(
                        window, img_src.transform
                    )
                    lbl_win_transform = rasterio.windows.transform(
                        window, lbl_src.transform
                    )

                    img_profile_out = img_profile.copy()
                    img_profile_out["transform"] = img_win_transform

                    lbl_profile_out = lbl_profile.copy()
                    lbl_profile_out["transform"] = lbl_win_transform

                    with rasterio.open(img_out, "w", **img_profile_out) as dst:
                        dst.write(img_tile)

                    with rasterio.open(lbl_out, "w", **lbl_profile_out) as dst:
                        dst.write(lbl_tile)

                    count += 1

        logger.info("生成完成: 共 %d 个样本", count)
        return count

    def generate_batch(self, image_label_pairs: list) -> int:
        """
        批量生成样本。

        Args:
            image_label_pairs: [(image_path, label_path), ...]

        Returns:
            总样本数
        """
        total = 0
        for img_path, lbl_path in image_label_pairs:
            try:
                n = self.generate(img_path, lbl_path)
                total += n
            except Exception as e:
                logger.error("处理失败 %s: %s", img_path, e)
        logger.info("批量生成完成: 总计 %d 个样本", total)
        return total
