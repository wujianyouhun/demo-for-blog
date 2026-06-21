"""GeoAI Core - Sentinel-2 影像下载器 (基于 Planetary Computer STAC API)"""
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.transform import from_bounds

logger = logging.getLogger(__name__)


class DataDownloader:
    """从 Microsoft Planetary Computer 下载 Sentinel-2 L2A 影像数据。"""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_sentinel2(
        self,
        bbox: List[float],
        date_range: Tuple[str, str],
        max_cloud_cover: float = 20.0,
        bands: Optional[List[str]] = None,
    ) -> Path:
        """
        搜索并下载 Sentinel-2 L2A 影像。

        Args:
            bbox: 空间范围 [min_lon, min_lat, max_lon, max_lat]
            date_range: 时间范围 ("YYYY-MM-DD", "YYYY-MM-DD")
            max_cloud_cover: 最大云量百分比, 默认 20%
            bands: 要下载的波段列表, 默认 ["B02","B03","B04"] (RGB)

        Returns:
            保存的 GeoTIFF 文件路径
        """
        try:
            import planetary_computer as pc
            import xarray as xr
            from pystac_client import Client
        except ImportError as e:
            raise ImportError(
                "需要安装 pystac_client, planetary_computer, xarray: "
                "pip install pystac-client planetary-computer xarray"
            ) from e

        if bands is None:
            bands = ["B04", "B03", "B02"]  # RGB 顺序

        logger.info(
            "搜索 Sentinel-2 影像: bbox=%s, 日期=%s, 云量<=%.0f%%",
            bbox, date_range, max_cloud_cover,
        )

        try:
            catalog = Client.open(
                "https://planetarycomputer.microsoft.com/api/stac/v1"
            )

            search = catalog.search(
                collections=["sentinel-2-l2a"],
                bbox=bbox,
                datetime=f"{date_range[0]}/{date_range[1]}",
                query={"eo:cloud_cover": {"lt": max_cloud_cover}},
                sortby=[{"field": "eo:cloud_cover", "direction": "asc"}],
            )

            items = list(search.items())
            if not items:
                raise ValueError(
                    f"未找到符合条件的影像 (bbox={bbox}, 日期={date_range}, "
                    f"云量<={max_cloud_cover}%)"
                )

            # 选取云量最低的一景
            best_item = items[0]
            cloud_cover = best_item.properties.get("eo:cloud_cover", "N/A")
            logger.info(
                "选中影像: %s (云量=%s%%, 日期=%s)",
                best_item.id, cloud_cover,
                best_item.properties.get("datetime", "unknown"),
            )

            # 下载指定波段并合成
            band_arrays = []
            profile = None

            for band_name in bands:
                asset = best_item.assets.get(band_name)
                if asset is None:
                    raise ValueError(f"波段 {band_name} 在影像资产中不存在")

                signed_url = pc.sign_url(asset.href)
                logger.info("下载波段: %s", band_name)

                with rasterio.open(signed_url) as src:
                    data = src.read(1)
                    band_arrays.append(data.astype(np.float32))

                    if profile is None:
                        profile = src.profile.copy()

            # 合成多波段影像
            stack = np.stack(band_arrays, axis=0)

            # 保存为 GeoTIFF
            timestamp = date_range[0].replace("-", "")
            out_name = f"S2_{best_item.id}_{timestamp}.tif"
            out_path = self.output_dir / out_name

            profile.update(
                count=len(bands),
                dtype="float32",
                compress="deflate",
                tiled=True,
                blockxsize=256,
                blockysize=256,
            )

            with rasterio.open(out_path, "w", **profile) as dst:
                for i, band_data in enumerate(band_arrays):
                    dst.write(band_data, i + 1)

            logger.info("影像已保存: %s (shape=%s)", out_path, stack.shape)
            return out_path

        except ImportError:
            raise
        except ValueError:
            raise
        except Exception as e:
            logger.error("下载失败: %s", e, exc_info=True)
            raise RuntimeError(f"Sentinel-2 影像下载失败: {e}") from e

    def list_downloaded(self) -> List[Path]:
        """列出已下载的所有 GeoTIFF 文件。"""
        files = sorted(self.output_dir.glob("*.tif"))
        logger.info("已下载文件数: %d", len(files))
        return files
