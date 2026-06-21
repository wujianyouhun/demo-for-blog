"""双时相数据下载器"""
import logging
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BiTemporalDownloader:
    """双时相 Sentinel-2 影像下载器"""

    STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.time_a_dir = self.output_dir / "time_a"
        self.time_b_dir = self.output_dir / "time_b"
        self.time_a_dir.mkdir(parents=True, exist_ok=True)
        self.time_b_dir.mkdir(parents=True, exist_ok=True)

    def download_pair(self, bbox, date_a, date_b, max_cloud_cover=20.0,
                      bands=None, out_name=None):
        if bands is None:
            bands = ["B04", "B03", "B02"]
        prefix = out_name or f"s2_{bbox[0]:.2f}_{bbox[1]:.2f}"
        logger.info(f"下载: A={date_a}, B={date_b}")

        path_a = self._download_single(bbox, date_a, max_cloud_cover, bands,
                                        self.time_a_dir, f"{prefix}_A")
        path_b = self._download_single(bbox, date_b, max_cloud_cover, bands,
                                        self.time_b_dir, f"{prefix}_B")
        if path_a is None or path_b is None:
            raise RuntimeError("未能下载到足够的影像")

        aligned_b = self._align_pair(path_a, path_b)
        return {"time_a": path_a, "time_b": aligned_b,
                "meta": {"date_a": date_a, "date_b": date_b, "bbox": bbox}}

    def _download_single(self, bbox, date, max_cloud, bands, output_dir, prefix):
        from pystac_client import Client
        import planetary_computer as pc
        from rasterio.warp import transform_bounds
        from rasterio.windows import from_bounds

        client = Client.open(self.STAC_API_URL)
        parts = date.split("-")
        year, month = int(parts[0]), int(parts[1])
        date_end = f"{year + (1 if month == 12 else 0)}-{(month % 12) + 1:02d}-01"

        search = client.search(collections=["sentinel-2-l2a"], bbox=bbox,
                                datetime=f"{date}/{date_end}",
                                query={"eo:cloud_cover": {"lt": max_cloud}}, max_items=10)
        items = sorted(search.items(), key=lambda it: it.properties.get("eo:cloud_cover", 100))
        if not items:
            logger.warning(f"未找到 {date} 附近影像")
            return None

        best = items[0]
        signed = pc.sign(best)
        out_path = output_dir / f"{prefix}.tif"
        if out_path.exists():
            return out_path

        # GDAL HTTP 超时配置，避免无限等待
        gdal_opts = {
            "GDAL_HTTP_TIMEOUT": "60",
            "GDAL_HTTP_CONNECTTIMEOUT": "15",
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
            "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.jp2",
        }
        env = rasterio.Env(**gdal_opts)

        band_arrays = []
        ref_transform = ref_shape = ref_crs = None
        for band_name in bands:
            asset = signed.assets.get(band_name)
            if not asset:
                continue
            try:
                with env, rasterio.open(asset.href) as src:
                    # 首次打开时计算 bbox 对应的裁剪窗口
                    if ref_crs is None:
                        ref_crs = src.crs
                        # 将 bbox (WGS84) 变换到影像 CRS
                        img_bounds = transform_bounds("EPSG:4326", src.crs,
                                                      bbox[0], bbox[1], bbox[2], bbox[3])
                        window = from_bounds(*img_bounds, transform=src.transform)
                        # 取整数窗口并裁剪到影像范围
                        win = window.round_offsets().round_lengths()
                        col_off = max(0, int(win.col_off))
                        row_off = max(0, int(win.row_off))
                        width = min(int(win.width), src.width - col_off)
                        height = min(int(win.height), src.height - row_off)
                        if width <= 0 or height <= 0:
                            logger.warning(f"bbox 超出影像范围: {band_name}")
                            continue
                        crop_window = rasterio.windows.Window(col_off, row_off, width, height)
                        # 计算裁剪后的 transform
                        ref_transform = rasterio.windows.transform(crop_window, src.transform)
                        ref_shape = (height, width)
                        logger.info(f"窗口裁剪: {width}x{height} (原图 {src.width}x{src.height})")

                    data = src.read(1, window=crop_window)
                    band_arrays.append(data.astype(np.float32))
            except Exception as e:
                logger.error(f"读取 {band_name} 失败: {e}")
                continue

        if not band_arrays:
            return None

        stack = np.clip(np.stack(band_arrays) / 3000.0 * 255, 0, 255).astype(np.uint8)
        profile = {"driver": "GTiff", "dtype": "uint8", "width": ref_shape[1],
                    "height": ref_shape[0], "count": len(band_arrays),
                    "crs": ref_crs, "transform": ref_transform, "compress": "lzw"}
        with rasterio.open(out_path, "w", **profile) as dst:
            for i in range(len(band_arrays)):
                dst.write(stack[i], i + 1)

        logger.info(f"下载完成: {out_path}")
        return out_path

    def _align_pair(self, ref_path, target_path):
        out_path = target_path.parent / f"{target_path.stem}_aligned.tif"
        if out_path.exists():
            return out_path

        with rasterio.open(ref_path) as ref:
            rt, rc, rw, rh = ref.transform, ref.crs, ref.width, ref.height
        with rasterio.open(target_path) as src:
            aligned = np.zeros((src.count, rh, rw), dtype=src.dtypes[0])
            for i in range(src.count):
                reproject(source=rasterio.band(src, i + 1), destination=aligned[i],
                          src_transform=src.transform, src_crs=src.crs,
                          dst_transform=rt, dst_crs=rc, resampling=Resampling.bilinear)

        profile = {"driver": "GTiff", "dtype": "uint8", "width": rw, "height": rh,
                    "count": aligned.shape[0], "crs": rc, "transform": rt, "compress": "lzw"}
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(aligned)

        logger.info(f"空间对齐: {out_path}")
        return out_path

    def list_pairs(self):
        pairs = []
        for af in sorted(self.time_a_dir.glob("*.tif")):
            b_candidates = list(self.time_b_dir.glob(f"{af.stem.replace('_A', '_B')}*.tif"))
            if b_candidates:
                bf = b_candidates[0]
                pairs.append({"name": af.stem.replace("_A", ""), "time_a": str(af),
                              "time_b": str(bf), "size_a": af.stat().st_size,
                              "size_b": bf.stat().st_size})
        return pairs
