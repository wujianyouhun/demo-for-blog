"""
TIF 瓦片生成与服务模块
=======================
将大型 GeoTIFF 切片为 XYZ 瓦片，供 OpenLayers 前端显示。
支持实时窗口读取和预生成缓存两种模式。
"""

import os
import math
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds
from PIL import Image


class TIFTileServer:
    """GeoTIFF → XYZ 瓦片服务"""

    def __init__(self, tif_path: str, cache_dir: str = None):
        self.tif_path = tif_path
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(tif_path), "tiles"
        )
        os.makedirs(self.cache_dir, exist_ok=True)

        # 读取元数据
        with rasterio.open(tif_path) as ds:
            self.width = ds.width
            self.height = ds.height
            self.crs = str(ds.crs)
            self.bounds = ds.bounds
            self.transform = ds.transform
            self.res = ds.res
            self.band_count = ds.count

        # 计算可用缩放级别
        self.max_zoom = self._calc_max_zoom()
        self.min_zoom = 0

    def get_info(self) -> Dict[str, Any]:
        """获取 TIF 元数据信息"""
        return {
            "width": self.width,
            "height": self.height,
            "crs": self.crs,
            "bounds": {
                "left": self.bounds.left,
                "bottom": self.bounds.bottom,
                "right": self.bounds.right,
                "top": self.bounds.top,
            },
            "resolution": {
                "x": self.res[0],
                "y": self.res[1],
                "approx_meters": round(self.res[0] * 111000, 2),
            },
            "band_count": self.band_count,
            "max_zoom": self.max_zoom,
            "pixel_count_millions": round(self.width * self.height / 1e6, 1),
        }

    def get_tile(self, z: int, x: int, y: int) -> Optional[bytes]:
        """获取指定瓦片的 PNG 数据"""
        # 检查缓存
        cache_path = self._cache_path(z, x, y)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return f.read()

        # 实时生成
        tile_data = self._render_tile(z, x, y)
        if tile_data is not None:
            # 写入缓存
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                f.write(tile_data)
        return tile_data

    def generate_tiles(self, zoom_levels: list = None,
                       progress_callback=None) -> int:
        """预生成指定缩放级别的瓦片"""
        if zoom_levels is None:
            zoom_levels = list(range(
                max(0, self.max_zoom - 4), self.max_zoom + 1
            ))

        total = 0
        for z in zoom_levels:
            tiles = self._tiles_at_zoom(z)
            for i, (x, y) in enumerate(tiles):
                tile_data = self._render_tile(z, x, y)
                if tile_data:
                    cache_path = self._cache_path(z, x, y)
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    with open(cache_path, "wb") as f:
                        f.write(tile_data)
                    total += 1
                if progress_callback:
                    progress_callback(z, i + 1, len(tiles))
        return total

    def pixel_to_latlon(self, px: int, py: int) -> Tuple[float, float]:
        """像素坐标转经纬度"""
        geo_x = self.bounds.left + px * self.res[0]
        geo_y = self.bounds.top - py * self.res[1]
        return geo_y, geo_x  # lat, lon

    def latlon_to_pixel(self, lat: float, lon: float) -> Tuple[int, int]:
        """经纬度转像素坐标"""
        px = int((lon - self.bounds.left) / self.res[0])
        py = int((self.bounds.top - lat) / self.res[1])
        return px, py

    def read_region(self, bounds_latlon: dict) -> Optional[np.ndarray]:
        """读取指定经纬度范围的影像数据

        Args:
            bounds_latlon: {"left": lon, "bottom": lat, "right": lon, "top": lat}
        """
        left_px, top_px = self.latlon_to_pixel(bounds_latlon["top"], bounds_latlon["left"])
        right_px, bottom_px = self.latlon_to_pixel(
            bounds_latlon["bottom"], bounds_latlon["right"]
        )

        col_off = max(0, min(left_px, right_px))
        row_off = max(0, min(top_px, bottom_px))
        w = abs(right_px - left_px)
        h = abs(bottom_px - top_px)

        if w < 1 or h < 1:
            return None

        # 限制范围
        col_off = min(col_off, self.width - 1)
        row_off = min(row_off, self.height - 1)
        w = min(w, self.width - col_off)
        h = min(h, self.height - row_off)

        window = Window(col_off, row_off, w, h)
        with rasterio.open(self.tif_path) as ds:
            return ds.read(window=window)

    # ── 内部方法 ──

    def _calc_max_zoom(self) -> int:
        """根据分辨率计算最大缩放级别"""
        # 瓦片分辨率 = 地球周长 / (256 * 2^zoom)
        meters_per_pixel = self.res[0] * 111000  # 近似
        for z in range(22, 0, -1):
            tile_res = 40075016.686 / (256 * (2 ** z))
            if tile_res <= meters_per_pixel:
                return min(z + 2, 20)
        return 14

    def _tile_bounds(self, z: int, x: int, y: int) -> Tuple[float, float, float, float]:
        """计算瓦片的经纬度范围 (left, bottom, right, top)"""
        n = 2 ** z
        left = x / n * 360.0 - 180.0
        right = (x + 1) / n * 360.0 - 180.0
        top = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
        bottom = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
        return left, bottom, right, top

    def _render_tile(self, z: int, x: int, y: int) -> Optional[bytes]:
        """从 TIF 中读取并渲染瓦片"""
        tile_left, tile_bottom, tile_right, tile_top = self._tile_bounds(z, x, y)

        # 检查瓦片是否与 TIF 范围相交
        if (tile_right < self.bounds.left or tile_left > self.bounds.right or
                tile_top < self.bounds.bottom or tile_bottom > self.bounds.top):
            return None

        # 计算在 TIF 中的像素范围
        col_start = max(0, int((tile_left - self.bounds.left) / self.res[0]))
        row_start = max(0, int((self.bounds.top - tile_top) / self.res[1]))
        col_end = min(self.width, int((tile_right - self.bounds.left) / self.res[0]))
        row_end = min(self.height, int((self.bounds.top - tile_bottom) / self.res[1]))

        w = col_end - col_start
        h = row_end - row_start
        if w < 1 or h < 1:
            return None

        try:
            window = Window(col_start, row_start, w, h)
            with rasterio.open(self.tif_path) as ds:
                data = ds.read(window=window)

            # (bands, h, w) → (h, w, bands) → uint8
            img = np.transpose(data[:3], (1, 2, 0))
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)

            # 缩放到 256x256 瓦片
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize((256, 256), Image.Resampling.BILINEAR)

            import io
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            return buf.getvalue()

        except Exception as e:
            print(f"[Tile] z={z} x={x} y={y} 渲染失败: {e}")
            return None

    def _tiles_at_zoom(self, z: int) -> list:
        """计算指定缩放级别下与 TIF 范围重叠的所有瓦片"""
        tiles = []
        n = 2 ** z
        # 经度 → x
        x_min = int((self.bounds.left + 180) / 360 * n)
        x_max = int((self.bounds.right + 180) / 360 * n)
        # 纬度 → y
        y_min = int((1 - math.asinh(math.tan(math.radians(self.bounds.top))) / math.pi) / 2 * n)
        y_max = int((1 - math.asinh(math.tan(math.radians(self.bounds.bottom))) / math.pi) / 2 * n)

        for x in range(max(0, x_min), min(n, x_max + 1)):
            for y in range(max(0, y_min), min(n, y_max + 1)):
                tiles.append((x, y))
        return tiles

    def _cache_path(self, z: int, x: int, y: int) -> str:
        return os.path.join(self.cache_dir, str(z), str(x), f"{y}.png")
