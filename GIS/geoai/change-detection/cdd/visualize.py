"""可视化工具"""
import logging
from pathlib import Path
import numpy as np
import rasterio
from PIL import Image

logger = logging.getLogger(__name__)


class ChangeVisualizer:
    @staticmethod
    def create_comparison_image(image_a_path, image_b_path, change_map_path=None,
                                 output_path="comparison.png", mode="side_by_side",
                                 change_color=(255, 0, 0), opacity=0.5, max_size=1024):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        a = ChangeVisualizer._rgb(image_a_path, max_size)
        b = ChangeVisualizer._rgb(image_b_path, max_size)
        h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
        a, b = a[:h, :w], b[:h, :w]

        if mode == "side_by_side":
            result = np.concatenate([a, b], axis=1)
        elif mode == "overlay" and change_map_path:
            change = ChangeVisualizer._chg(change_map_path, (h, w))
            result = a.copy()
            mask = change > 0
            cc = np.array(change_color, dtype=np.float32)
            result[mask] = (result[mask].astype(np.float32) * (1 - opacity) + cc * opacity).astype(np.uint8)
        elif mode == "swipe":
            mid = w // 2
            result = np.concatenate([a[:, :mid], b[:, mid:]], axis=1)
            result[:, mid-1:mid+1] = 255
        else:
            result = a

        Image.fromarray(result).save(output_path)
        return output_path

    @staticmethod
    def create_difference_heatmap(image_a_path, image_b_path, output_path="diff.png", max_size=1024):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        a = ChangeVisualizer._float(image_a_path, max_size)
        b = ChangeVisualizer._float(image_b_path, max_size)
        h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
        diff = np.mean(np.abs(a[:h, :w].astype(np.float32) - b[:h, :w].astype(np.float32)), axis=2)
        diff /= max(diff.max(), 1e-8)
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                v = diff[i, j]
                if v < 0.25:
                    rgb[i, j] = [0, int(v / 0.25 * 255), 255]
                elif v < 0.5:
                    rgb[i, j] = [0, 255, int((1 - (v - 0.25) / 0.25) * 255)]
                elif v < 0.75:
                    rgb[i, j] = [int((v - 0.5) / 0.25 * 255), 255, 0]
                else:
                    rgb[i, j] = [255, int((1 - (v - 0.75) / 0.25) * 255), 0]
        Image.fromarray(rgb).save(output_path)
        return output_path

    @staticmethod
    def create_change_overlay(base_path, change_map_path, output_path="overlay.png",
                               color=(255, 0, 0), opacity=0.5, max_size=1024):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        base = ChangeVisualizer._rgb(base_path, max_size)
        change = ChangeVisualizer._chg(change_map_path, base.shape[:2])
        result = base.copy()
        mask = change > 0
        cc = np.array(color, dtype=np.float32)
        result[mask] = (result[mask].astype(np.float32) * (1 - opacity) + cc * opacity).astype(np.uint8)
        Image.fromarray(result).save(output_path)
        return output_path

    @staticmethod
    def _rgb(path, ms):
        with rasterio.open(path) as s:
            d = s.read(out_shape=(min(3, s.count), min(ms, s.height), min(ms, s.width)),
                       resampling=rasterio.enums.Resampling.bilinear)
        if d.shape[0] < 3: d = np.repeat(d, 3, axis=0)
        elif d.shape[0] > 3: d = d[:3]
        d = np.transpose(d, (1, 2, 0)).astype(np.float32)
        if d.max() > 1: d /= 255.0
        return (d * 255).astype(np.uint8)

    @staticmethod
    def _float(path, ms):
        with rasterio.open(path) as s:
            d = s.read(out_shape=(min(3, s.count), min(ms, s.height), min(ms, s.width)),
                       resampling=rasterio.enums.Resampling.bilinear)
        if d.shape[0] < 3: d = np.repeat(d, 3, axis=0)
        elif d.shape[0] > 3: d = d[:3]
        d = np.transpose(d, (1, 2, 0)).astype(np.float32)
        return d / 255.0 if d.max() > 1 else d

    @staticmethod
    def _chg(path, shape):
        with rasterio.open(path) as s:
            return s.read(1, out_shape=shape, resampling=rasterio.enums.Resampling.nearest).astype(np.uint8)
