"""推理引擎"""
import logging
from pathlib import Path
from typing import Optional, Dict
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes as rasterio_shapes
from scipy.ndimage import gaussian_filter
from shapely.geometry import shape as shapely_shape
import geopandas as gpd
import torch
import torch.nn as nn
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ChangeDetector:
    def __init__(self, model, device="auto", tile_size=256, overlap=32, batch_size=4):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.tile_size, self.overlap, self.batch_size = tile_size, overlap, batch_size

    @torch.no_grad()
    def detect(self, image_a_path, image_b_path, output_path, threshold=0.5, smoothing_sigma=1.0):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(image_a_path) as sa:
            w, h, crs, tf = sa.width, sa.height, sa.crs, sa.transform

        stride = self.tile_size - self.overlap
        windows = sorted(set(
            (min(y, max(0, h - self.tile_size)), min(x, max(0, w - self.tile_size)))
            for y in range(0, max(1, h - self.tile_size + 1), stride)
            for x in range(0, max(1, w - self.tile_size + 1), stride)
        ))

        prob_sum = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)
        batch_a, batch_b, batch_pos = [], [], []

        with rasterio.open(image_a_path) as sa, rasterio.open(image_b_path) as sb:
            for i, (y, x) in enumerate(tqdm(windows, desc="推理")):
                win = Window(x, y, self.tile_size, self.tile_size)
                da = self._ensure3(sa.read(window=win).astype(np.float32))
                db = self._ensure3(sb.read(window=win).astype(np.float32))
                if da.max() > 1: da /= 255.0
                if db.max() > 1: db /= 255.0
                batch_a.append(da)
                batch_b.append(db)
                batch_pos.append((y, x))

                if len(batch_a) >= self.batch_size or i == len(windows) - 1:
                    ta = torch.from_numpy(np.stack(batch_a)).float().to(self.device)
                    tb = torch.from_numpy(np.stack(batch_b)).float().to(self.device)
                    probs = torch.softmax(self.model(ta, tb)["out"], dim=1)[:, 1].cpu().numpy()
                    for j, (py, px) in enumerate(batch_pos):
                        ph, pw = probs[j].shape
                        prob_sum[py:py+ph, px:px+pw] += probs[j]
                        count_map[py:py+ph, px:px+pw] += 1.0
                    batch_a.clear(); batch_b.clear(); batch_pos.clear()

        prob_avg = prob_sum / np.maximum(count_map, 1.0)
        if smoothing_sigma > 0:
            prob_avg = gaussian_filter(prob_avg, sigma=smoothing_sigma)
        change_map = (prob_avg > threshold).astype(np.uint8)

        profile = {"driver": "GTiff", "dtype": "float32", "width": w, "height": h,
                    "count": 2, "crs": crs, "transform": tf, "compress": "lzw"}
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(change_map.astype(np.float32), 1)
            dst.write(prob_avg, 2)

        logger.info(f"检测完成: {np.sum(change_map)} 变化像素")
        return output_path

    def vectorize(self, change_map_path, output_path, min_area_pixels=30):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(change_map_path) as src:
            label, tf, crs = src.read(1).astype(np.uint8), src.transform, src.crs

        feats = []
        for geom, val in rasterio_shapes(label, mask=label == 1, transform=tf):
            if val != 1: continue
            poly = shapely_shape(geom)
            if poly.is_empty or not poly.is_valid or poly.area < min_area_pixels: continue
            feats.append({"change_type": "changed", "area_px": poly.area, "geometry": poly})

        gdf = gpd.GeoDataFrame(feats if feats else [], crs=crs,
                                columns=["change_type", "area_px", "geometry"])
        drv = "GeoJSON" if str(output_path).endswith(".geojson") else "GPKG"
        gdf.to_file(output_path, driver=drv)
        logger.info(f"矢量化: {len(gdf)} 个区域")
        return output_path

    def detect_and_vectorize(self, image_a, image_b, output_dir, threshold=0.5,
                              smoothing_sigma=1.0, min_area_pixels=30):
        output_dir = Path(output_dir)
        name = f"{Path(image_a).stem}_vs_{Path(image_b).stem}"
        mask = output_dir / f"{name}_change.tif"
        vec = output_dir / f"{name}_change.gpkg"
        self.detect(image_a, image_b, mask, threshold, smoothing_sigma)
        self.vectorize(mask, vec, min_area_pixels)
        return {"mask": mask, "vectors": vec}

    @staticmethod
    def _ensure3(d):
        return np.repeat(d, 3, axis=0)[:3] if d.shape[0] < 3 else d[:3]
