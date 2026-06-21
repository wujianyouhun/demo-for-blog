"""测试 - 推理引擎"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import rasterio
import torch
from rasterio.transform import from_bounds


@pytest.fixture
def sample_images(tmp_path):
    """创建临时双时相影像"""
    h, w = 64, 64
    transform = from_bounds(116.3, 39.8, 116.4, 39.9, w, h)
    profile = {
        "driver": "GTiff", "dtype": "uint8",
        "width": w, "height": h, "count": 3,
        "crs": "EPSG:4326", "transform": transform,
    }

    rng = np.random.RandomState(0)
    a = rng.randint(50, 200, (3, h, w), dtype=np.uint8)
    b = a.copy()
    b[:, 10:30, 10:30] = 255  # 变化区域

    pa = tmp_path / "time_a.tif"
    pb = tmp_path / "time_b.tif"

    with rasterio.open(pa, "w", **profile) as dst:
        dst.write(a)
    with rasterio.open(pb, "w", **profile) as dst:
        dst.write(b)

    return str(pa), str(pb)


class TestChangeDetector:
    def test_detect(self, sample_images, tmp_path):
        from cdd.models import build_model
        from cdd.inference import ChangeDetector

        model = build_model("siamese_unet", in_channels=3)
        model.eval()

        detector = ChangeDetector(model=model, device="cpu", tile_size=64, overlap=0, batch_size=1)
        pa, pb = sample_images
        out = tmp_path / "change.tif"

        result = detector.detect(pa, pb, out, threshold=0.5, smoothing_sigma=0)
        assert result.exists()

        with rasterio.open(result) as src:
            assert src.count == 2  # label + prob
            label = src.read(1)
            assert label.shape == (64, 64)
            assert set(np.unique(label)).issubset({0, 1})

    def test_vectorize(self, sample_images, tmp_path):
        from cdd.models import build_model
        from cdd.inference import ChangeDetector

        model = build_model("siamese_unet", in_channels=3)
        model.eval()

        detector = ChangeDetector(model=model, device="cpu", tile_size=64, overlap=0, batch_size=1)
        pa, pb = sample_images

        mask_out = tmp_path / "change.tif"
        detector.detect(pa, pb, mask_out, smoothing_sigma=0)

        vec_out = tmp_path / "change.gpkg"
        detector.vectorize(mask_out, vec_out, min_area_pixels=5)
        assert vec_out.exists()
