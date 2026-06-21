"""测试 - 推理引擎"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import rasterio
from rasterio.transform import from_bounds


@pytest.fixture
def sample_image(tmp_path):
    """创建临时测试影像"""
    h, w = 64, 64
    transform = from_bounds(116.3, 39.8, 116.4, 39.9, w, h)
    profile = {
        "driver": "GTiff", "dtype": "uint8",
        "width": w, "height": h, "count": 3,
        "crs": "EPSG:4326", "transform": transform,
    }
    rng = np.random.RandomState(0)
    data = rng.randint(50, 200, (3, h, w), dtype=np.uint8)
    p = tmp_path / "test_image.tif"
    with rasterio.open(p, "w", **profile) as dst:
        dst.write(data)
    return str(p)


class TestInferenceEngine:
    def test_predict(self, sample_image, tmp_path):
        from geoai_core.models import build_model
        from geoai_core.inference import InferenceEngine
        model = build_model("deeplabv3p_resnet50")
        model.eval()
        engine = InferenceEngine(model=model, device="cpu", tile_size=64, overlap=0, batch_size=1)
        out = tmp_path / "prediction.tif"
        result = engine.predict(sample_image, str(out), threshold=0.5, smoothing_sigma=0)
        assert Path(result).exists()
        with rasterio.open(result) as src:
            assert src.count >= 1
            label = src.read(1)
            assert label.shape == (64, 64)
