"""测试 - 后端 API"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient
import numpy as np
import rasterio
from rasterio.transform import from_origin


@pytest.fixture
def client():
    from backend.main import app
    return TestClient(app)


class TestAPI:
    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        data = r.json()
        assert "name" in data

    def test_health(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_config(self, client):
        r = client.get("/api/config")
        assert r.status_code == 200
        data = r.json()
        assert "regions" in data
        assert "models" in data

    def test_list_regions(self, client):
        r = client.get("/api/data/regions")
        assert r.status_code == 200

    def test_list_models(self, client):
        r = client.get("/api/detect/models")
        assert r.status_code == 200
        assert "models" in r.json()

    def test_list_results(self, client):
        r = client.get("/api/detect/results")
        assert r.status_code == 200

    def test_list_files(self, client):
        r = client.get("/api/files")
        assert r.status_code == 200

    def test_samples_status(self, client):
        r = client.get("/api/detect/samples")
        assert r.status_code == 200
        assert "counts" in r.json()

    def test_cdd_detection_requires_model_path(self, client):
        r = client.post("/api/detect/run", json={
            "engine": "cdd",
            "image_a": "data/raw/time_a/a.tif",
            "image_b": "data/raw/time_b/b.tif",
        })
        assert r.status_code == 400
        assert "model_path" in r.json()["detail"]

    def test_geoai_detection_does_not_require_model_path(self, client, monkeypatch):
        import cdd.geoai_change

        def fake_run(**kwargs):
            return {"engine": "geoai", "mask": "data/output/fake_change.tif", "vectors": "data/output/fake_change.gpkg"}

        monkeypatch.setattr(cdd.geoai_change, "run_geoai_change_detection", fake_run)
        r = client.post("/api/detect/run", json={
            "engine": "geoai",
            "image_a": "data/raw/time_a/a.tif",
            "image_b": "data/raw/time_b/b.tif",
        })
        assert r.status_code == 200
        task = client.get(f"/api/detect/status/{r.json()['task_id']}").json()
        assert task["status"] == "completed"

    def test_output_preview_file(self, client):
        from config import OUTPUT_DIR

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / "test_preview_unit.tif"
        profile = {
            "driver": "GTiff",
            "height": 8,
            "width": 8,
            "count": 1,
            "dtype": "uint8",
            "crs": "EPSG:4326",
            "transform": from_origin(116.0, 40.0, 0.01, 0.01),
        }
        try:
            with rasterio.open(path, "w", **profile) as dst:
                data = np.zeros((8, 8), dtype=np.uint8)
                data[2:5, 2:5] = 1
                dst.write(data, 1)
            r = client.get("/api/data/preview-file", params={"path": str(path)})
            assert r.status_code == 200
            assert r.headers["content-type"] == "image/png"
            assert "x-image-bounds" in r.headers
        finally:
            if path.exists():
                path.unlink()
