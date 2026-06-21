"""测试 - 后端 API"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient


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
