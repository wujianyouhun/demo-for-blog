from fastapi.testclient import TestClient

from backend.main import app


def test_health_and_download_path_guard():
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert client.get("/api/download", params={"path": "C:/Windows/win.ini"}).status_code == 404
