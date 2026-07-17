from fastapi.testclient import TestClient

from backend.main import app


def test_health_and_models():
    client = TestClient(app)
    assert client.get("/api/health").status_code == 200
    payload = client.get("/api/models").json()
    assert {item["name"] for item in payload["models"]} == {"unet", "no_skip", "unetpp", "deeplabv3plus"}
