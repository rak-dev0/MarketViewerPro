from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def test_health_ok():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("up", "degraded")
    assert isinstance(data["providers"], list)
    assert any(p["name"] == "api" for p in data["providers"])
