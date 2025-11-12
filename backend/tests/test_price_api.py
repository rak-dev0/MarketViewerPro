from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_price_requires_ticker():
    resp = client.get("/api/price")
    assert resp.status_code == 422  # missing param


def test_price_aapl_basic():
    resp = client.get("/api/price", params={"ticker": "AAPL", "range": "1mo", "interval": "1d"})
    # This can fail if upstream is down; keep assertions conservative.
    assert resp.status_code in (200, 502)
    if resp.status_code == 200:
        data = resp.json()
        assert data["ticker"] == "AAPL"
        assert isinstance(data["candles"], list)
        assert "source" in data
