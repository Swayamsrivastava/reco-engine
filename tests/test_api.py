import numpy as np
from fastapi.testclient import TestClient
from serving.api import app

client = TestClient(app)

def test_health():
    assert client.get("/health").json()["status"] == "ok"

def test_known_user_gets_full_pipeline():
    uid = int(np.load("retrieval/user_ids.npy")[0])
    r = client.get(f"/recommend?user_id={uid}&k=10").json()
    assert r["source"] in ("two_stage", "cache") and len(r["items"]) == 10

def test_cold_start_routes():
    assert client.get("/recommend?user_id=999999991&k=10").json()["source"] == "cold_popular"
    r = client.get("/recommend?user_id=999999992&k=10&liked=1,260,296").json()
    assert r["source"] == "cold_content" and len(r["items"]) == 10