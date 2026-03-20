from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"

def test_home():
    r = client.get("/")
    assert r.status_code == 200

def test_predict_works():
    payload = {"age":25,"bmi":22.0,"experience_years":3.0,"weight_lifted_kg":60.0,"sets":3,"reps":10,"rest_days":2,"sleep_hours":7.0,"warm_up_done":1,"past_injury":0}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert "risk_level" in r.json()

def test_invalid_input():
    r = client.post("/predict", json={"age": 5})
    assert r.status_code == 422