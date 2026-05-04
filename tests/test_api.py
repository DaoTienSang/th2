import json
from pathlib import Path

from fastapi.testclient import TestClient

from app import app

md = Path(__file__).resolve().parents[1] / "models"


def test_predict():
    if not (md / "best_model.pkl").exists():
        return
    x = json.loads((md / "sample_input.json").read_text(encoding="utf-8"))
    cl = TestClient(app)
    r = cl.post("/predict", json=x)
    assert r.status_code == 200
    js = r.json()
    assert "label" in js
    assert "probability" in js
