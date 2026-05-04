import json

from fastapi.testclient import TestClient

from src.paths import md
from src.serve import app


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
