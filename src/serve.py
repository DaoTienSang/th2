import os

import joblib
import mlflow
import mlflow.sklearn
from fastapi import FastAPI

from src.monitor import drift, log_req, prep
from src.paths import md, mlr, uri

os.environ["MLFLOW_TRACKING_URI"] = uri(mlr)
mlflow.set_tracking_uri(uri(mlr))
app = FastAPI(title="churn api")
mm = None


def load():
    global mm
    if mm is not None:
        return mm
    try:
        mm = mlflow.sklearn.load_model("models:/churn_model/Production")
    except Exception:
        mm = joblib.load(md / "best_model.pkl")
    return mm


@app.get("/")
def home():
    return {"status": "ok"}


@app.post("/predict")
def predict(x: dict):
    m = load()
    tb = prep(x)
    pr = int(m.predict(tb)[0])
    if hasattr(m, "predict_proba"):
        pb = float(m.predict_proba(tb)[0][1])
    else:
        pb = float(pr)
    dr = drift(tb)
    log_req(x, pr, pb, dr)
    return {"label": pr, "probability": pb, "drift": bool(dr), "drift_features": dr}
