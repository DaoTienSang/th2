import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_pipeline import run as run_data

rt = Path(__file__).resolve().parent
dd = rt / "data"
md = rt / "models"
mlr = rt / "mlruns"
md.mkdir(exist_ok=True)


def oh():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def ev(m, x, y):
    pr = m.predict(x)
    return {"accuracy": float(accuracy_score(y, pr)), "f1": float(f1_score(y, pr))}


def main(fast=False):
    mlflow.set_tracking_uri(mlr.resolve().as_uri())
    mlflow.set_experiment("churn")
    fp = dd / "processed_churn.csv"
    if not fp.exists():
        run_data()
    df = pd.read_csv(fp)
    if fast and len(df) > 12000:
        df = df.sample(12000, random_state=42).reset_index(drop=True)
    cm = json.loads((dd / "columns.json").read_text(encoding="utf-8"))
    fs, ns, cs = cm["features"], cm["num"], cm["cat"]
    x = df[fs]
    y = df["churn"].astype(int)
    xtr, xt, ytr, yt = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)
    xv, xte, yv, yte = train_test_split(xt, yt, test_size=0.5, stratify=yt, random_state=42)
    ms = {
        "lr": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "rf": RandomForestClassifier(n_estimators=80 if fast else 150, random_state=42, class_weight="balanced", n_jobs=-1),
    }
    rs = []
    for nm, al in ms.items():
        with mlflow.start_run(run_name=nm) as rn:
            pp = ColumnTransformer([("n", StandardScaler(), ns), ("c", oh(), cs)])
            m = Pipeline([("pp", pp), ("md", al)])
            m.fit(xtr, ytr)
            va = ev(m, xv, yv)
            te = ev(m, xte, yte)
            pm = al.get_params()
            mlflow.log_params({k: v for k, v in pm.items() if isinstance(v, (str, int, float, bool, type(None)))})
            mlflow.log_metric("val_accuracy", va["accuracy"])
            mlflow.log_metric("val_f1", va["f1"])
            mlflow.log_metric("test_accuracy", te["accuracy"])
            mlflow.log_metric("test_f1", te["f1"])
            mlflow.sklearn.log_model(m, "model")
            rs.append({"name": nm, "run_id": rn.info.run_id, "val_accuracy": va["accuracy"], "val_f1": va["f1"], "test_accuracy": te["accuracy"], "test_f1": te["f1"]})
            print(nm, round(va["accuracy"], 4), round(va["f1"], 4), round(te["accuracy"], 4), round(te["f1"], 4))
    best = max(rs, key=lambda r: r["val_f1"])
    bm = mlflow.sklearn.load_model(f"runs:/{best['run_id']}/model")
    joblib.dump(bm, md / "best_model.pkl")
    (md / "runs.json").write_text(json.dumps(rs, indent=2), encoding="utf-8")
    (md / "best_run.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    (md / "sample_input.json").write_text(json.dumps(x.iloc[0].to_dict(), indent=2), encoding="utf-8")
    print("best", best["name"], round(best["val_f1"], 4))
    return rs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ar = ap.parse_args()
    main(ar.fast)
