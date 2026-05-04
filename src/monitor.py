import json
from datetime import datetime

import pandas as pd

from src.data_pipeline import cn, fe
from src.paths import dd, ld


def prep(x):
    df = pd.DataFrame([x])
    df.columns = [cn(c) for c in df.columns]
    df = fe(df)
    cm = json.loads((dd / "columns.json").read_text(encoding="utf-8"))
    for c in cm["features"]:
        if c not in df.columns:
            df[c] = 0
    return df[cm["features"]]


def drift(df, th=3.0):
    st = json.loads((dd / "train_stats.json").read_text(encoding="utf-8"))
    rs = {}
    for c, v in st.items():
        if c in df.columns:
            sd = v["std"] if v["std"] else 1.0
            z = abs(float(df[c].mean()) - v["mean"]) / sd
            if z > th:
                rs[c] = round(z, 4)
    return rs


def log_req(x, y, p, dr):
    ld.mkdir(parents=True, exist_ok=True)
    fp = ld / "predictions.jsonl"
    rc = {"time": datetime.utcnow().isoformat(), "input": x, "label": int(y), "probability": float(p), "drift": dr}
    with fp.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rc, ensure_ascii=False) + "\n")
    return rc
