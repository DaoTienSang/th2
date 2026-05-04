import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

rt = Path(__file__).resolve().parent
dd = rt / "data"
md = rt / "models"


def cn(x):
    x = str(x).strip().lower()
    x = re.sub(r"[^a-z0-9]+", "_", x)
    return x.strip("_")


def fe(df):
    df = df.copy()
    df["spend_per_tenure"] = df["total_spend"] / df["tenure"].replace(0, np.nan)
    df["call_delay_ratio"] = df["support_calls"] / (df["payment_delay"] + 1)
    df["usage_recent_score"] = df["usage_frequency"] / (df["last_interaction"] + 1)
    df["spend_usage_ratio"] = df["total_spend"] / (df["usage_frequency"] + 1)
    return df.replace([np.inf, -np.inf], np.nan)


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


if __name__ == "__main__":
    if (md / "sample_input.json").exists():
        x = json.loads((md / "sample_input.json").read_text(encoding="utf-8"))
        print(drift(prep(x)))
    else:
        print("train model first")
