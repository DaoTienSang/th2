import json
import re

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer

from src.paths import dd, raw


def cn(x):
    x = str(x).strip().lower()
    x = re.sub(r"[^a-z0-9]+", "_", x)
    return x.strip("_")


def cl(df):
    df = df.copy()
    df.columns = [cn(c) for c in df.columns]
    df = df.drop_duplicates()
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan})
    if df["churn"].dtype == "object":
        df["churn"] = df["churn"].str.lower().map({"yes": 1, "no": 0, "true": 1, "false": 0}).fillna(df["churn"])
    df["churn"] = pd.to_numeric(df["churn"], errors="coerce")
    return df


def fe(df):
    df = df.copy()
    df["spend_per_tenure"] = df["total_spend"] / df["tenure"].replace(0, np.nan)
    df["call_delay_ratio"] = df["support_calls"] / (df["payment_delay"] + 1)
    df["usage_recent_score"] = df["usage_frequency"] / (df["last_interaction"] + 1)
    df["spend_usage_ratio"] = df["total_spend"] / (df["usage_frequency"] + 1)
    return df.replace([np.inf, -np.inf], np.nan)


def miss_a(df):
    df = df.copy()
    ns = df.select_dtypes(include=np.number).columns
    cs = [c for c in df.columns if c not in ns]
    for c in ns:
        df[c] = df[c].fillna(df[c].median())
    for c in cs:
        m = df[c].mode(dropna=True)
        df[c] = df[c].fillna(m.iloc[0] if len(m) else "unknown")
    return df


def miss_b(df):
    df = df.copy()
    ns = list(df.select_dtypes(include=np.number).columns)
    cs = [c for c in df.columns if c not in ns]
    if ns:
        ki = KNNImputer(n_neighbors=5)
        df[ns] = ki.fit_transform(df[ns])
    if cs:
        si = SimpleImputer(strategy="most_frequent")
        df[cs] = si.fit_transform(df[cs])
    return df


def out(df):
    df = df.copy()
    ns = [c for c in df.select_dtypes(include=np.number).columns if c not in ["churn", "customerid", "customer_id"]]
    mk = pd.Series(True, index=df.index)
    for c in ns:
        q1 = df[c].quantile(0.25)
        q3 = df[c].quantile(0.75)
        iq = q3 - q1
        if iq > 0:
            mk &= df[c].between(q1 - 1.5 * iq, q3 + 1.5 * iq)
    rs = df.loc[mk].reset_index(drop=True)
    return rs if len(rs) else df.reset_index(drop=True)


def cols(df):
    fs = [c for c in df.columns if c not in ["churn", "customerid", "customer_id"]]
    ns = [c for c in fs if pd.api.types.is_numeric_dtype(df[c])]
    cs = [c for c in fs if c not in ns]
    return fs, ns, cs


def run(src=raw, method="knn"):
    df = pd.read_csv(src)
    df = fe(cl(df))
    a = out(miss_a(df))
    b = out(miss_b(df))
    a.to_csv(dd / "processed_median_mode.csv", index=False)
    b.to_csv(dd / "processed_knn_mode.csv", index=False)
    tb = b if method == "knn" else a
    tb["churn"] = tb["churn"].round().astype(int)
    fs, ns, cs = cols(tb)
    st = {c: {"mean": float(tb[c].mean()), "std": float(tb[c].std() or 1.0)} for c in ns}
    (dd / "processed_churn.csv").write_text(tb.to_csv(index=False), encoding="utf-8")
    (dd / "columns.json").write_text(json.dumps({"features": fs, "num": ns, "cat": cs}, indent=2), encoding="utf-8")
    (dd / "train_stats.json").write_text(json.dumps(st, indent=2), encoding="utf-8")
    print("data ok", tb.shape[0], tb.shape[1])
    return tb


def main():
    run()


if __name__ == "__main__":
    main()
