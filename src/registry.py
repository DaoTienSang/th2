import json
import os

import mlflow
from mlflow.tracking import MlflowClient

from src.paths import md, mlr, uri


def main():
    os.environ["MLFLOW_TRACKING_URI"] = uri(mlr)
    mlflow.set_tracking_uri(uri(mlr))
    rs = json.loads((md / "runs.json").read_text(encoding="utf-8"))
    rs = sorted(rs, key=lambda r: r["val_f1"], reverse=True)
    cn = "churn_model"
    cl = MlflowClient()
    vs = []
    for r in rs:
        mv = mlflow.register_model(f"runs:/{r['run_id']}/model", cn)
        vs.append({"name": r["name"], "version": mv.version, "val_f1": r["val_f1"], "test_f1": r["test_f1"]})
    for i, v in enumerate(vs):
        st = "Production" if i == 0 else "Staging"
        try:
            cl.transition_model_version_stage(cn, v["version"], st, archive_existing_versions=False)
        except Exception:
            cl.set_registered_model_alias(cn, st.lower(), v["version"])
        v["stage"] = st
    (md / "registry.json").write_text(json.dumps(vs, indent=2), encoding="utf-8")
    print("production", vs[0]["name"], vs[0]["version"], round(vs[0]["val_f1"], 4))
    if len(vs) > 1:
        print("compare", vs[0]["name"], "better than", vs[1]["name"], "by val_f1")
    return vs


if __name__ == "__main__":
    main()
