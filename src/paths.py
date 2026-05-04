from pathlib import Path

rt = Path(__file__).resolve().parents[1]
raw = rt / "churnDataset.csv"
dd = rt / "data"
md = rt / "models"
ld = rt / "logs"
mlr = rt / "mlruns"
for p in [dd, md, ld]:
    p.mkdir(parents=True, exist_ok=True)


def uri(p):
    return p.resolve().as_uri()
