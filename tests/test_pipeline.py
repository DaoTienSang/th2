from pathlib import Path

import pandas as pd

from src.data_pipeline import run
from src.paths import dd


def test_data_pipeline():
    tb = run()
    assert "churn" in tb.columns
    assert tb.isna().sum().sum() == 0
    assert (dd / "processed_churn.csv").exists()
    assert (dd / "train_stats.json").exists()
    assert len(pd.read_csv(dd / "processed_churn.csv")) > 100
