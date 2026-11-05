"""A minimal experiment log: one row per run (settings + results), saved as CSV.

    log = ExperimentLog(env.out_dir / "cifar_runs.csv")
    log.add(name="baseline", lr=0.1, augment="none", val_acc=0.71, minutes=1.2)
    log.table(sort_by="val_acc")

Tools like Weights & Biases or MLflow do the same with dashboards, artifacts and team sharing;
the habit that matters is the same: log every run, settings and results together.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pandas as pd


class ExperimentLog:
    def __init__(self, path, reset: bool = False):
        self.path = Path(path)
        if reset and self.path.exists():
            self.path.unlink()

    def add(self, **row):
        row = {"time": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"), **row}
        df = pd.DataFrame([row])
        if self.path.exists():
            df = pd.concat([pd.read_csv(self.path), df], ignore_index=True)
        df.to_csv(self.path, index=False)
        return row

    def table(self, sort_by: str | None = None, ascending: bool = False) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame()
        df = pd.read_csv(self.path)
        return df.sort_values(sort_by, ascending=ascending) if sort_by else df
