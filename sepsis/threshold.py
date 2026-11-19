"""Pick the alarm threshold on VALIDATION patients only.

A risk score becomes an alarm once it passes a threshold. Lower = more septic
patients caught but more alarms; higher = quieter but more misses. We pick the
threshold with the best challenge utility among those that keep the alarm load
within a budget (alerts per bed per 12-hour shift).

Default budget 0.30: the load the adapted qSOFA rule put on hospital A's
validation patients (results/baseline.json). So the question becomes:
"for the same number of alarms nurses already get, how much more does the
model catch?" The best threshold with no budget is reported too.

Note: the alert COUNT is not monotonic in the threshold. An alert is a
warning switching on, and lowering the threshold can join two separate
warnings into one. The share of hours spent in warning always rises.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import UtilityTable, utility_table, warning_lead_times

DEFAULT_BUDGET = 0.30
SHIFT_HOURS = 12


class Sweeper:
    """Scores many thresholds fast on one set of patient-hours."""

    def __init__(self, df: pd.DataFrame, risk: np.ndarray):
        self.df = df
        self.risk = np.asarray(risk)
        self.ut: UtilityTable = utility_table(df)
        self.first = (df.groupby("patient", sort=False)["hour"].transform("min") == df["hour"]).to_numpy()
        self.septic = df.groupby("patient", sort=False)["SepsisLabel"].transform("max").to_numpy().astype(bool)
        self.shifts = len(df) / SHIFT_HOURS

    def score(self, thr: float) -> dict:
        w = self.risk >= thr
        rising = w & (self.first | ~np.r_[False, w[:-1]])
        n = int(rising.sum())
        return {"threshold": float(thr), "utility": self.ut.normalized(w),
                "alerts_per_bed_shift": n / self.shifts, "warning_hour_share": float(w.mean()),
                "septic_alert_share": float(rising[self.septic].sum() / n) if n else np.nan}

    def sweep(self, n: int = 200) -> pd.DataFrame:
        qs = np.unique(np.quantile(self.risk, np.linspace(0.5, 0.9995, n)))
        return pd.DataFrame([self.score(t) for t in qs])


def choose(sweep: pd.DataFrame, budget: float | None = DEFAULT_BUDGET) -> pd.Series:
    """Best utility within the alarm budget (or overall when budget is None)."""
    ok = sweep if budget is None else sweep[sweep.alerts_per_bed_shift <= budget]
    if ok.empty:                                    # nothing fits: take the quietest setting
        return sweep.loc[sweep.alerts_per_bed_shift.idxmin()]
    return ok.loc[ok.utility.idxmax()]


def septic_caught(df: pd.DataFrame, warn: np.ndarray) -> tuple[int, int]:
    """(septic patients warned no later than 3 h after onset, all septic patients)."""
    lead = warning_lead_times(df, warn)
    return int(lead.notna().sum()), int(len(lead))
