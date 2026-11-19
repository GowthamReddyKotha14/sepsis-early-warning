"""The PhysioNet 2019 utility score, vectorized.

Per patient-hour the official score rewards or penalizes a warning:
  septic patient, warning 12 to 6 h before onset   -> reward rising to +1
  septic patient, warning 6 h before to 3 h after  -> reward falling to 0
  septic patient, warning too early (> 12 h before) -> small penalty (-0.05)
  septic patient, NO warning in the last 6 h before onset onward -> penalty down to -2
  non-septic patient, warning                     -> -0.05 (a false alarm)
Normalized utility: 1 = a perfect forecaster, 0 = never warning.

Key trick: each row's utility depends only on that patient's labels, so
"utility if we warn" (u_pos) and "if we don't" (u_neg) are computed once, and
any threshold is then scored with one np.where. This makes threshold sweeps
over 1.5M rows instant. Verified against the official code in tests.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DT_EARLY, DT_OPTIMAL, DT_LATE = -12, -6, 3
MAX_U_TP, MIN_U_FN, U_FP, U_TN = 1.0, -2.0, -0.05, 0.0


@dataclass
class UtilityTable:
    u_pos: np.ndarray       # utility of warning at each row
    u_neg: np.ndarray       # utility of staying silent at each row
    best: np.ndarray        # the official "optimal" prediction at each row (bool)

    @property
    def best_total(self) -> float:
        return float(np.where(self.best, self.u_pos, self.u_neg).sum())

    @property
    def inaction_total(self) -> float:
        return float(self.u_neg.sum())

    def normalized(self, warn: np.ndarray) -> float:
        observed = float(np.where(warn, self.u_pos, self.u_neg).sum())
        return (observed - self.inaction_total) / (self.best_total - self.inaction_total)


def _patient(labels: np.ndarray):
    n = len(labels)
    t = np.arange(n, dtype=float)
    u_pos = np.full(n, U_FP)
    u_neg = np.full(n, U_TN)
    best = np.zeros(n, dtype=bool)
    if labels.any():
        ts = float(np.argmax(labels) - DT_OPTIMAL)                     # onset = first label + 6
        m1 = MAX_U_TP / (DT_OPTIMAL - DT_EARLY); b1 = -m1 * DT_EARLY
        m2 = -MAX_U_TP / (DT_LATE - DT_OPTIMAL); b2 = -m2 * DT_LATE
        m3 = MIN_U_FN / (DT_LATE - DT_OPTIMAL); b3 = -m3 * DT_OPTIMAL
        rel = t - ts
        early = t <= ts + DT_OPTIMAL
        inside = t <= ts + DT_LATE
        u_pos = np.where(early, np.maximum(m1 * rel + b1, U_FP), m2 * rel + b2)
        u_neg = np.where(early, 0.0, m3 * rel + b3)
        u_pos[~inside] = 0.0                                           # after onset+3 h: not scored
        u_neg[~inside] = 0.0
        lo, hi = max(0, int(ts) + DT_EARLY), min(int(ts) + DT_LATE + 1, n)
        best[lo:hi] = True
    return u_pos, u_neg, best


def utility_table(df: pd.DataFrame) -> UtilityTable:
    """df must be sorted by patient then hour, with a SepsisLabel column."""
    parts = [_patient(g.to_numpy()) for _, g in df.groupby("patient", sort=False)["SepsisLabel"]]
    return UtilityTable(*(np.concatenate([p[i] for p in parts]) for i in range(3)))


def warning_lead_times(df: pd.DataFrame, warn: np.ndarray) -> pd.Series:
    """Hours between a septic patient's FIRST warning and sepsis onset (positive = early).
    NaN for septic patients never warned. Warnings more than 12 h early still count here."""
    d = df[["patient", "hour", "SepsisLabel"]].assign(warn=np.asarray(warn, bool))
    out = {}
    for pid, g in d.groupby("patient", sort=False):
        lab = g.SepsisLabel.to_numpy()
        if not lab.any():
            continue
        onset = int(np.argmax(lab)) - DT_OPTIMAL
        w = np.flatnonzero(g.warn.to_numpy()[: onset + DT_LATE + 1])
        out[pid] = float(onset - w[0]) if len(w) else np.nan
    return pd.Series(out, name="lead_hours")
