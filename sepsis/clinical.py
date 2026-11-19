"""Rule-based screening scores used in hospitals today: the baseline any model must beat.

Both use only information available at that hour: each measurement is
carried forward from its last recorded value, never back from the future.

SIRS (Systemic Inflammatory Response Syndrome) criteria, alert at >= 2 of:
  heart rate > 90, respiratory rate > 20 (or PaCO2 < 32 mmHg),
  temperature > 38 or < 36 C, white cell count > 12 or < 4 (x10^3/uL)
qSOFA, adapted: the full score also needs altered mental status, which this
dataset doesn't record, so this version alerts when BOTH respiratory rate >= 22
AND systolic BP <= 100.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def carry_forward(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Last observed value per patient (causal: never fills from the future)."""
    return df.groupby("patient", sort=False)[cols].ffill()


def sirs_count(df: pd.DataFrame) -> np.ndarray:
    v = carry_forward(df, ["HR", "Resp", "PaCO2", "Temp", "WBC"])
    criteria = [
        v.HR > 90,
        (v.Resp > 20) | (v.PaCO2 < 32),
        (v.Temp > 38) | (v.Temp < 36),
        (v.WBC > 12) | (v.WBC < 4),
    ]
    return np.sum([c.to_numpy() for c in criteria], axis=0)       # NaN comparisons are False


def sirs_alert(df: pd.DataFrame, threshold: int = 2) -> np.ndarray:
    return sirs_count(df) >= threshold


def qsofa_alert(df: pd.DataFrame) -> np.ndarray:
    v = carry_forward(df, ["Resp", "SBP"])
    return ((v.Resp >= 22) & (v.SBP <= 100)).to_numpy()
