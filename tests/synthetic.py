"""Tiny synthetic patients in the exact PhysioNet .psv format, for tests only.

Real results always come from the real PhysioNet data. These exist so the
pipeline can be tested in CI without downloading 40,336 patients. Septic
patients drift (heart rate and lactate up, blood pressure down) in the hours
before onset, so a model has a learnable signal.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sepsis.data import COLUMNS, LABS


def make_patient(rng: np.random.Generator, septic: bool, hours: int) -> pd.DataFrame:
    t = np.arange(hours)
    onset = int(rng.integers(12, hours - 3)) if septic else None
    drift = np.zeros(hours)
    if septic:
        drift = np.clip((t - (onset - 10)) / 10, 0, 1.5)       # deterioration starts ~10 h before onset
    df = pd.DataFrame(np.nan, index=t, columns=COLUMNS)
    df["HR"] = 85 + 25 * drift + rng.normal(0, 6, hours)
    df["MAP"] = 82 - 18 * drift + rng.normal(0, 5, hours)
    df["SBP"] = df["MAP"] + 40 + rng.normal(0, 4, hours)
    df["DBP"] = df["MAP"] - 20 + rng.normal(0, 3, hours)
    df["Resp"] = 17 + 7 * drift + rng.normal(0, 2, hours)
    df["Temp"] = 37 + 1.2 * drift + rng.normal(0, 0.3, hours)
    df["O2Sat"] = 97 - 4 * drift + rng.normal(0, 1, hours)
    for col in ("HR", "MAP", "SBP", "DBP", "Resp", "O2Sat"):
        df.loc[rng.random(hours) < 0.1, col] = np.nan             # ~10% missing vitals
    df.loc[rng.random(hours) < 0.75, "Temp"] = np.nan             # temperature charted rarely
    lab_hours = rng.random(hours) < 0.08                          # labs drawn every ~12 h
    df.loc[lab_hours, "Lactate"] = (1.2 + 2.5 * drift + rng.normal(0, 0.3, hours))[lab_hours]
    df.loc[lab_hours, "WBC"] = (9 + 6 * drift + rng.normal(0, 2, hours))[lab_hours]
    df.loc[lab_hours, "Creatinine"] = (1.0 + 0.8 * drift + rng.normal(0, 0.2, hours))[lab_hours]
    df["Age"] = float(rng.integers(20, 90))
    df["Gender"] = float(rng.integers(0, 2))
    df["Unit1"], df["Unit2"] = (1.0, 0.0) if rng.random() < 0.5 else (0.0, 1.0)
    df["HospAdmTime"] = -float(rng.integers(0, 48))
    df["ICULOS"] = t + 1.0
    df["SepsisLabel"] = (t >= onset - 6).astype(int) if septic else 0
    assert set(LABS) <= set(df.columns)
    return df


def write_dataset(root: Path, n_per_hospital: int = 60, septic_share: float = 0.3, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    pid = 0
    for hosp in ("A", "B"):
        d = root / f"training_set{hosp}"
        d.mkdir(parents=True, exist_ok=True)
        for _ in range(n_per_hospital):
            pid += 1
            df = make_patient(rng, rng.random() < septic_share, int(rng.integers(24, 60)))
            df.to_csv(d / f"p{pid:06d}.psv", sep="|", index=False, na_rep="NaN")
    return root
