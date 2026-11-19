"""What does the cohort look like? Run before modeling, and keep the output:
it's what the README's data section and the model's choices are based on."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DEMOGRAPHICS, LABS, VITALS


def audit(df: pd.DataFrame) -> dict:
    per_patient = df.groupby("patient", sort=False).agg(
        hospital=("hospital", "first"), hours=("hour", "size"), septic=("SepsisLabel", "max"))
    out = {"patients": int(len(per_patient)), "patient_hours": int(len(df)), "hospitals": {}}
    for hosp, g in per_patient.groupby("hospital", observed=True):
        rows = df[df.hospital == hosp]
        out["hospitals"][str(hosp)] = {
            "patients": int(len(g)),
            "septic_patients": int(g.septic.sum()),
            "septic_share": round(float(g.septic.mean()), 4),
            "median_stay_hours": float(g.hours.median()),
            "positive_hour_share": round(float(rows.SepsisLabel.mean()), 4),
        }
    out["missing_share"] = {c: round(float(df[c].isna().mean()), 4) for c in VITALS + LABS}
    # how often is each variable measured, per patient-day? (labs ~ a few times a day)
    out["measurements_per_day"] = {c: round(float(df[c].notna().mean() * 24), 2) for c in VITALS + LABS}
    out["demographics"] = {
        "age_median": float(df.groupby("patient").Age.first().median()),
        "male_share": round(float(df.groupby("patient").Gender.first().mean()), 4),
    }
    return out


def save(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
