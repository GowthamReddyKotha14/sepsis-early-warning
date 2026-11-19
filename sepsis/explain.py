"""Why did the model alert on this patient, at this hour?

Method (group occlusion): the model's ~90 features come from ~40 measurements
(heart rate -> HR_last, HR_age, HR_mean6h, HR_delta6h). For one alert, swap ONE
measurement's features for typical values from patients who were not septic,
re-score, and see how far the risk falls. The measurements whose removal drops
the risk most are the reasons, reported in plain words with the patient's
actual values.

Limits, stated plainly: it measures one measurement at a time, so two signals
that only matter together are under-credited; and "typical" means the median
over non-septic training hours. It explains the MODEL, not the patient's
biology: a reason is "what the model reacted to", not a diagnosis.

Only measurements the patient actually has are offered as reasons, so an
explanation never reads "lactate (never measured)".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import model as model_mod
from .data import DEMOGRAPHICS, LABS, VITALS

NAMES = {"HR": "heart rate", "O2Sat": "oxygen saturation", "Temp": "temperature", "SBP": "systolic BP",
         "MAP": "mean arterial pressure", "DBP": "diastolic BP", "Resp": "respiratory rate", "EtCO2": "end-tidal CO2",
         "BaseExcess": "base excess", "HCO3": "bicarbonate", "FiO2": "inspired oxygen (FiO2)", "pH": "blood pH",
         "PaCO2": "PaCO2", "SaO2": "arterial O2 saturation", "AST": "AST (liver)", "BUN": "BUN (kidney)",
         "Alkalinephos": "alkaline phosphatase", "Calcium": "calcium", "Chloride": "chloride",
         "Creatinine": "creatinine (kidney)", "Bilirubin_direct": "direct bilirubin", "Glucose": "glucose",
         "Lactate": "lactate", "Magnesium": "magnesium", "Phosphate": "phosphate", "Potassium": "potassium",
         "Bilirubin_total": "bilirubin", "TroponinI": "troponin", "Hct": "hematocrit", "Hgb": "hemoglobin",
         "PTT": "clotting time (PTT)", "WBC": "white cell count", "Fibrinogen": "fibrinogen", "Platelets": "platelets",
         "shock_index": "shock index (HR / systolic BP)", "labs_drawn_so_far": "number of labs drawn", "labs_drawn_24h": "labs drawn in the last 24 h",
         "ICULOS": "hours in ICU", "HospAdmTime": "time from hospital admission", "Age": "age"}


def feature_groups(columns: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for c in columns:
        base = c
        for suffix in ("_last", "_age", "_mean6h", "_delta6h"):
            if c.endswith(suffix):
                base = c[: -len(suffix)]
        if base in ("Gender", "Unit1", "Unit2"):
            continue                                       # not actionable reasons
        groups.setdefault(base, []).append(c)
    return groups


def reference(X_train_negative: pd.DataFrame) -> pd.Series:
    """Typical value of each feature over non-septic training hours."""
    return X_train_negative.median(numeric_only=True)


def describe(group: str, row: pd.Series) -> str:
    name = NAMES.get(group, group)
    last, age, delta = row.get(f"{group}_last"), row.get(f"{group}_age"), row.get(f"{group}_delta6h")
    if group in VITALS + LABS and pd.notna(last):
        s = f"{name} {last:.3g}"
        if group in VITALS and pd.notna(delta) and abs(delta) >= 0.05 * max(abs(last), 1):
            s += f" ({'up' if delta > 0 else 'down'} {abs(delta):.2g} in 6 h)"
        if group in LABS and pd.notna(age):
            s += f", drawn {int(age)} h ago" if age >= 1 else ", drawn this hour"
        return s
    v = row.get(group)
    if group == "HospAdmTime" and pd.notna(v):
        return f"in hospital {abs(v):.0f} h before ICU admission"
    if group == "ICULOS" and pd.notna(v):
        return f"{v:.0f} hours in ICU"
    return f"{name} {v:.3g}" if pd.notna(v) else name


def explain(bundle: dict, X_rows: pd.DataFrame, ref: pd.Series, top: int = 3) -> list[list[dict]]:
    """For each row: the top reasons [{measure, text, risk_drop}], largest drop first."""
    cols = bundle["columns"]
    X_rows = X_rows[cols]
    groups = feature_groups(cols)
    base = model_mod.predict(bundle, X_rows)
    n, gnames = len(X_rows), list(groups)
    # one batch: every row with every group swapped out
    big = pd.concat([X_rows] * len(gnames), ignore_index=True)
    for i, g in enumerate(gnames):
        sl = slice(i * n, (i + 1) * n)
        for c in groups[g]:
            big.iloc[sl, big.columns.get_loc(c)] = ref.get(c, np.nan)
    drops = (base[None, :] - model_mod.predict(bundle, big).reshape(len(gnames), n)).T   # rows x groups
    out = []
    for r in range(n):
        row = X_rows.iloc[r]
        cands = []
        for j, g in enumerate(gnames):
            if g in VITALS + LABS and pd.isna(row.get(f"{g}_last")):
                continue                                   # only measurements the patient has
            if g not in VITALS + LABS and g in row.index and pd.isna(row.get(g)):
                continue                                   # nor derived values that can't be computed yet
            if drops[r, j] > 0:
                cands.append({"measure": g, "text": describe(g, row), "risk_drop": round(float(drops[r, j]), 4)})
        cands.sort(key=lambda d: -d["risk_drop"])
        out.append(cands[:top])
    return out
