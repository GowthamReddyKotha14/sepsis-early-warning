"""Features a model may use at hour t: only what was known by hour t.

Per patient, in time order:
  * latest value of every vital and lab (carried forward from its last measurement)
  * hours since each variable was last measured. Ordering a lab is itself a
    signal: a lactate drawn an hour ago means someone was worried an hour ago.
  * for the 8 vitals: mean over the last 6 hours and change versus 6 hours ago
  * shock index (heart rate / systolic BP), a bedside sign of circulatory trouble
  * labs drawn so far, demographics, hours in ICU
Missing stays missing (NaN): the model handles it natively, and "never measured"
is information too.

Nothing here looks at later rows of a patient. tests/test_features.py checks
that by changing a patient's future and asserting earlier features don't move.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import DEMOGRAPHICS, LABS, VITALS

MEASURED = VITALS + LABS
WINDOW = 6


def build(df: pd.DataFrame) -> pd.DataFrame:
    """df sorted by patient, hour. Returns a float32 feature frame aligned to df's rows."""
    g = df.groupby("patient", sort=False)
    out = {}
    last = g[MEASURED].ffill()
    for c in MEASURED:
        out[f"{c}_last"] = last[c]

    # hours since last measurement: hour - hour of the most recent non-missing value
    hour = df["hour"].astype("float32")
    for c in MEASURED:
        seen_at = hour.where(df[c].notna())
        out[f"{c}_age"] = hour - seen_at.groupby(df["patient"], sort=False).ffill()

    # short-term trends of the vitals, on carried-forward values
    for c in VITALS:
        lc = last[c]
        lag = lc.groupby(df["patient"], sort=False)
        shifted = [lc] + [lag.shift(k) for k in range(1, WINDOW)]
        out[f"{c}_mean{WINDOW}h"] = pd.concat(shifted, axis=1).mean(axis=1, skipna=True)
        out[f"{c}_delta{WINDOW}h"] = lc - lag.shift(WINDOW)

    out["shock_index"] = last["HR"] / last["SBP"]
    out["labs_drawn_so_far"] = df[LABS].notna().sum(axis=1).groupby(df["patient"], sort=False).cumsum()
    for c in DEMOGRAPHICS:
        out[c] = df[c]
    return pd.DataFrame(out, index=df.index).astype("float32")
