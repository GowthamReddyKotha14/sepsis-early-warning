import numpy as np
import pandas as pd

from sepsis.clinical import qsofa_alert, sirs_count
from sepsis.metrics import alert_burden


def one_patient(**cols):
    n = len(next(iter(cols.values())))
    base = {c: [np.nan] * n for c in ["HR", "Resp", "PaCO2", "Temp", "WBC", "SBP"]}
    base.update(cols)
    return pd.DataFrame({"patient": "p1", "hour": range(n), "SepsisLabel": 0, **base})


def test_sirs_counts_criteria_and_carries_values_forward_only():
    df = one_patient(HR=[95, np.nan, 80], Resp=[np.nan, 25, np.nan], Temp=[38.5, np.nan, np.nan])
    # hour 0: HR + Temp = 2; hour 1: HR (carried) + Resp + Temp (carried) = 3; hour 2: HR now 80 -> 2
    assert sirs_count(df).tolist() == [2, 3, 2]
    df2 = one_patient(HR=[np.nan, 95])
    assert sirs_count(df2).tolist() == [0, 1]                       # never filled backwards


def test_qsofa_needs_both_signs():
    df = one_patient(Resp=[24, 24, 18], SBP=[120, 95, 95])
    assert qsofa_alert(df).tolist() == [False, True, False]


def test_alert_burden_counts_switch_ons_per_patient():
    df = pd.DataFrame({"patient": ["a"] * 6 + ["b"] * 6, "hour": list(range(6)) * 2,
                       "SepsisLabel": [0] * 6 + [0, 0, 0, 1, 1, 1]})
    warn = np.array([1, 1, 0, 1, 0, 0,   1, 1, 1, 1, 1, 1], bool)  # a: 2 alerts; b: 1 alert (starts at its hour 0)
    r = alert_burden(df, warn)
    assert r["alerts"] == 3
    assert r["alerts_per_bed_shift"] == round(3 / (12 / 12), 4)
    assert r["septic_alert_share"] == round(1 / 3, 4)
