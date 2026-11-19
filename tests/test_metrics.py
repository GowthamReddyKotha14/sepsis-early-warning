import numpy as np
import pandas as pd
import pytest

from sepsis.metrics import utility_table, warning_lead_times
from tests.reference.physionet_utility import compute_prediction_utility


def frame(label_lists):
    rows = [(f"p{i}", h, l) for i, labs in enumerate(label_lists) for h, l in enumerate(labs)]
    return pd.DataFrame(rows, columns=["patient", "hour", "SepsisLabel"])


def random_patients(rng, n):
    out = []
    for _ in range(n):
        length = int(rng.integers(1, 80))
        labs = np.zeros(length, int)
        if rng.random() < 0.4:
            labs[int(rng.integers(0, length)):] = 1
        out.append(labs)
    return out


def test_per_patient_utility_matches_official_code():
    rng = np.random.default_rng(1)
    pats = random_patients(rng, 400)
    df = frame(pats)
    ut = utility_table(df)
    warn = rng.random(len(df)) < 0.3
    start = 0
    for labs in pats:
        n = len(labs)
        mine = np.where(warn[start:start + n], ut.u_pos[start:start + n], ut.u_neg[start:start + n]).sum()
        ref = compute_prediction_utility(labs, warn[start:start + n].astype(int))
        assert mine == pytest.approx(ref, abs=1e-9)
        start += n


def test_normalization_anchors():
    rng = np.random.default_rng(2)
    df = frame(random_patients(rng, 200))
    ut = utility_table(df)
    assert ut.normalized(np.zeros(len(df), bool)) == pytest.approx(0.0)
    assert ut.normalized(ut.best) == pytest.approx(1.0)


def test_alarming_on_everyone_scores_below_zero_at_icu_prevalence():
    """At a realistic ~7% sepsis rate, alarming constantly is worse than never alarming."""
    pats = [[0] * 40 for _ in range(93)] + [[0] * 30 + [1] * 10 for _ in range(7)]
    df = frame(pats)
    assert utility_table(df).normalized(np.ones(len(df), bool)) < 0


def test_lead_time():
    # septic patient: labels start at hour 10 -> onset at hour 16
    df = frame([[0] * 10 + [1] * 10, [0] * 20])
    warn = np.zeros(40, bool)
    warn[8] = True                                              # first warning at hour 8 -> 8 h early
    lead = warning_lead_times(df, warn)
    assert lead.to_dict() == {"p0": 8.0}
    assert warning_lead_times(df, np.zeros(40, bool)).isna().all()
