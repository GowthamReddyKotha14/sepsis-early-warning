import numpy as np
import pandas as pd
import pytest

from sepsis.threshold import Sweeper, choose


def toy():
    # two septic patients (labels from hour 10 -> onset hour 16), eight not
    rows = []
    for i in range(10):
        for h in range(24):
            rows.append((f"p{i}", h, int(i < 2 and h >= 10)))
    df = pd.DataFrame(rows, columns=["patient", "hour", "SepsisLabel"])
    rng = np.random.default_rng(0)
    risk = rng.random(len(df)) * 0.5 + np.where(df.SepsisLabel == 1, 0.5, 0)
    return df, risk


def test_warning_time_grows_as_threshold_drops_but_alert_count_need_not():
    df, risk = toy()
    sw = Sweeper(df, risk).sweep(50)
    assert sw.warning_hour_share.is_monotonic_decreasing      # thresholds rise down the table
    # the alert count can dip: a lower threshold may merge two warnings into one
    assert sw.alerts_per_bed_shift.iloc[0] > sw.alerts_per_bed_shift.iloc[-1]


def test_choice_respects_budget_and_falls_back():
    df, risk = toy()
    sw = Sweeper(df, risk).sweep(80)
    pick = choose(sw, budget=0.5)
    assert pick.alerts_per_bed_shift <= 0.5
    assert pick.utility == pytest.approx(sw[sw.alerts_per_bed_shift <= 0.5].utility.max())
    quiet = choose(sw, budget=-1)                    # impossible budget -> quietest threshold
    assert quiet.alerts_per_bed_shift == pytest.approx(sw.alerts_per_bed_shift.min())


def test_sweeper_agrees_with_full_utility():
    from sepsis.metrics import utility_table
    df, risk = toy()
    s = Sweeper(df, risk).score(0.6)
    assert s["utility"] == pytest.approx(utility_table(df).normalized(risk >= 0.6))
