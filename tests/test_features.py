import numpy as np
import pandas as pd

from sepsis import features


def test_shape_and_names(table):
    X = features.build(table)
    assert len(X) == len(table) and X.dtypes.eq("float32").all()
    for name in ("HR_last", "Lactate_age", "HR_delta6h", "MAP_mean6h", "shock_index", "labs_drawn_so_far", "ICULOS"):
        assert name in X.columns


def test_no_peeking_at_the_future(table):
    """Scramble one patient's later hours: features for earlier hours must not change."""
    pid = table.patient.iloc[0]
    rows = table.index[table.patient == pid]
    cut = rows[len(rows) // 2]
    future = table.copy()
    later = rows[rows > cut]
    num = [c for c in features.MEASURED]
    future.loc[later, num] = future.loc[later, num].to_numpy()[::-1] * 3 + 7
    a, b = features.build(table), features.build(future)
    early = rows[rows <= cut]
    pd.testing.assert_frame_equal(a.loc[early], b.loc[early])


def test_carry_forward_and_age():
    df = pd.DataFrame({"patient": "p", "hour": range(4)})
    for c in features.MEASURED + features.DEMOGRAPHICS:
        df[c] = np.nan
    df["Lactate"] = [np.nan, 3.0, np.nan, np.nan]
    X = features.build(df)
    assert X.Lactate_last.tolist()[1:] == [3.0, 3.0, 3.0] and np.isnan(X.Lactate_last.iloc[0])
    assert np.isnan(X.Lactate_age.iloc[0]) and X.Lactate_age.tolist()[1:] == [0.0, 1.0, 2.0]
    assert X.labs_drawn_so_far.tolist() == [0, 1, 1, 1]


def test_patients_do_not_bleed_into_each_other():
    df = pd.DataFrame({"patient": ["a", "a", "b", "b"], "hour": [0, 1, 0, 1]})
    for c in features.MEASURED + features.DEMOGRAPHICS:
        df[c] = np.nan
    df["HR"] = [100.0, np.nan, np.nan, 80.0]
    X = features.build(df)
    assert np.isnan(X.HR_last.iloc[2])                 # b's first hour must not inherit a's 100
    assert X.HR_last.iloc[3] == 80.0
