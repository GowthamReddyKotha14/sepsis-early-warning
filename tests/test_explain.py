import numpy as np
import pandas as pd

from sepsis import explain, features, model
from sepsis.split import assign_splits


def test_groups_cover_measurements_and_skip_admin_flags(table):
    cols = list(features.build(table).columns)
    g = explain.feature_groups(cols)
    assert set(g["HR"]) == {"HR_last", "HR_age", "HR_mean6h", "HR_delta6h"}
    assert "Gender" not in g and "Unit1" not in g


def test_finds_a_planted_cause(table, tmp_path):
    """A model that only looks at lactate must be explained by lactate."""
    X = features.build(table)
    X = X[[c for c in X.columns if X[c].notna().any()]]          # as model.train does
    y = (X["Lactate_last"].fillna(1.0) > 2.5).astype(int).to_numpy()
    clf = model.fit(X, y, dict(max_iter=50, early_stopping=False, random_state=0))
    bundle = {"model": clf, "columns": list(X.columns)}
    ref = explain.reference(X[y == 0])
    hot = X[(X["Lactate_last"] > 3.5)].head(5)
    reasons = explain.explain(bundle, hot, ref)
    assert all(r and r[0]["measure"] == "Lactate" for r in reasons)
    assert "lactate" in reasons[0][0]["text"] and "drawn" in reasons[0][0]["text"]


def test_never_offers_unmeasured_labs(table, tmp_path):
    split = assign_splits(table)
    bundle = model.train(table, split, tmp_path / "m.joblib", log=lambda *_: None)
    X = features.build(table)
    ref = explain.reference(X[(split == "train").to_numpy() & (table.SepsisLabel == 0).to_numpy()])
    rows = X[X["Lactate_last"].isna()].head(20)
    for reasons in explain.explain(bundle, rows, ref):
        assert all(r["measure"] != "Lactate" for r in reasons)


def test_describe_reads_like_a_chart_note():
    row = pd.Series({"HR_last": 118.0, "HR_delta6h": 24.0, "Lactate_last": 3.4, "Lactate_age": 2.0})
    assert explain.describe("HR", row) == "heart rate 118 (up 24 in 6 h)"
    assert explain.describe("Lactate", row) == "lactate 3.4, drawn 2 h ago"
