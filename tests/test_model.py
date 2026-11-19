import numpy as np
from sklearn.metrics import roc_auc_score

from sepsis import features, model
from sepsis.split import assign_splits


def test_trains_on_train_split_and_learns_signal(table, tmp_path):
    split = assign_splits(table)
    bundle = model.train(table, split, tmp_path / "m.joblib", log=lambda *_: None)
    assert bundle["train_rows"] == int((split == "train").sum())
    X = features.build(table)
    te = (split == "test").to_numpy()                      # hospital B, never seen in training
    p = model.predict(bundle, X[te])
    assert p.min() >= 0 and p.max() <= 1
    assert roc_auc_score(table.SepsisLabel[te], p) > 0.75   # synthetic septic patients do deteriorate


def test_columns_matched_by_name(table, tmp_path):
    split = assign_splits(table)
    bundle = model.train(table, split, tmp_path / "m.joblib", log=lambda *_: None)
    X = features.build(table).iloc[:50]
    shuffled = X[X.columns[::-1]]
    assert np.allclose(model.predict(bundle, X), model.predict(bundle, shuffled))
