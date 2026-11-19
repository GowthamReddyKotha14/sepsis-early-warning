"""Gradient-boosted trees on the hourly features.

Trained on hospital A's TRAIN patients only. Validation patients choose the
alarm threshold (threshold.py); hospital B is touched once, at the end.

Choices:
  * HistGradientBoostingClassifier: handles missing values natively (most labs
    are missing most hours), fast on ~1M rows on a laptop, no extra installs.
  * class_weight="balanced": only ~2% of hours are labeled, so unweighted trees
    learn to say "fine" every time.
  * Fixed settings, no built-in early stopping: sklearn's early stopping holds
    out random ROWS, which would put hours of the same patient on both sides
    and overstate how well it's doing.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

from . import features

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "sepsis_hgb.joblib"

PARAMS = dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=200,
              l2_regularization=1.0, class_weight="balanced", early_stopping=False, random_state=2019)


def fit(X: pd.DataFrame, y: np.ndarray, params: dict | None = None) -> HistGradientBoostingClassifier:
    clf = HistGradientBoostingClassifier(**(params or PARAMS))
    clf.fit(X, y)
    return clf


def save(clf, columns: list[str], path: Path = MODEL_PATH, **meta) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "columns": columns, "sklearn": sklearn.__version__, **meta}, path)


def load(path: Path = MODEL_PATH) -> dict:
    return joblib.load(path)


def predict(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    """Risk score per patient-hour (0-1). Columns are matched by name, not position."""
    return bundle["model"].predict_proba(X[bundle["columns"]])[:, 1]


def train(df: pd.DataFrame, split: pd.Series, path: Path = MODEL_PATH, log=print) -> dict:
    t0 = time.time()
    X = features.build(df)
    tr = (split == "train").to_numpy()
    # A column with no values at all in training carries no information, and
    # sklearn's binning fails on it, so it is left out (and recorded).
    empty = [c for c in X.columns if X.loc[tr, c].isna().all()]
    cols = [c for c in X.columns if c not in empty]
    log(f"features: {len(cols)} columns ({len(empty)} empty in training, left out); training on "
        f"{tr.sum():,} patient-hours ({df.patient[tr].nunique():,} patients, hospital A only)")
    clf = fit(X.loc[tr, cols], df.SepsisLabel.to_numpy()[tr])
    save(clf, cols, path, dropped_empty=empty, train_rows=int(tr.sum()), train_seconds=round(time.time() - t0, 1))
    log(f"trained in {time.time() - t0:.0f}s -> {path}")
    return load(path)
