"""Patient-level splits.

Hospital A -> train (80%) / validation (20%), stratified so both have the
same sepsis share. Hospital B -> held out entirely as an EXTERNAL test set.

Why: the challenge's main finding was that models degrade at hospitals they
weren't trained on. Testing on hours from the same patients you trained on
(a row-level split) leaks information and inflates every number. Testing on
a different hospital is the honest question: "would this work elsewhere?"
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 2019


def assign_splits(df: pd.DataFrame, val_share: float = 0.2, seed: int = SEED) -> pd.Series:
    """Return a 'split' label (train / val / test) aligned with df's rows."""
    pt = df.groupby("patient", sort=False).agg(hospital=("hospital", "first"), septic=("SepsisLabel", "max"))
    rng = np.random.default_rng(seed)
    split = pd.Series("test", index=pt.index)
    a = pt[pt.hospital == "A"]
    for _, grp in a.groupby("septic"):
        ids = grp.index.to_numpy().copy()
        rng.shuffle(ids)
        n_val = int(round(len(ids) * val_share))
        split[ids[:n_val]] = "val"
        split[ids[n_val:]] = "train"
    return df["patient"].map(split).rename("split")
