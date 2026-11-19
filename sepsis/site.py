"""Export everything the showcase site shows into docs/data.js.

Nothing on the site is typed by hand: summary numbers come from results/*.json
and the ward replay is real hospital B patients, scored hour by hour by the
physiology-only model at the alarm level chosen on hospital A.

The ward: 16 hospital B patients, chosen reproducibly (seed 2019) to show each
outcome honestly, not a highlight reel:
  caught      septic, first alert 0-12 h before onset   (5 beds)
  missed      septic, no alert by 3 h after onset        (2 beds)
  early       septic, first alert more than 12 h early   (1 bed)
  false alarm never septic, alerted                      (3 beds)
  quiet       never septic, never alerted                (5 beds)
Data: PhysioNet/CinC Challenge 2019 (de-identified, open access); see README.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, explain, features, model
from .split import assign_splits

SITE = data.ROOT / "docs" / "data.js"
WARD = {"caught": 5, "missed": 2, "early": 1, "false alarm": 3, "quiet": 5}
SHOW = ["HR", "MAP", "Resp", "Temp", "O2Sat"]


def _r(x, n=1):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def classify(g: pd.DataFrame, warn: np.ndarray) -> tuple[str, int | None, int | None]:
    lab = g.SepsisLabel.to_numpy()
    onset = int(lab.argmax()) + 6 if lab.any() else None
    hits = np.flatnonzero(warn)
    first = int(hits[0]) if len(hits) else None
    if onset is None:
        return ("false alarm" if first is not None else "quiet"), None, first
    in_time = np.flatnonzero(warn[: onset + 4])
    if not len(in_time):
        return "missed", onset, None
    lead = onset - int(in_time[0])
    return ("early" if lead > 12 else "caught"), onset, int(in_time[0])


def build(table: Path, results: Path, variant: str = "physiology", out: Path = SITE,
          budget: float = 0.3, seed: int = 2019, log=print, model_path: Path | None = None) -> dict:
    df = data.load(table)
    split = assign_splits(df).to_numpy()
    path, suffix = model.variant_paths(variant)
    bundle = model.load(model_path or path)
    res = json.loads((Path(results) / f"model{suffix}.json").read_text())
    thr = res["results"][f"budget {budget} alerts/bed/shift"]["threshold"]

    X = features.build(df)
    ref = explain.reference(X[(split == "train") & (df.SepsisLabel.to_numpy() == 0)][bundle["columns"]])
    m = split == "test"
    bdf, bX = df[m].reset_index(drop=True), X[m].reset_index(drop=True)
    risk = model.predict(bundle, bX)

    by_kind: dict[str, list] = {k: [] for k in WARD}
    for pid, idx in bdf.groupby("patient", sort=False).indices.items():
        g = bdf.iloc[idx]
        n = len(g)
        kind, onset, first = classify(g, risk[idx] >= thr)
        if not 24 <= n <= 96 or (onset is not None and onset < 16):
            continue                                   # readable on screen, with a lead-up before onset
        by_kind[kind].append(pid)
    rng = np.random.default_rng(seed)
    chosen = []
    for kind, k in WARD.items():
        pool = sorted(by_kind[kind])
        chosen += [(kind, p) for p in (rng.choice(pool, size=min(k, len(pool)), replace=False) if pool else [])]
    rng.shuffle(chosen)

    beds = []
    for b, (kind, pid) in enumerate(chosen, start=1):
        idx = np.flatnonzero(bdf.patient.to_numpy() == pid)
        g, r = bdf.iloc[idx], risk[idx]
        w = r >= thr
        rising = [int(i) for i in np.flatnonzero(w & ~np.r_[False, w[:-1]])]
        why = explain.explain(bundle, bX.iloc[idx[rising]], ref) if rising else []
        last = g[SHOW].ffill()
        _, onset, first = classify(g, w)
        beds.append({
            "bed": f"Bed {b}", "record": str(pid), "kind": kind, "age": _r(g.Age.iloc[0], 0),
            "hours": int(len(g)), "onset": onset, "first_alert": first,
            "risk": [_r(x, 3) for x in r],
            "vitals": {c: [_r(x) for x in last[c]] for c in SHOW},
            "alerts": [{"hour": h, "risk": _r(r[h], 3), "reasons": [x["text"] for x in rs]}
                       for h, rs in zip(rising, why)],
        })
    log(f"ward: {len(beds)} hospital B patients " + ", ".join(f"{k} {sum(b['kind'] == k for b in beds)}" for k in WARD))

    R = Path(results)
    load = lambda name: json.loads((R / name).read_text()) if (R / name).exists() else None
    sweep = []
    sp = R / f"threshold_sweep{suffix}.csv"
    if sp.exists():
        with sp.open() as f:
            sweep = [{k: _r(float(v), 4) if v not in ("", "nan") else None for k, v in row.items()} for row in csv.DictReader(f)]
    payload = {"threshold": thr, "budget": budget, "variant": variant, "ward": beds,
               "audit": load("audit.json"), "baseline": load("baseline.json"),
               "model_full": load("model.json"), "model_physiology": load("model_physiology.json"),
               "explain_full": load("explanations.json"), "explain_physiology": load("explanations_physiology.json"),
               "sweep_validation": sweep}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("window.SEPSIS = " + json.dumps(payload, separators=(",", ":")) + ";\n")
    log(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    return payload
