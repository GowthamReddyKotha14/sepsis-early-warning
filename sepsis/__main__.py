"""python -m sepsis {data,audit,baseline,train,evaluate}

  data      download PhysioNet 2019 (about 41 MB) and build data/hourly.parquet
  audit     cohort summary              -> results/audit.json
  baseline  SIRS and qSOFA scored on validation (hospital A) and external test (hospital B)
                                        -> results/baseline.json
  train     features + gradient-boosted trees on hospital A's training patients
                                        -> models/sepsis_hgb.joblib
  evaluate  pick the alarm threshold on validation, then score hospital B once
                                        -> results/model.json, results/threshold_sweep.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import audit as audit_mod
from . import clinical, data, features, metrics, model, threshold
from .split import assign_splits

RESULTS = data.ROOT / "results"


def score(df, warn) -> dict:
    ut = metrics.utility_table(df)
    lead = metrics.warning_lead_times(df, warn)
    septic = int(df.groupby("patient").SepsisLabel.max().sum())
    return {
        "normalized_utility": round(ut.normalized(warn), 4),
        "septic_patients": septic,
        "septic_warned_in_time": int((lead >= 0).sum()),
        "median_lead_hours": None if lead.dropna().empty else float(lead.dropna().median()),
        **metrics.alert_burden(df, warn),
    }


def cmd_data(a) -> int:
    data.download(Path(a.raw))
    data.build_table(Path(a.raw), Path(a.table))
    return 0


def cmd_audit(a) -> int:
    report = audit_mod.audit(data.load(Path(a.table)))
    audit_mod.save(report, Path(a.out) / "audit.json")
    print(json.dumps(report["hospitals"], indent=2))
    return 0


def cmd_baseline(a) -> int:
    df = data.load(Path(a.table))
    df["split"] = assign_splits(df).to_numpy()
    out = {}
    for name, fn in (("SIRS >= 2", clinical.sirs_alert), ("qSOFA (adapted)", clinical.qsofa_alert)):
        out[name] = {}
        for part, label in (("val", "validation (hospital A)"), ("test", "external test (hospital B)")):
            sub = df[df.split == part].reset_index(drop=True)
            out[name][label] = score(sub, fn(sub))
            r = out[name][label]
            print(f"{name:16} {label:28} utility {r['normalized_utility']:+.3f}  "
                  f"alerts/bed/shift {r['alerts_per_bed_shift']:.2f}  septic share of alerts {r['septic_alert_share']}")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "baseline.json").write_text(json.dumps(out, indent=2))
    return 0


def cmd_train(a) -> int:
    df = data.load(Path(a.table))
    model.train(df, assign_splits(df), Path(a.model))
    return 0


def cmd_evaluate(a) -> int:
    from sklearn.metrics import average_precision_score, roc_auc_score
    df = data.load(Path(a.table))
    split = assign_splits(df).to_numpy()
    bundle = model.load(Path(a.model))
    risk = model.predict(bundle, features.build(df))
    parts = {}
    for part in ("val", "test"):
        m = split == part
        parts[part] = (df[m].reset_index(drop=True), risk[m])
    vdf, vrisk = parts["val"]
    sweep = threshold.Sweeper(vdf, vrisk).sweep()
    picks = {f"budget {a.budget} alerts/bed/shift": threshold.choose(sweep, a.budget),
             "best utility, no budget": threshold.choose(sweep, None)}
    out = {"model": {"features": len(bundle["columns"]), "train_rows": bundle.get("train_rows")},
           "thresholds_chosen_on": "validation (hospital A)", "results": {}}
    for name, pick in picks.items():
        thr = float(pick.threshold)
        out["results"][name] = {"threshold": round(thr, 5)}
        for part, label in (("val", "validation (hospital A)"), ("test", "external test (hospital B)")):
            sdf, srisk = parts[part]
            r = score(sdf, srisk >= thr)
            r["auroc_hourly"] = round(float(roc_auc_score(sdf.SepsisLabel, srisk)), 4)
            r["auprc_hourly"] = round(float(average_precision_score(sdf.SepsisLabel, srisk)), 4)
            out["results"][name][label] = r
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "model.json").write_text(json.dumps(out, indent=2))
    sweep.round(5).to_csv(Path(a.out) / "threshold_sweep.csv", index=False)

    base_path = Path(a.out) / "baseline.json"
    rows = []
    if base_path.exists():
        for rule, res in json.loads(base_path.read_text()).items():
            rows.append((rule, res["external test (hospital B)"]))
    for name in picks:
        rows.append((f"Model ({name})", out["results"][name]["external test (hospital B)"]))
    print("\nExternal test, hospital B (never used for training or tuning):")
    print(f"{'':44}{'utility':>8}{'alerts/bed/shift':>18}{'real-sepsis alerts':>20}{'septic warned':>16}")
    for name, r in rows:
        print(f"{name:44}{r['normalized_utility']:>+8.3f}{r['alerts_per_bed_shift']:>18.2f}"
              f"{(r['septic_alert_share'] or 0):>20.0%}{r['septic_warned_in_time']:>8}/{r['septic_patients']}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m sepsis", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default=str(data.RAW))
    p.add_argument("--table", default=str(data.PARQUET))
    p.add_argument("--out", default=str(RESULTS))
    p.add_argument("--model", default=str(model.MODEL_PATH))
    p.add_argument("--budget", type=float, default=threshold.DEFAULT_BUDGET,
                   help="max alerts per bed per 12-hour shift when choosing the threshold")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("data", cmd_data), ("audit", cmd_audit), ("baseline", cmd_baseline),
                     ("train", cmd_train), ("evaluate", cmd_evaluate)):
        sub.add_parser(name).set_defaults(func=fn)
    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
