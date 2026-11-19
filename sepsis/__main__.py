"""python -m sepsis {data,audit,baseline,train,evaluate,explain}

  data      download PhysioNet 2019 (about 41 MB) and build data/hourly.parquet
  audit     cohort summary              -> results/audit.json
  baseline  SIRS and qSOFA scored on validation (hospital A) and external test (hospital B)
                                        -> results/baseline.json
  train     features + gradient-boosted trees on hospital A's training patients
                                        -> models/sepsis_hgb.joblib
  evaluate  pick the alarm threshold on validation, then score hospital B once
                                        -> results/model.json, results/threshold_sweep.csv
  explain   why each hospital B alert fired: top reasons, real alerts vs false alarms
                                        -> results/explanations.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import audit as audit_mod
from . import clinical, data, explain, features, metrics, model, threshold
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


def cmd_explain(a) -> int:
    from collections import Counter
    df = data.load(Path(a.table))
    split = assign_splits(df).to_numpy()
    bundle = model.load(Path(a.model))
    res = json.loads((Path(a.out) / "model.json").read_text())
    thr = res["results"][f"budget {a.budget} alerts/bed/shift"]["threshold"]
    X = features.build(df)
    ref = explain.reference(X[(split == "train") & (df.SepsisLabel.to_numpy() == 0)][bundle["columns"]])
    m = split == "test"
    bdf, bX = df[m].reset_index(drop=True), X[m].reset_index(drop=True)
    risk = model.predict(bundle, bX)
    warn = risk >= thr
    first = (bdf.groupby("patient", sort=False)["hour"].transform("min") == bdf["hour"]).to_numpy()
    rising = np.flatnonzero(warn & (first | ~np.r_[False, warn[:-1]]))
    septic = bdf.groupby("patient", sort=False)["SepsisLabel"].transform("max").to_numpy().astype(bool)
    onset = bdf.groupby("patient", sort=False)["SepsisLabel"].transform(lambda s: s.to_numpy().argmax() + 6 if s.any() else -1).to_numpy()
    print(f"explaining {len(rising):,} alerts on hospital B (threshold {thr:.3f}) ...")
    reasons = []
    for i in range(0, len(rising), 500):
        reasons += explain.explain(bundle, bX.iloc[rising[i:i + 500]], ref)
    tops = {"real sepsis": Counter(), "false alarm": Counter()}
    for idx, r in zip(rising, reasons):
        if r:
            tops["real sepsis" if septic[idx] else "false alarm"][r[0]["measure"]] += 1
    summary = {k: [{"measure": g, "name": explain.NAMES.get(g, g), "share_of_alerts": round(n / max(sum(c.values()), 1), 4)}
                   for g, n in c.most_common(10)] for k, c in tops.items()}
    rng = np.random.default_rng(7)
    examples = []
    for kind, mask in (("real sepsis", septic[rising]), ("false alarm", ~septic[rising])):
        pick = rng.choice(np.flatnonzero(mask), size=min(6, int(mask.sum())), replace=False) if mask.any() else []
        for j in pick:
            idx = rising[j]
            examples.append({"kind": kind, "patient": bdf.patient[idx], "hour": int(bdf.hour[idx]),
                             "risk": round(float(risk[idx]), 3),
                             "hours_before_onset": int(onset[idx] - bdf.hour[idx]) if septic[idx] else None,
                             "reasons": reasons[j]})
    out = {"hospital": "B (external test)", "threshold": thr, "alerts": int(len(rising)),
           "method": "group occlusion against typical non-septic values; see sepsis/explain.py",
           "top_reason_counts": summary, "examples": examples}
    (Path(a.out) / "explanations.json").write_text(json.dumps(out, indent=2))
    for k, rows in summary.items():
        print(f"\nMost common top reason, {k}:")
        for r in rows[:5]:
            print(f"  {r['name']:34} {r['share_of_alerts']:.0%}")
    print("\nExample real-sepsis alert:", "; ".join(x["text"] for x in next((e["reasons"] for e in examples if e["kind"] == "real sepsis"), [])))
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
                     ("train", cmd_train), ("evaluate", cmd_evaluate), ("explain", cmd_explain)):
        sub.add_parser(name).set_defaults(func=fn)
    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
