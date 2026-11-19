"""python -m sepsis {data,audit,baseline}

  data      download PhysioNet 2019 (about 41 MB) and build data/hourly.parquet
  audit     cohort summary              -> results/audit.json
  baseline  SIRS and qSOFA scored on validation (hospital A) and external test (hospital B)
                                        -> results/baseline.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import audit as audit_mod
from . import clinical, data, metrics
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m sepsis", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default=str(data.RAW))
    p.add_argument("--table", default=str(data.PARQUET))
    p.add_argument("--out", default=str(RESULTS))
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("data", cmd_data), ("audit", cmd_audit), ("baseline", cmd_baseline)):
        sub.add_parser(name).set_defaults(func=fn)
    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
