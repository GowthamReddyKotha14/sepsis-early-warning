import json

from sepsis.__main__ import main


def test_audit_and_baseline_end_to_end(raw_dir, tmp_path):
    from sepsis.data import build_table
    table = tmp_path / "hourly.parquet"
    build_table(raw_dir, table, log=lambda *_: None)
    common = ["--table", str(table), "--out", str(tmp_path / "results")]
    assert main(common + ["audit"]) == 0
    assert main(common + ["baseline"]) == 0
    audit = json.loads((tmp_path / "results/audit.json").read_text())
    base = json.loads((tmp_path / "results/baseline.json").read_text())
    assert audit["patients"] == 120
    assert set(base) == {"SIRS >= 2", "qSOFA (adapted)"}
    r = base["SIRS >= 2"]["external test (hospital B)"]
    assert -10 < r["normalized_utility"] < 1 and r["alerts"] >= 0
