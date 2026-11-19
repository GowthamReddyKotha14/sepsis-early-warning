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


def test_train_and_evaluate_end_to_end(raw_dir, tmp_path):
    from sepsis.data import build_table
    table = tmp_path / "hourly.parquet"
    build_table(raw_dir, table, log=lambda *_: None)
    common = ["--table", str(table), "--out", str(tmp_path / "results"), "--model", str(tmp_path / "m.joblib")]
    assert main(common + ["baseline"]) == 0
    assert main(common + ["train"]) == 0
    assert main(common + ["evaluate"]) == 0
    res = json.loads((tmp_path / "results/model.json").read_text())
    assert res["thresholds_chosen_on"] == "validation (hospital A)"
    budget = res["results"]["budget 0.3 alerts/bed/shift"]
    assert budget["validation (hospital A)"]["alerts_per_bed_shift"] <= 0.3 + 1e-9
    assert 0 <= budget["external test (hospital B)"]["auroc_hourly"] <= 1
    assert (tmp_path / "results/threshold_sweep.csv").exists()
