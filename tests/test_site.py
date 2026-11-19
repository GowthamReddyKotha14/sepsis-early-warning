import json

from sepsis.__main__ import main


def test_site_export_from_real_pipeline_outputs(raw_dir, tmp_path):
    from sepsis.data import build_table
    table = tmp_path / "hourly.parquet"
    build_table(raw_dir, table, log=lambda *_: None)
    out = tmp_path / "results"
    phys = ["--table", str(table), "--out", str(out), "--model", str(tmp_path / "p.joblib"), "--variant", "physiology"]
    for cmd in ("train", "evaluate", "explain"):
        assert main(phys + [cmd]) == 0
    site = tmp_path / "data.js"
    assert main(phys + ["--site", str(site), "site"]) == 0
    text = site.read_text()
    assert text.startswith("window.SEPSIS = ")
    d = json.loads(text[len("window.SEPSIS = "):].rstrip().rstrip(";"))
    assert d["variant"] == "physiology" and d["ward"]
    for bed in d["ward"]:
        assert len(bed["risk"]) == bed["hours"] == len(bed["vitals"]["HR"])
        assert bed["kind"] in ("caught", "missed", "early", "false alarm", "quiet")
        assert (bed["onset"] is None) == (bed["kind"] in ("false alarm", "quiet"))
        for al in bed["alerts"]:
            assert bed["risk"][al["hour"]] >= d["threshold"] and isinstance(al["reasons"], list)
    assert d["model_physiology"] and d["sweep_validation"]
