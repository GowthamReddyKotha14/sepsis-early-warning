from sepsis.audit import audit


def test_audit_counts(table):
    r = audit(table)
    assert r["patients"] == 120 and r["patient_hours"] == len(table)
    assert set(r["hospitals"]) == {"A", "B"}
    a = r["hospitals"]["A"]
    assert 0 < a["septic_share"] < 1 and a["septic_patients"] <= a["patients"]
    assert r["missing_share"]["Lactate"] > r["missing_share"]["HR"]
