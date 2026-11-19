from sepsis.split import assign_splits


def test_split_is_by_patient_and_keeps_hospital_b_out(table):
    s = assign_splits(table)
    by_patient = s.groupby(table.patient).nunique()
    assert (by_patient == 1).all()                                  # no patient in two splits
    assert set(s[table.hospital == "B"]) == {"test"}
    assert set(s[table.hospital == "A"]) == {"train", "val"}


def test_split_is_stratified_and_reproducible(table):
    s1, s2 = assign_splits(table), assign_splits(table)
    assert (s1 == s2).all()
    septic = table.groupby("patient").SepsisLabel.max()
    sp = s1.groupby(table.patient).first()
    for part in ("train", "val"):
        assert septic[sp == part].sum() > 0                         # both contain septic patients
