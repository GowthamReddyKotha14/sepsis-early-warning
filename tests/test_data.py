import pandas as pd
import pytest

from sepsis.data import COLUMNS, read_patient


def test_table_shape_and_types(table):
    assert table.patient.nunique() == 120
    assert set(table.hospital.unique()) == {"A", "B"}
    assert list(table.columns[:3]) == ["patient", "hospital", "hour"]
    assert set(COLUMNS) <= set(table.columns)
    assert table.SepsisLabel.dtype == "int8"


def test_hours_count_up_from_zero_per_patient(table):
    for _, g in table.groupby("patient"):
        assert g.hour.tolist() == list(range(len(g)))


def test_label_never_switches_back_off(table):
    """Once a patient's label turns 1 it stays 1 (it marks 6 h before onset onward)."""
    for _, g in table.groupby("patient"):
        lab = g.SepsisLabel.to_numpy()
        assert (lab[1:] >= lab[:-1]).all()


def test_missing_values_survive_as_nan(table):
    assert table["Lactate"].isna().mean() > 0.8          # labs are sparse
    assert table["HR"].isna().mean() < 0.2


def test_bad_file_is_rejected(tmp_path):
    p = tmp_path / "p999999.psv"
    pd.DataFrame({"HR": [80.0]}).to_csv(p, sep="|", index=False)
    with pytest.raises(ValueError, match="missing columns"):
        read_patient(p, "A")
