import pytest

from sepsis.data import build_table
from tests.synthetic import write_dataset


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory):
    return write_dataset(tmp_path_factory.mktemp("raw"))


@pytest.fixture(scope="session")
def table(raw_dir, tmp_path_factory):
    return build_table(raw_dir, tmp_path_factory.mktemp("out") / "hourly.parquet", log=lambda *_: None)
