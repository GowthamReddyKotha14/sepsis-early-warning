"""The downloader against a local server that mimics PhysioNet's file tree."""
import functools
import http.server
import threading

import pytest

from sepsis.data import build_table, download
from tests.synthetic import write_dataset


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


@pytest.fixture
def physionet(tmp_path):
    site = tmp_path / "site" / "training"
    write_dataset(site, n_per_hospital=25)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(tmp_path / "site")))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_falls_back_to_file_tree_when_archive_is_gone(physionet, tmp_path):
    raw = tmp_path / "raw"
    msgs = []
    download(raw, base=physionet + "/training/", zips={"A": physionet + "/gone.zip", "B": physionet + "/gone.zip"},
             workers=4, log=msgs.append)
    assert any("HTTP 404" in m for m in msgs)
    assert len(list((raw / "training_setA").glob("*.psv"))) == 25
    assert len(list((raw / "training_setB").glob("*.psv"))) == 25
    assert not list(raw.rglob("*.part"))
    df = build_table(raw, tmp_path / "t.parquet", log=lambda *_: None)
    assert df.patient.nunique() == 50


def test_resumes_after_interruption(physionet, tmp_path):
    raw = tmp_path / "raw"
    download(raw, base=physionet + "/training/", zips=None, workers=4, log=lambda *_: None)
    victims = sorted((raw / "training_setB").glob("*.psv"))[:7]
    for v in victims:
        v.unlink()                                       # simulate a download cut off part-way
    msgs = []
    download(raw, base=physionet + "/training/", zips=None, workers=4, log=msgs.append)
    assert any("7 to download" in m for m in msgs)
    assert len(list((raw / "training_setB").glob("*.psv"))) == 25
