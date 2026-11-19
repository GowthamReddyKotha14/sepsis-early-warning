"""Download and load the PhysioNet/CinC Challenge 2019 training data.

Each patient is one pipe-separated .psv file: one row per ICU hour, 40
clinical columns plus SepsisLabel. Missing measurements are "NaN".
Hospital system A = training_setA, B = training_setB.
"""
from __future__ import annotations

import io
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PARQUET = ROOT / "data" / "hourly.parquet"

# The dataset's official file tree. Each hospital is a folder of one .psv per patient.
BASE = "https://physionet.org/files/challenge-2019/1.0.0/training/"
# Older bulk archives, tried first because 2 requests beat 40,336; PhysioNet has
# been retiring archive.physionet.org, so a 404 here just means "use the file tree".
ZIPS = {
    "A": "https://archive.physionet.org/users/shared/challenge-2019/training_setA.zip",
    "B": "https://archive.physionet.org/users/shared/challenge-2019/training_setB.zip",
}
UA = {"User-Agent": "sepsis-early-warning/0.1"}
CERT_HINT = ("Python could not verify PhysioNet's HTTPS certificate. On a Mac with Python from python.org, "
             "open Finder > Applications > Python 3.x and double-click 'Install Certificates.command', then retry.")


def _get(url: str, timeout: int = 120, tries: int = 4) -> bytes:
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code < 500 or attempt == tries - 1:
                raise
        except urllib.error.URLError as e:
            if isinstance(e.reason, ssl.SSLCertVerificationError):
                raise SystemExit(CERT_HINT) from e
            if attempt == tries - 1:
                raise
        time.sleep(2 ** attempt)                     # brief backoff on server hiccups
    raise RuntimeError("unreachable")


def _links(url: str) -> list[str]:
    html = _get(url).decode("utf-8", "replace")
    return [urllib.parse.unquote(h) for h in re.findall(r'href="([^"?#]+)"', html)]


def _download_tree(folder_url: str, target: Path, workers: int, log) -> int:
    names = sorted({Path(h).name for h in _links(folder_url) if h.endswith(".psv")})
    if not names:
        raise RuntimeError(f"No .psv files listed at {folder_url}")
    todo = [n for n in names if not (target / n).exists()]
    log(f"  {len(names):,} patient files listed, {len(todo):,} to download")
    target.mkdir(parents=True, exist_ok=True)

    def fetch(name: str):
        tmp = target / (name + ".part")
        tmp.write_bytes(_get(folder_url + name))
        tmp.rename(target / name)                    # never leave a half-written .psv behind

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(fetch, todo):
            done += 1
            if done % 2000 == 0:
                log(f"  {done:,}/{len(todo):,}")
    return len(names)

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = ["BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos", "Calcium",
        "Chloride", "Creatinine", "Bilirubin_direct", "Glucose", "Lactate", "Magnesium", "Phosphate",
        "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets"]
DEMOGRAPHICS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
COLUMNS = VITALS + LABS + DEMOGRAPHICS + ["SepsisLabel"]


def download(dest: Path = RAW, base: str = BASE, zips: dict | None = ZIPS, workers: int = 16, log=print) -> None:
    """Resumable: files already on disk are skipped, so re-running after an interruption continues."""
    dest.mkdir(parents=True, exist_ok=True)
    folders = {}
    for name in ("A", "B"):
        target = dest / f"training_set{name}"
        if zips and name in zips and not any(target.glob("*.psv")):
            try:
                with zipfile.ZipFile(io.BytesIO(_get(zips[name], timeout=300))) as z:
                    members = [m for m in z.namelist() if m.endswith(".psv")]
                    target.mkdir(exist_ok=True)
                    for m in members:
                        (target / Path(m).name).write_bytes(z.read(m))
                log(f"hospital {name}: {len(members):,} patients (bulk archive)")
                continue
            except urllib.error.HTTPError as e:
                log(f"hospital {name}: bulk archive unavailable (HTTP {e.code}); using the file tree")
        if not folders:
            folders = {Path(h.rstrip("/")).name: h for h in _links(base) if h.endswith("/")}
        sub = next((f for k, f in folders.items() if k.lower().endswith(f"set{name.lower()}")), None)
        if sub is None:
            raise RuntimeError(f"Couldn't find hospital {name}'s folder at {base} (found: {sorted(folders)})")
        url = urllib.parse.urljoin(base, sub if sub.endswith("/") else sub + "/")
        log(f"hospital {name}: {url}")
        n = _download_tree(url, target, workers, log)
        log(f"hospital {name}: {n:,} patients")


def read_patient(path: Path, hospital: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="|")
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    df = df[COLUMNS].astype("float32")
    df["SepsisLabel"] = df["SepsisLabel"].astype("int8")
    df.insert(0, "hour", np.arange(len(df), dtype="int16"))
    df.insert(0, "hospital", hospital)
    df.insert(0, "patient", path.stem)
    return df


def build_table(raw: Path = RAW, out: Path = PARQUET, log=print) -> pd.DataFrame:
    """All patients in one long table: patient, hospital, hour, 40 variables, label."""
    parts = []
    for hosp in ("A", "B"):
        files = sorted((raw / f"training_set{hosp}").glob("*.psv"))
        parts += [read_patient(f, hosp) for f in files]
    if not parts:
        raise FileNotFoundError(f"No .psv files under {raw}. Run `python -m sepsis data` first.")
    df = pd.concat(parts, ignore_index=True)
    df["hospital"] = df["hospital"].astype("category")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    log(f"{df.patient.nunique():,} patients, {len(df):,} patient-hours -> {out}")
    return df


def load(path: Path = PARQUET) -> pd.DataFrame:
    return pd.read_parquet(path)
