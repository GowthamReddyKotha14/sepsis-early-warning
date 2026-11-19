"""Download and load the PhysioNet/CinC Challenge 2019 training data.

Each patient is one pipe-separated .psv file: one row per ICU hour, 40
clinical columns plus SepsisLabel. Missing measurements are "NaN".
Hospital system A = training_setA, B = training_setB.
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PARQUET = ROOT / "data" / "hourly.parquet"

# Two ~20 MB archives on PhysioNet's archive server: 2 requests instead of 40,336.
ZIPS = {
    "A": "https://archive.physionet.org/users/shared/challenge-2019/training_setA.zip",
    "B": "https://archive.physionet.org/users/shared/challenge-2019/training_setB.zip",
}

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = ["BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos", "Calcium",
        "Chloride", "Creatinine", "Bilirubin_direct", "Glucose", "Lactate", "Magnesium", "Phosphate",
        "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets"]
DEMOGRAPHICS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
COLUMNS = VITALS + LABS + DEMOGRAPHICS + ["SepsisLabel"]


def download(dest: Path = RAW, log=print) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name, url in ZIPS.items():
        target = dest / f"training_set{name}"
        if target.exists() and any(target.glob("*.psv")):
            log(f"hospital {name}: already downloaded")
            continue
        log(f"hospital {name}: downloading {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "sepsis-early-warning/0.1"})
        with urllib.request.urlopen(req, timeout=300) as r:
            blob = r.read()
        target.mkdir(exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            members = [m for m in z.namelist() if m.endswith(".psv")]
            for m in members:                      # flatten whatever folder layout the zip uses
                (target / Path(m).name).write_bytes(z.read(m))
        log(f"hospital {name}: {len(members):,} patients")


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
