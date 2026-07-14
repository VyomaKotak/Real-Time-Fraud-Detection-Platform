"""Tests for the Kaggle download helper (zip extraction) — no network needed."""

import importlib.util
import zipfile
from pathlib import Path

import pandas as pd

from fraud_platform.data.generate_synthetic import generate

# scripts/ isn't a package; load download_data.py by path.
_SPEC = importlib.util.spec_from_file_location(
    "download_data",
    Path(__file__).resolve().parents[1] / "scripts" / "download_data.py",
)
download_data = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(download_data)


def test_extract_csv_from_real_kaggle_style_zip(tmp_path):
    # Simulate Kaggle's archive: a single CSV with the real long file name.
    df = generate(n_rows=1_000, seed=2)
    inner_csv = tmp_path / "PS_20174392719_1491204439457_log.csv"
    df.to_csv(inner_csv, index=False)
    zip_path = tmp_path / "paysim1.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(inner_csv, arcname=inner_csv.name)
    inner_csv.unlink()

    dest = tmp_path / "paysim.csv"
    assert download_data.extract_csv_from_zip(zip_path, dest) is True
    assert dest.exists()
    out = pd.read_csv(dest)
    assert len(out) == 1_000
    assert "isFraud" in out.columns


def test_extract_returns_false_when_no_csv(tmp_path):
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("readme.txt", "no data here")
    assert download_data.extract_csv_from_zip(zip_path, tmp_path / "x.csv") is False


def test_credentials_detection_via_env(monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "u")
    monkeypatch.setenv("KAGGLE_KEY", "k")
    assert download_data._have_kaggle_credentials() is True
