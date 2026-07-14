"""Fetch the transaction dataset used to train the fraud model.

Resolution order:
  1. If the target CSV already exists, do nothing.
  2. Download the **real PaySim** dataset from Kaggle (``ealaxi/paysim1``,
     ~6.3M rows) — requires the ``kaggle`` package and API credentials.
  3. Fall back to a synthetic PaySim-like dataset so the platform is always
     runnable (unless ``--kaggle`` is passed, which makes a failed download a
     hard error instead).

Kaggle credentials (either is fine):
  * ``~/.kaggle/kaggle.json``  (chmod 600) — download from your Kaggle account
    page ("Create New API Token"), or
  * environment variables ``KAGGLE_USERNAME`` and ``KAGGLE_KEY``.

Usage:
    python scripts/download_data.py                 # real if creds, else synthetic
    python scripts/download_data.py --kaggle        # real only; error if it fails
    python scripts/download_data.py --force-synthetic --rows 500000
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd

# Allow running as a plain script (``python scripts/download_data.py``).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fraud_platform.config import CONFIG  # noqa: E402
from fraud_platform.data.generate_synthetic import generate  # noqa: E402
from fraud_platform.data.validate import describe, validate_paysim  # noqa: E402


def _have_kaggle_credentials() -> bool:
    """True if Kaggle API credentials are discoverable (file or env vars)."""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    cred_file = Path.home() / ".kaggle" / "kaggle.json"
    return cred_file.exists()


def extract_csv_from_zip(zip_path: Path, dest: Path) -> bool:
    """Extract the first CSV inside ``zip_path`` and move it to ``dest``.

    The real PaySim archive contains one CSV with a long generated name
    (e.g. ``PS_20174392719_1491204439457_log.csv``); we normalise it to the
    configured target path. Returns True on success.
    """
    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            print(f"[kaggle] no CSV inside {zip_path.name}.")
            return False
        extracted_name = csv_names[0]
        zf.extract(extracted_name, path=dest.parent)
    (dest.parent / extracted_name).replace(dest)  # atomic rename
    return True


def _download_from_kaggle(dataset: str, dest: Path) -> bool:
    """Download + unzip the dataset from Kaggle into ``dest``. Returns success."""
    if not _have_kaggle_credentials():
        print(
            "[kaggle] no credentials found. Set KAGGLE_USERNAME / KAGGLE_KEY, "
            "or place kaggle.json in ~/.kaggle/ (chmod 600)."
        )
        return False

    try:
        # Imported lazily: authentication happens on API construction.
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as exc:  # ImportError, etc.
        print(f"[kaggle] client unavailable ({exc}). `pip install kaggle`.")
        return False

    try:
        api = KaggleApi()
        api.authenticate()
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[kaggle] downloading {dataset} (this can take a few minutes) ...")
        api.dataset_download_files(dataset, path=str(dest.parent), unzip=False)
    except Exception as exc:  # network / auth / API errors
        print(f"[kaggle] download failed: {exc}")
        return False

    zips = sorted(dest.parent.glob("*.zip"))
    if not zips:
        print("[kaggle] no zip archive produced by the download.")
        return False
    try:
        if not extract_csv_from_zip(zips[0], dest):
            return False
    finally:
        for z in zips:
            z.unlink(missing_ok=True)
    print(f"[kaggle] saved dataset to {dest}")
    return True


def _write_synthetic(dest: Path, rows: int) -> pd.DataFrame:
    print(f"Generating {rows:,} synthetic PaySim-like rows ...")
    df = generate(n_rows=rows, seed=CONFIG.data.random_state)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Download or generate PaySim data")
    parser.add_argument(
        "--kaggle",
        action="store_true",
        help="Require the real Kaggle dataset; error out instead of falling "
        "back to synthetic data.",
    )
    parser.add_argument(
        "--force-synthetic",
        action="store_true",
        help="Skip Kaggle entirely and generate synthetic data.",
    )
    parser.add_argument("--rows", type=int, default=CONFIG.data.synthetic_rows)
    parser.add_argument("--out", type=str, default=CONFIG.data.raw_path)
    args = parser.parse_args()

    if args.kaggle and args.force_synthetic:
        parser.error("--kaggle and --force-synthetic are mutually exclusive")

    dest = Path(args.out)
    if dest.exists():
        print(f"Dataset already present at {dest}; nothing to do.")
        return

    got_real = False
    if not args.force_synthetic:
        got_real = _download_from_kaggle(CONFIG.data.kaggle_dataset, dest)
        if args.kaggle and not got_real:
            raise SystemExit(
                "Kaggle download failed and --kaggle was requested. "
                "Check your credentials and network, then retry."
            )

    if not got_real:
        _write_synthetic(dest, args.rows)

    # Validate + normalise whatever we ended up with, and report a summary.
    df = validate_paysim(pd.read_csv(dest))
    df.to_csv(dest, index=False)  # persist any column normalisation
    source = "real PaySim (Kaggle)" if got_real else "synthetic"
    print(f"Ready [{source}] -> {dest}\n  {describe(df)}")


if __name__ == "__main__":
    main()
