"""Schema validation / normalisation for PaySim data.

The real PaySim CSV (Kaggle ``ealaxi/paysim1``) and the synthetic generator
share one canonical schema. This module verifies a loaded DataFrame matches it
and repairs the couple of harmless variations seen in the wild (a stray index
column, or the ``oldbalanceOrig`` spelling some mirrors use instead of the
official ``oldbalanceOrg``). Both the downloader and the trainer call
``validate_paysim`` so a malformed file fails fast with a clear message rather
than deep inside feature engineering.
"""

from __future__ import annotations

import pandas as pd

# The canonical PaySim columns, in the official order.
PAYSIM_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",  # note: official PaySim misspells "Org" (not "Orig")
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]

# Column aliases occasionally seen in mirrors / re-exports -> canonical name.
_ALIASES = {
    "oldbalanceOrig": "oldbalanceOrg",
    "oldBalanceOrig": "oldbalanceOrg",
    "newBalanceOrig": "newbalanceOrig",
    "oldBalanceDest": "oldbalanceDest",
    "newBalanceDest": "newbalanceDest",
    "nameorig": "nameOrig",
    "namedest": "nameDest",
}

_REQUIRED = set(PAYSIM_COLUMNS)


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known column aliases and drop a leading unnamed index column."""
    df = df.rename(columns={k: v for k, v in _ALIASES.items() if k in df.columns})
    # Some exports carry an unnamed integer index as the first column.
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


def validate_paysim(df: pd.DataFrame, *, normalise: bool = True) -> pd.DataFrame:
    """Return ``df`` guaranteed to have the canonical PaySim schema.

    Raises ``ValueError`` if required columns are missing after normalisation.
    """
    if normalise:
        df = normalise_columns(df)

    missing = _REQUIRED - set(df.columns)
    if missing:
        raise ValueError(
            f"PaySim data is missing required columns: {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )

    if df["isFraud"].nunique() < 2:
        raise ValueError(
            "PaySim data has no fraud examples (isFraud is constant); "
            "the model cannot be trained on it."
        )

    # Keep canonical columns first, preserve any extras after them.
    ordered = PAYSIM_COLUMNS + [c for c in df.columns if c not in _REQUIRED]
    return df[ordered]


def describe(df: pd.DataFrame) -> str:
    """A short human-readable summary for logging after a load/download."""
    fraud = int(df["isFraud"].sum())
    return (
        f"{len(df):,} rows | {fraud:,} fraud ({df['isFraud'].mean():.4%}) | "
        f"types: {', '.join(sorted(df['type'].unique()))}"
    )
