"""Tests for PaySim schema validation / normalisation."""

import pytest

from fraud_platform.data.generate_synthetic import generate
from fraud_platform.data.validate import (
    PAYSIM_COLUMNS,
    normalise_columns,
    validate_paysim,
)


def test_synthetic_passes_validation():
    df = validate_paysim(generate(n_rows=5_000, seed=3))
    assert list(df.columns)[: len(PAYSIM_COLUMNS)] == PAYSIM_COLUMNS


def test_alias_columns_are_normalised():
    df = generate(n_rows=2_000, seed=4).rename(
        columns={"oldbalanceOrg": "oldbalanceOrig"}
    )
    assert "oldbalanceOrig" in df.columns
    out = validate_paysim(df)
    assert "oldbalanceOrg" in out.columns
    assert "oldbalanceOrig" not in out.columns


def test_unnamed_index_column_dropped():
    df = generate(n_rows=1_000, seed=5)
    df.insert(0, "Unnamed: 0", range(len(df)))
    out = normalise_columns(df)
    assert not any(str(c).startswith("Unnamed") for c in out.columns)


def test_missing_columns_raise():
    df = generate(n_rows=1_000, seed=6).drop(columns=["isFlaggedFraud"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_paysim(df)


def test_constant_isfraud_raises():
    df = generate(n_rows=1_000, seed=7)
    df["isFraud"] = 0  # no positives
    with pytest.raises(ValueError, match="no fraud examples"):
        validate_paysim(df)


def test_extra_columns_preserved_after_canonical():
    df = generate(n_rows=5_000, seed=8)
    df["extra_signal"] = 1.0
    out = validate_paysim(df)
    assert out.columns[-1] == "extra_signal"
    assert list(out.columns)[: len(PAYSIM_COLUMNS)] == PAYSIM_COLUMNS
