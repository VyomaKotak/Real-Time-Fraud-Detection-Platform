"""Tests for the shared feature-engineering layer."""

import numpy as np
import pandas as pd
import pytest

from fraud_platform.features.engineering import (
    FEATURE_COLUMNS,
    build_features,
    engineer_transaction,
)


@pytest.fixture
def sample_batch() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "type": "TRANSFER",
                "amount": 181000.0,
                "oldbalanceOrg": 181000.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
                "nameDest": "C1666544295",
            },
            {
                "type": "PAYMENT",
                "amount": 4200.0,
                "oldbalanceOrg": 25000.0,
                "newbalanceOrig": 20800.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
                "nameDest": "M1979787155",
            },
        ]
    )


def test_build_features_columns_and_order(sample_batch):
    feats = build_features(sample_batch)
    assert list(feats.columns) == FEATURE_COLUMNS
    assert len(feats) == len(sample_batch)


def test_error_balance_terms(sample_batch):
    feats = build_features(sample_batch)
    # errorBalanceOrig = newbalanceOrig + amount - oldbalanceOrg
    # Row 0: 0 + 181000 - 181000 == 0 (consistent debit)
    assert feats.loc[0, "errorBalanceOrig"] == pytest.approx(0.0)
    # Row 1 PAYMENT: 20800 + 4200 - 25000 == 0
    assert feats.loc[1, "errorBalanceOrig"] == pytest.approx(0.0)


def test_orig_zeroed_out_flag(sample_batch):
    feats = build_features(sample_batch)
    assert feats.loc[0, "orig_zeroed_out"] == 1  # drained account
    assert feats.loc[1, "orig_zeroed_out"] == 0


def test_dest_is_merchant_flag(sample_batch):
    feats = build_features(sample_batch)
    assert feats.loc[0, "dest_is_merchant"] == 0  # C... account
    assert feats.loc[1, "dest_is_merchant"] == 1  # M... merchant


def test_type_one_hot(sample_batch):
    feats = build_features(sample_batch)
    assert feats.loc[0, "type_TRANSFER"] == 1
    assert feats.loc[0, "type_CASH_OUT"] == 0
    assert feats.loc[1, "type_TRANSFER"] == 0


def test_missing_columns_raises():
    with pytest.raises(ValueError):
        build_features(pd.DataFrame([{"amount": 10.0}]))


def test_engineer_single_transaction():
    feats = engineer_transaction(
        {
            "type": "CASH_OUT",
            "amount": 1000.0,
            "oldbalanceOrg": 1000.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 1000.0,
            "nameDest": "C123",
        }
    )
    assert feats.shape == (1, len(FEATURE_COLUMNS))
    assert feats.loc[0, "type_CASH_OUT"] == 1


def test_no_inf_or_nan_leaks():
    # amount_to_oldOrg divides by (oldbalanceOrg + 1); ensure it never blows up.
    df = pd.DataFrame(
        [
            {
                "type": "TRANSFER",
                "amount": 1e9,
                "oldbalanceOrg": 0.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
                "nameDest": "C1",
            }
        ]
    )
    feats = build_features(df)
    assert np.isfinite(feats.to_numpy()).all()
