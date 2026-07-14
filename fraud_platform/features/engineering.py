"""Feature engineering shared by training, streaming, and serving.

Having a single source of truth for feature construction is what keeps the
offline model and the online scorer consistent: the exact same transformation
runs at training time (on a pandas DataFrame) and at serving time (on a single
transaction dict). Skew between the two is the classic cause of "great offline
metrics, useless in production".

Engineered features (the balance-error features are the strongest signals in
PaySim — genuine transactions obey accounting identities, fraud does not):

    amount               raw transaction amount
    errorBalanceOrig     newbalanceOrig + amount - oldbalanceOrg
    errorBalanceDest     oldbalanceDest + amount - newbalanceDest
    oldbalanceOrg        origin balance before
    newbalanceOrig       origin balance after
    oldbalanceDest       destination balance before
    newbalanceDest       destination balance after
    orig_zeroed_out      1 if the origin was fully drained to 0
    dest_is_merchant     1 if destination is a merchant ("M..." account)
    amount_to_oldOrg     amount / (oldbalanceOrg + 1)
    type_TRANSFER        one-hot: transaction type is TRANSFER
    type_CASH_OUT        one-hot: transaction type is CASH_OUT
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

# Order matters: the model is trained on exactly this column order and the
# serving path must reproduce it. Do not reorder without retraining.
FEATURE_COLUMNS: list[str] = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "errorBalanceOrig",
    "errorBalanceDest",
    "amount_to_oldOrg",
    "orig_zeroed_out",
    "dest_is_merchant",
    "type_TRANSFER",
    "type_CASH_OUT",
]

_REQUIRED_RAW_COLUMNS = [
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised feature engineering for a batch of transactions.

    Returns a new DataFrame with exactly ``FEATURE_COLUMNS`` columns.
    """
    missing = [c for c in _REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required raw columns: {missing}")

    out = pd.DataFrame(index=df.index)
    amount = df["amount"].astype(float)
    old_org = df["oldbalanceOrg"].astype(float)
    new_org = df["newbalanceOrig"].astype(float)
    old_dest = df["oldbalanceDest"].astype(float)
    new_dest = df["newbalanceDest"].astype(float)

    out["amount"] = amount
    out["oldbalanceOrg"] = old_org
    out["newbalanceOrig"] = new_org
    out["oldbalanceDest"] = old_dest
    out["newbalanceDest"] = new_dest

    # Accounting-identity error terms: 0 for consistent transactions.
    out["errorBalanceOrig"] = new_org + amount - old_org
    out["errorBalanceDest"] = old_dest + amount - new_dest

    out["amount_to_oldOrg"] = amount / (old_org + 1.0)
    out["orig_zeroed_out"] = ((new_org == 0) & (amount > 0)).astype(int)

    name_dest = df.get("nameDest")
    if name_dest is not None:
        out["dest_is_merchant"] = name_dest.astype(str).str.startswith("M").astype(int)
    else:
        out["dest_is_merchant"] = 0

    tx_type = df["type"].astype(str)
    out["type_TRANSFER"] = (tx_type == "TRANSFER").astype(int)
    out["type_CASH_OUT"] = (tx_type == "CASH_OUT").astype(int)

    return out[FEATURE_COLUMNS].replace([np.inf, -np.inf], 0.0).fillna(0.0)


def engineer_transaction(tx: Mapping[str, object]) -> pd.DataFrame:
    """Feature-engineer a single transaction dict into a 1-row DataFrame.

    Used by the FastAPI serving layer. Missing balance fields default to 0.
    """
    row = {
        "type": tx.get("type", ""),
        "amount": tx.get("amount", 0.0),
        "oldbalanceOrg": tx.get("oldbalanceOrg", 0.0),
        "newbalanceOrig": tx.get("newbalanceOrig", 0.0),
        "oldbalanceDest": tx.get("oldbalanceDest", 0.0),
        "newbalanceDest": tx.get("newbalanceDest", 0.0),
        "nameDest": tx.get("nameDest", ""),
    }
    return build_features(pd.DataFrame([row]))
