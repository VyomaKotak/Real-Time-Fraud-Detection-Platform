"""Generate a synthetic PaySim-like transaction dataset.

The real PaySim dataset (``ealaxi/paysim1`` on Kaggle, ~6.3M rows) requires
Kaggle credentials to download. To keep the whole platform runnable end to end
in any environment (CI, a fresh clone, an interview demo), this module produces
a dataset with the **same schema and the same fraud dynamics** as PaySim:

    step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
    nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud

Key PaySim properties that make the trained model realistic:
  * Fraud occurs **only** in TRANSFER and CASH_OUT transactions.
  * A fraudulent transaction drains the origin account (newbalanceOrig == 0)
    and the amount equals the origin's old balance.
  * ``isFlaggedFraud`` fires for TRANSFERs over 200,000.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_platform.logging_config import get_logger

logger = get_logger(__name__)

TRANSACTION_TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
# Approximate PaySim type frequencies.
TYPE_WEIGHTS = [0.34, 0.084, 0.35, 0.22, 0.006]
FLAG_THRESHOLD = 200_000


def generate(
    n_rows: int = 200_000,
    fraud_rate: float = 0.0013,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a synthetic PaySim-like ``DataFrame`` with ``n_rows`` rows."""
    rng = np.random.default_rng(seed)

    # Steps: 1 unit = 1 hour, 30 days of simulation like PaySim (744 steps).
    steps = rng.integers(1, 744, size=n_rows)
    tx_type = rng.choice(TRANSACTION_TYPES, size=n_rows, p=TYPE_WEIGHTS)

    # Amounts: heavy-tailed, log-normal like real transaction values.
    amount = np.round(rng.lognormal(mean=8.0, sigma=1.4, size=n_rows), 2)

    # Origin opening balance, loosely correlated with amount.
    oldbalance_org = np.round(np.abs(amount * rng.uniform(0.5, 3.0, size=n_rows)), 2)
    # Normal accounting: money leaves the origin.
    newbalance_orig = np.round(np.maximum(oldbalance_org - amount, 0.0), 2)

    oldbalance_dest = np.round(
        np.abs(rng.lognormal(mean=8.5, sigma=1.5, size=n_rows)), 2
    )
    newbalance_dest = np.round(oldbalance_dest + amount, 2)

    is_fraud = np.zeros(n_rows, dtype=int)

    # Fraud only in TRANSFER / CASH_OUT. Pick a subset of those to be fraud.
    fraud_eligible = np.isin(tx_type, ["TRANSFER", "CASH_OUT"])
    eligible_idx = np.flatnonzero(fraud_eligible)
    n_fraud = int(n_rows * fraud_rate)
    n_fraud = min(n_fraud, eligible_idx.size)
    fraud_idx = rng.choice(eligible_idx, size=n_fraud, replace=False)

    is_fraud[fraud_idx] = 1
    # Fraud signature: the account is fully drained (amount == old balance,
    # new balance == 0) and the destination balances stay at zero (mule accts).
    oldbalance_org[fraud_idx] = amount[fraud_idx]
    newbalance_orig[fraud_idx] = 0.0
    oldbalance_dest[fraud_idx] = 0.0
    newbalance_dest[fraud_idx] = 0.0

    # Origins are customers ("C..."), destinations may be customers or
    # merchants ("M..."). Merchants never appear as fraud destinations.
    name_orig = np.array([f"C{n}" for n in rng.integers(1e8, 1e9, size=n_rows)])
    dest_is_merchant = tx_type == "PAYMENT"
    name_dest = np.where(
        dest_is_merchant,
        [f"M{n}" for n in rng.integers(1e8, 1e9, size=n_rows)],
        [f"C{n}" for n in rng.integers(1e8, 1e9, size=n_rows)],
    )
    # Merchant destinations have no tracked balance in PaySim.
    oldbalance_dest[dest_is_merchant] = 0.0
    newbalance_dest[dest_is_merchant] = 0.0

    is_flagged = ((tx_type == "TRANSFER") & (amount > FLAG_THRESHOLD)).astype(int)

    frame = pd.DataFrame(
        {
            "step": steps,
            "type": tx_type,
            "amount": amount,
            "nameOrig": name_orig,
            "oldbalanceOrg": oldbalance_org,
            "newbalanceOrig": newbalance_orig,
            "nameDest": name_dest,
            "oldbalanceDest": oldbalance_dest,
            "newbalanceDest": newbalance_dest,
            "isFraud": is_fraud,
            "isFlaggedFraud": is_flagged,
        }
    )
    return frame.sort_values("step").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic PaySim data")
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--fraud-rate", type=float, default=0.0013)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/paysim.csv")
    args = parser.parse_args()

    df = generate(n_rows=args.rows, fraud_rate=args.fraud_rate, seed=args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(
        "Wrote %s rows (%s fraud, %.4f%%) to %s",
        f"{len(df):,}",
        f"{df.isFraud.sum():,}",
        df.isFraud.mean() * 100,
        out_path,
    )


if __name__ == "__main__":
    main()
