# Real-Time Fraud Detection Platform

Streaming fraud detection on **6.3M real mobile-money transactions**, scored end to end through
**Kafka → Spark Structured Streaming → an ML model → FastAPI → a live dashboard**.

**PR-AUC 0.998** at a 0.13% fraud base rate. 1,638 of 1,643 frauds caught, at a cost of 41 false
positives across 554,082 held-out transactions.

![Streamlit dashboard streaming monitor](docs/monitor.png)

---

## The problem

Banks score every transaction as it happens. A transfer has to be judged fraudulent or legitimate
in milliseconds, at thousands per second, and the model has to be good enough that blocking a real
customer stays rare.

The difficulty is not the classifier. It is that **fraud is 0.129% of transactions** — 8,213 in
6,362,620. A model that predicts "never fraud" is 99.87% accurate and completely worthless. That
imbalance shapes every decision in this project, including which metric to report.

## Results (real PaySim, 6,362,620 rows)

| Metric | Value |
|---|---|
| **PR-AUC** | **0.9980** |
| ROC-AUC | 0.9985 |
| Fraud precision | 0.9756 |
| Fraud recall | 0.9970 |
| Fraud F1 | 0.9862 |
| Train / test rows | 2,216,327 / 554,082 |

**Confusion matrix** (554,082 held-out transactions):

|  | Predicted legit | Predicted fraud |
|---|---|---|
| **Actually legit** | 552,398 | 41 |
| **Actually fraud** | 5 | 1,638 |

**PR-AUC is the headline, not ROC-AUC.** At a 0.13% base rate the true-negative pool is so large
that false positives barely move the false-positive *rate* — ROC-AUC flatters almost any model on
data this imbalanced. Precision-recall is the honest measure.

### The tradeoff

Those 41 false positives and 5 false negatives are not a defect. They are a *choice*, made at
`decision_threshold = 0.5`:

- **41 false positives** = 41 customers whose card is declined at a till. They call the bank.
- **5 false negatives** = 5 frauds that go through, and are likely reimbursed.

Which error costs more depends on the amounts involved and the price of a support call versus a
chargeback. A real fraud team tunes that threshold against a cost matrix and revisits it as fraud
patterns shift. The model does not decide this; the business does. The threshold is exposed in
`config/config.yaml` for exactly that reason.

---

## Architecture

```
                 ┌──────────────┐     ┌───────────────────────────┐     ┌─────────────┐
  PaySim CSV ──▶ │ Kafka        │ ──▶ │ Spark Structured Streaming│ ──▶ │ Kafka       │
  (producer)     │ transactions │     │  • parse JSON             │     │ fraud-scores│
                 └──────────────┘     │  • pandas_udf(model)      │     └─────────────┘
                                      │  • threshold → alert      │           │
                                      └───────────┬───────────────┘           ▼
                                                  │                    ┌─────────────┐
                                          parquet ▼  scores            │  Streamlit  │
                                      ┌───────────────────┐            │  dashboard  │
                                      │  data/scored/*.pq │ ─────────▶ │             │
                                      └───────────────────┘            └─────────────┘

  FastAPI scoring service ── same model, same feature code ──────────────────┘
      POST /score   POST /score/batch   GET /health   GET /model   GET /metrics
```

**The design decision that matters: one feature module, no train/serve skew.**

Training (pandas, batch), the FastAPI service (single transaction), and the Spark scorer (micro-batch
`pandas_udf`) all call the same `build_features`. Train/serve skew — where features computed in
production drift subtly from those computed at training time — is one of the most common and most
expensive failure modes in deployed ML. Sharing the code path eliminates it structurally rather than
hoping a test catches it.

**Proof it works:** the streaming run scored 30,000 live transactions and predicted 84 frauds against
84 actual — 83 true positives, 1 false positive, 1 false negative. The online numbers track the
offline evaluation.

---

## The signal: balance-error features

Fraud in PaySim has a signature, and finding it is the difference between throwing columns at a
classifier and understanding the domain.

Genuine transactions obey accounting identities: money out of one account equals money into another.
Fraud breaks them. So:

```python
errorBalanceOrig = newbalanceOrig + amount - oldbalanceOrg   # 0 when consistent
errorBalanceDest = oldbalanceDest + amount - newbalanceDest
```

The trained model confirms this is where the signal lives:

| Feature | Importance |
|---|---|
| `errorBalanceOrig` | **0.356** |
| `amount_to_oldOrg` | 0.274 |
| `oldbalanceOrg` | 0.117 |
| `newbalanceOrig` | 0.057 |
| `newbalanceDest` | 0.049 |
| `orig_zeroed_out` | 0.048 |

Over a third of the model's decision comes from a balance identity that should never be violated.

Two further domain rules are enforced in code rather than learned:

- **Fraud only ever occurs in `TRANSFER` and `CASH_OUT`.** The other 3.6M transactions are never
  scored — they cannot be fraud, so troubling the model with them only adds noise.
- **Fraudsters drain the account.** `amount == oldbalanceOrg`, `newbalanceOrig == 0`.

The raw data shows the attack pattern plainly. Flagged transactions come in pairs at identical
amounts — a `TRANSFER` out of the victim's account, then an immediate `CASH_OUT`:

| type | amount | fraud_probability |
|---|---|---|
| TRANSFER | 1,996.17 | 1.00 |
| CASH_OUT | 1,996.17 | 1.00 |
| TRANSFER | 181.00 | 1.00 |
| CASH_OUT | 181.00 | 1.00 |

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

python scripts/download_data.py    # real PaySim via Kaggle, or synthetic fallback
fraud-train                        # -> models/fraud_model.joblib + metadata

uvicorn fraud_platform.serving.app:app --port 8000   # http://localhost:8000/docs
streamlit run fraud_platform/dashboard/app.py        # http://localhost:8501
```

**The full streaming pipeline**, Kafka and Spark in containers — no local Java needed:

```bash
docker compose -f docker-compose.streaming.yml up --build
```

Then open the dashboard's **Streaming monitor** tab.

Score a transaction directly:

```bash
curl -X POST http://localhost:8000/score -H 'Content-Type: application/json' -d '{
  "type": "TRANSFER", "amount": 181000,
  "oldbalanceOrg": 181000, "newbalanceOrig": 0,
  "oldbalanceDest": 0, "newbalanceDest": 0, "nameDest": "C1666544295"
}'
# -> {"fraud_probability": 1.0, "is_fraud": true, "threshold": 0.5, "scored": true}
```

---

## Engineering

Not a notebook. A system.

| | |
|---|---|
| **26 passing tests** | Feature layer, data validation, model wrapper, all four algorithms, FastAPI endpoints via `TestClient`. No Kafka or Spark needed, so they run anywhere and in CI. |
| **CI** | ruff + black + pytest on Python 3.10 / 3.11 / 3.12, every push. |
| **Config-driven** | One `config.yaml`, env-var overridable. No magic numbers scattered through the code. |
| **Containerised** | Whole Kafka + Spark pipeline in one command. |
| **Graceful degradation** | No Kaggle account? A synthetic generator with the same schema and fraud dynamics means a fresh clone runs end to end. |
| **Executor-side model cache** | The Spark UDF loads the model once per executor, not once per micro-batch. |

### A note on the synthetic fallback

The synthetic generator produces **ROC-AUC 1.0000** — perfect precision, perfect recall, zero errors.
That is not a good result; it means the generator encodes the fraud rule so cleanly that the classes
separate trivially. It is useful for smoke-testing the pipeline and worthless as a measure of the
model.

Every number in this README comes from the real 6.3M-row dataset. A perfect score is a red flag, not
an achievement.

---

## Project layout

```
fraud_platform/
  config.py                   Config loader, env-var overrides
  data/                       Synthetic generator + schema validation
  features/engineering.py     Single source of truth for features
  training/train.py           Train, evaluate, persist
  serving/
    model.py                  Model wrapper — used by API and Spark alike
    app.py                    FastAPI scoring service
  streaming/
    producer.py               Kafka producer replaying PaySim
    spark_stream.py           Spark Structured Streaming scorer (pandas_udf)
  dashboard/app.py            Streamlit dashboard
tests/                        26 tests
config/config.yaml            Central config
docker-compose.streaming.yml  One-command Kafka + Spark pipeline
.github/workflows/ci.yml      CI: ruff + black + pytest
notebooks/eda.ipynb           Exploratory analysis
```

## Model options

`random_forest` (default), `xgboost` (with `scale_pos_weight` for the imbalance),
`gradient_boosting`, `logistic` — selectable via `fraud-train --algorithm`.

## Stack

Python · scikit-learn · XGBoost · Apache Kafka · Apache Spark (Structured Streaming) ·
FastAPI · Streamlit · Docker · pytest · GitHub Actions

## Data

[PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) — a 6.3M-row simulation of mobile-money
transactions built from real African mobile-money logs, with labelled fraud. Real fraud data is
confidential; this is the closest public proxy.

