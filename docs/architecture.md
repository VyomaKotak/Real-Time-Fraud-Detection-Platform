# Architecture

## Overview

The platform scores mobile-money transactions for fraud in real time. It has an
**offline path** (train a model) and two **online paths** that reuse the exact
same feature engineering and model wrapper — eliminating train/serve skew.

```
                       ┌─────────────────────┐
   PaySim CSV ───────▶ │ feature engineering │ ◀─── shared by all paths
   (or synthetic)      └──────────┬──────────┘
                                  │
                          ┌───────▼────────┐
                          │  train model   │  RandomForest / GBM / Logistic
                          └───────┬────────┘
                                  │  models/fraud_model.joblib (+ metadata)
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
        ┌───────────────────┐          ┌────────────────────────┐
        │  FastAPI service  │          │ Spark Structured Stream │
        │  POST /score      │          │  Kafka ▶ pandas_udf ▶   │
        │  (online, low-lat)│          │  Kafka + parquet sinks  │
        └─────────┬─────────┘          └────────────┬───────────┘
                  │                                  │
                  └───────────► Streamlit dashboard ◀┘
                        (live scoring + streaming monitor)
```

## Components

| Module | Responsibility |
|---|---|
| `fraud_platform/config.py` | YAML config with env-var overrides, attribute access |
| `fraud_platform/data/` | Synthetic PaySim generator + schema validation |
| `fraud_platform/features/engineering.py` | **Single source of truth** for features |
| `fraud_platform/training/train.py` | Train, evaluate, persist model + metadata |
| `fraud_platform/serving/` | `FraudModel` wrapper, FastAPI app, Pydantic schemas |
| `fraud_platform/streaming/producer.py` | Replay PaySim into Kafka at a throttled rate |
| `fraud_platform/streaming/spark_stream.py` | Score the stream with a `pandas_udf` |
| `fraud_platform/dashboard/app.py` | Streamlit UI |

## Key design decisions

- **One feature module, no skew.** Training (pandas batch) and serving (single
  transaction / Spark micro-batch) call the same `build_features`.
- **Domain rule enforced in code.** Only `TRANSFER` / `CASH_OUT` can be fraud in
  PaySim; other types are scored `0` without troubling the model.
- **Balance-error features** (`errorBalanceOrig`, `errorBalanceDest`) are the
  strongest signals: genuine transactions obey accounting identities, fraud does
  not.
- **Executor-side model cache** in the Spark UDF loads the joblib model once per
  executor instead of per micro-batch.
- **Java 8/11/17 for Spark 3.5** (its Arrow layer breaks on Java 18+); the job
  warns if it detects an unsupported JDK. The Dockerised path avoids this
  entirely by pinning the runtime inside the container.

## Data flow (streaming)

1. `producer` reads `data/paysim.csv`, publishes JSON transactions to the
   `transactions` Kafka topic (keyed by origin account).
2. `spark_stream` consumes the topic, parses JSON, applies the scoring UDF, and
   writes to (a) the `fraud-scores` topic and (b) `data/scored/*.parquet`.
3. The dashboard reads the parquet for the streaming monitor and calls the
   FastAPI service for interactive scoring.
