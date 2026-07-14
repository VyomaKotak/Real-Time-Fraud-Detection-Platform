"""Kafka producer that replays the PaySim dataset as a live transaction stream.

Reads ``data/paysim.csv`` and publishes one JSON transaction per Kafka message
to the input topic, throttled to a configurable transactions-per-second rate so
it behaves like a real-time feed rather than a bulk load.

Run:
    python -m fraud_platform.streaming.producer                  # whole dataset
    python -m fraud_platform.streaming.producer --limit 10000 --rate 200
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from fraud_platform.config import CONFIG
from fraud_platform.data.generate_synthetic import generate
from fraud_platform.logging_config import get_logger

logger = get_logger(__name__)

# PaySim / synthetic columns published on the wire.
WIRE_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]


def _load_source(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        logger.info("%s not found; generating synthetic data", csv_path)
        df = generate(n_rows=CONFIG.data.synthetic_rows, seed=CONFIG.data.random_state)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        return df
    return pd.read_csv(csv_path)


def _make_producer():
    # Imported lazily so unit tests / feature work don't require kafka-python.
    from kafka import KafkaProducer

    return KafkaProducer(
        bootstrap_servers=CONFIG.kafka.bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: str(k).encode("utf-8"),
        linger_ms=50,
        acks="all",
        retries=3,
    )


def run(limit: int | None = None, rate: float | None = None) -> None:
    rate = float(rate or CONFIG.kafka.produce_rate)
    topic = CONFIG.kafka.input_topic
    df = _load_source(CONFIG.data.raw_path)
    if limit:
        df = df.head(limit)

    producer = _make_producer()
    interval = 1.0 / rate if rate > 0 else 0.0
    logger.info(
        "Producing %s transactions to topic '%s' @ %g tx/s (bootstrap=%s)",
        f"{len(df):,}",
        topic,
        rate,
        CONFIG.kafka.bootstrap_servers,
    )

    sent = 0
    start = time.perf_counter()
    try:
        for record in df[WIRE_COLUMNS].to_dict(orient="records"):
            # Key by origin account so a customer's transactions land on the
            # same partition (useful for future stateful features).
            producer.send(topic, key=record.get("nameOrig"), value=record)
            sent += 1
            if sent % 1000 == 0:
                elapsed = time.perf_counter() - start
                logger.info(
                    "  sent %s (%.0f tx/s effective)", f"{sent:,}", sent / elapsed
                )
            if interval:
                time.sleep(interval)
    except KeyboardInterrupt:
        logger.warning("Interrupted; flushing ...")
    finally:
        producer.flush()
        producer.close()
    logger.info("Done. Sent %s transactions.", f"{sent:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay PaySim into Kafka")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rate", type=float, default=None)
    args = parser.parse_args()
    run(limit=args.limit, rate=args.rate)


if __name__ == "__main__":
    main()
