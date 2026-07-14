#!/usr/bin/env bash
# One-shot streaming pipeline, run inside the `pipeline` container by
# docker-compose.streaming.yml. It keeps all the Java / Spark / Kafka
# complexity inside Linux containers so it works identically on Windows/Mac/Linux.
#
# Steps: ensure data + model, replay transactions into Kafka, then run the
# Spark scorer (available-now) which writes results to the fraud-scores Kafka
# topic and to data/scored/*.parquet (mounted to the host for the dashboard).
set -euo pipefail

echo "=== [1/4] Ensuring dataset (data/paysim.csv) ==="
python scripts/download_data.py --force-synthetic --rows "${PAYSIM_ROWS:-200000}"

echo "=== [2/4] Ensuring trained model (models/fraud_model.joblib) ==="
if [ ! -f models/fraud_model.joblib ]; then
  python -m fraud_platform.training.train
else
  echo "model already present; skipping training"
fi

echo "=== [3/4] Producing ${PRODUCE_LIMIT:-30000} transactions into Kafka ==="
python -m fraud_platform.streaming.producer --limit "${PRODUCE_LIMIT:-30000}" --rate "${PRODUCE_RATE:-5000}"

echo "=== [4/4] Running Spark Structured Streaming scorer (available-now) ==="
# Fresh checkpoints + output so every run is a clean, deterministic batch.
rm -rf checkpoints data/scored
python -m fraud_platform.streaming.spark_stream --available-now

echo ""
echo "=== DONE. Scores written to Kafka topic 'fraud-scores' and data/scored/*.parquet ==="
echo "=== Open/refresh the dashboard's 'Streaming monitor' tab. ==="
