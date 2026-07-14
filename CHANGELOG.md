# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- XGBoost as a training option (`--algorithm xgboost`) with `scale_pos_weight`
  for the class imbalance.
- `notebooks/eda.ipynb` — exploratory data analysis of the PaySim fraud signal.

### Fixed
- Docker image pinned to `python:3.11-slim-bookworm` with `openjdk-17`; the
  previous base resolved to Debian trixie whose default JRE is Java 21, which
  crashes Spark's Arrow / `pandas_udf`.

## [0.1.0] - 2026-07-01

Initial release.

### Added
- Synthetic PaySim data generator and real Kaggle download with schema
  validation/normalisation.
- Shared feature-engineering module (balance-error terms, drained-account flag,
  transaction-type one-hots) used by both training and serving.
- Model training pipeline (RandomForest / GradientBoosting / Logistic) with
  ROC-AUC, PR-AUC, precision/recall/F1, confusion matrix and feature importances.
- FastAPI scoring service: `/score`, `/score/batch`, `/health`, `/model`,
  `/metrics`.
- Kafka producer and Spark Structured Streaming scorer (pandas UDF) writing to a
  Kafka topic and parquet, with a bounded `--available-now` mode.
- Streamlit dashboard: live scoring + streaming monitor.
- One-command Dockerised streaming pipeline (`docker-compose.streaming.yml`).
- Packaging via `pyproject.toml` with console entry points, ruff + black +
  pre-commit, structured logging, and CI across Python 3.10–3.12.
