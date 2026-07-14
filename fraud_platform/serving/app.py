"""FastAPI real-time fraud scoring service.

Endpoints:
    GET  /health           liveness + model status
    GET  /metrics          in-process scoring counters (Prometheus-friendly text)
    GET  /model            model metadata (algorithm, metrics, feature importances)
    POST /score            score a single transaction
    POST /score/batch      score a list of transactions

Run:
    uvicorn fraud_platform.serving.app:app --host 0.0.0.0 --port 8000
    # or: python -m fraud_platform.serving.app
"""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response

from fraud_platform.config import CONFIG
from fraud_platform.logging_config import get_logger
from fraud_platform.serving.model import FraudModel
from fraud_platform.serving.schemas import (
    BatchScoreRequest,
    BatchScoreResponse,
    HealthResponse,
    ScoreResponse,
    Transaction,
)

logger = get_logger(__name__)

# Simple in-process counters so the dashboard has live metrics without needing
# a separate metrics store. Guarded by a lock for thread safety under uvicorn.
_metrics_lock = threading.Lock()
_metrics = {"scored_total": 0, "fraud_total": 0, "amount_flagged": 0.0}

model = FraudModel()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load the model at startup if it exists; otherwise start in a degraded
    # state so /health can report it rather than crashing the process.
    try:
        model.load()
        logger.info("Model loaded: %s", model.metadata.get("algorithm", "unknown"))
    except FileNotFoundError as exc:
        logger.warning("Starting in degraded mode: %s", exc)
    yield


app = FastAPI(
    title="Real-Time Fraud Detection API",
    description="Scores PaySim-style transactions for fraud probability.",
    version="0.1.0",
    lifespan=lifespan,
)


def _record(result: dict, amount: float) -> None:
    with _metrics_lock:
        if result.get("scored"):
            _metrics["scored_total"] += 1
        if result.get("is_fraud"):
            _metrics["fraud_total"] += 1
            _metrics["amount_flagged"] += float(amount)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if model.is_loaded else "degraded",
        model_loaded=model.is_loaded,
        algorithm=model.metadata.get("algorithm"),
        trained_at=model.metadata.get("trained_at"),
    )


@app.get("/model")
def model_info() -> dict:
    if not model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model.metadata


@app.get("/metrics")
def metrics() -> Response:
    with _metrics_lock:
        snapshot = dict(_metrics)
    lines = [
        "# HELP fraud_scored_total Transactions scored since startup",
        "# TYPE fraud_scored_total counter",
        f"fraud_scored_total {snapshot['scored_total']}",
        "# HELP fraud_flagged_total Transactions flagged as fraud",
        "# TYPE fraud_flagged_total counter",
        f"fraud_flagged_total {snapshot['fraud_total']}",
        "# HELP fraud_amount_flagged Total amount flagged as fraud",
        "# TYPE fraud_amount_flagged counter",
        f"fraud_amount_flagged {snapshot['amount_flagged']}",
    ]
    return Response("\n".join(lines) + "\n", media_type="text/plain")


def _require_model() -> None:
    if not model.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train it first: fraud-train",
        )


@app.post("/score", response_model=ScoreResponse)
def score(tx: Transaction) -> ScoreResponse:
    _require_model()
    result = model.score_one(tx.model_dump())
    _record(result, tx.amount)
    return ScoreResponse(**result)


@app.post("/score/batch", response_model=BatchScoreResponse)
def score_batch(req: BatchScoreRequest) -> BatchScoreResponse:
    _require_model()
    results = []
    for tx in req.transactions:
        result = model.score_one(tx.model_dump())
        _record(result, tx.amount)
        results.append(ScoreResponse(**result))
    return BatchScoreResponse(results=results)


def main() -> None:  # pragma: no cover - manual entrypoint
    import uvicorn

    uvicorn.run(
        "fraud_platform.serving.app:app",
        host=CONFIG.serving.host,
        port=int(CONFIG.serving.port),
        reload=False,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
