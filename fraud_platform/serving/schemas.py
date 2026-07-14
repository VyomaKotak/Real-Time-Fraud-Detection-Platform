"""Pydantic request/response models for the scoring API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    """A single PaySim-style transaction to score."""

    type: str = Field(..., description="PAYMENT|TRANSFER|CASH_OUT|CASH_IN|DEBIT")
    amount: float = Field(..., ge=0)
    oldbalanceOrg: float = Field(0.0, ge=0)
    newbalanceOrig: float = Field(0.0, ge=0)
    oldbalanceDest: float = Field(0.0, ge=0)
    newbalanceDest: float = Field(0.0, ge=0)
    nameOrig: str | None = None
    nameDest: str | None = None
    step: int | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "type": "TRANSFER",
                "amount": 181000.0,
                "oldbalanceOrg": 181000.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
                "nameOrig": "C1231006815",
                "nameDest": "C1666544295",
                "step": 1,
            }
        }
    }


class ScoreResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold: float
    scored: bool
    reason: str | None = None


class BatchScoreRequest(BaseModel):
    transactions: list[Transaction]


class BatchScoreResponse(BaseModel):
    results: list[ScoreResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    algorithm: str | None = None
    trained_at: str | None = None
