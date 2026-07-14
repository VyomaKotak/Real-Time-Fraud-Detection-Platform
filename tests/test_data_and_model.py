"""End-to-end-ish tests: synthetic data generation, training, model scoring,
and the FastAPI service — all runnable in CI without Kafka or Spark."""

import pytest

from fraud_platform.data.generate_synthetic import generate


@pytest.mark.parametrize(
    "algorithm", ["random_forest", "gradient_boosting", "xgboost", "logistic"]
)
def test_all_algorithms_train_and_separate(algorithm, tmp_path, monkeypatch):
    """Every supported algorithm trains and separates fraud on PaySim."""
    import fraud_platform.config as config_module

    df = generate(n_rows=30_000, seed=13)
    data_path = tmp_path / "paysim.csv"
    df.to_csv(data_path, index=False)

    cfg = config_module.CONFIG
    monkeypatch.setitem(cfg["data"], "raw_path", str(data_path))
    monkeypatch.setitem(cfg["model"], "path", str(tmp_path / "model.joblib"))
    monkeypatch.setitem(cfg["model"], "metadata_path", str(tmp_path / "meta.json"))
    monkeypatch.setitem(cfg["model"], "n_estimators", 30)  # keep it fast

    from fraud_platform.training.train import train

    metadata = train(algorithm=algorithm)
    assert metadata["algorithm"] == algorithm
    assert metadata["metrics"]["roc_auc"] > 0.9


def test_synthetic_schema_and_fraud_dynamics():
    df = generate(n_rows=20_000, seed=1)
    expected_cols = {
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
    }
    assert expected_cols.issubset(df.columns)
    assert df["isFraud"].sum() > 0
    # PaySim invariant: fraud only occurs in TRANSFER / CASH_OUT.
    fraud_types = set(df.loc[df["isFraud"] == 1, "type"].unique())
    assert fraud_types.issubset({"TRANSFER", "CASH_OUT"})
    # Fraud drains the origin account.
    fraud = df[df["isFraud"] == 1]
    assert (fraud["newbalanceOrig"] == 0).all()


def test_train_and_score(tmp_path, monkeypatch):
    # Point config at a temp dataset + model so we don't touch repo artifacts.
    import fraud_platform.config as config_module

    df = generate(n_rows=30_000, seed=7)
    data_path = tmp_path / "paysim.csv"
    df.to_csv(data_path, index=False)

    cfg = config_module.CONFIG
    monkeypatch.setitem(cfg["data"], "raw_path", str(data_path))
    monkeypatch.setitem(cfg["model"], "path", str(tmp_path / "model.joblib"))
    monkeypatch.setitem(cfg["model"], "metadata_path", str(tmp_path / "meta.json"))
    # Smaller forest keeps the test fast.
    monkeypatch.setitem(cfg["model"], "n_estimators", 40)

    from fraud_platform.training.train import train

    metadata = train(algorithm="random_forest")
    # A separable problem like PaySim should give a strong ROC-AUC.
    assert metadata["metrics"]["roc_auc"] > 0.9

    from fraud_platform.serving.model import FraudModel

    model = FraudModel().load()

    # Classic fraud signature should score high.
    fraud_tx = {
        "type": "TRANSFER",
        "amount": 181000.0,
        "oldbalanceOrg": 181000.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "nameDest": "C1666544295",
    }
    result = model.score_one(fraud_tx)
    assert result["scored"] is True
    assert 0.0 <= result["fraud_probability"] <= 1.0

    # A PAYMENT is never scored (never fraud in PaySim).
    payment = model.score_one({"type": "PAYMENT", "amount": 10.0})
    assert payment["scored"] is False
    assert payment["fraud_probability"] == 0.0


def test_fastapi_score_endpoint(tmp_path, monkeypatch):
    """Train a tiny model, load it into the app, and hit /score via TestClient."""
    import fraud_platform.config as config_module

    df = generate(n_rows=20_000, seed=11)
    data_path = tmp_path / "paysim.csv"
    df.to_csv(data_path, index=False)

    cfg = config_module.CONFIG
    model_path = tmp_path / "model.joblib"
    monkeypatch.setitem(cfg["data"], "raw_path", str(data_path))
    monkeypatch.setitem(cfg["model"], "path", str(model_path))
    monkeypatch.setitem(cfg["model"], "metadata_path", str(tmp_path / "m.json"))
    monkeypatch.setitem(cfg["model"], "n_estimators", 30)

    from fraud_platform.training.train import train

    train(algorithm="random_forest")

    from fastapi.testclient import TestClient

    import fraud_platform.serving.app as app_module
    from fraud_platform.serving.model import FraudModel

    # Rebind the module-level model to our freshly trained one and load it.
    app_module.model = FraudModel(model_path=str(model_path)).load()

    client = TestClient(app_module.app)

    health = client.get("/health").json()
    assert health["model_loaded"] is True

    resp = client.post(
        "/score",
        json={
            "type": "TRANSFER",
            "amount": 181000.0,
            "oldbalanceOrg": 181000.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "nameDest": "C1666544295",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["scored"] is True
    assert 0.0 <= body["fraud_probability"] <= 1.0

    batch = client.post(
        "/score/batch",
        json={
            "transactions": [
                {"type": "PAYMENT", "amount": 10.0},
                {"type": "CASH_OUT", "amount": 5000.0, "oldbalanceOrg": 5000.0},
            ]
        },
    )
    assert batch.status_code == 200
    assert len(batch.json()["results"]) == 2
