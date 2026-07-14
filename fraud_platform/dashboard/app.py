"""Streamlit dashboard for the Real-Time Fraud Detection Platform.

Two panes:
  * "Live scoring" — send a transaction to the FastAPI service and see the
    fraud probability, plus a couple of preset fraud / legitimate examples.
  * "Streaming monitor" — read the parquet scores written by the Spark job and
    plot fraud volume, amounts flagged, and a table of the latest alerts.

Run:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# ``streamlit run src/dashboard/app.py`` puts this file's directory on sys.path,
# not the project root, so ``import src`` would fail. Add the repo root (two
# levels up from src/dashboard/) so the dashboard runs from anywhere without
# needing PYTHONPATH set.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from fraud_platform.config import CONFIG

API_URL = CONFIG.dashboard.api_url
SCORED_PATH = Path(CONFIG.spark.output_path)

st.set_page_config(page_title="Fraud Detection Platform", layout="wide")
st.title("💳 Real-Time Fraud Detection Platform")

PRESETS = {
    "Fraudulent TRANSFER (account drained)": {
        "type": "TRANSFER",
        "amount": 181000.0,
        "oldbalanceOrg": 181000.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "nameDest": "C1666544295",
    },
    "Legitimate PAYMENT": {
        "type": "PAYMENT",
        "amount": 4200.0,
        "oldbalanceOrg": 25000.0,
        "newbalanceOrig": 20800.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "nameDest": "M1979787155",
    },
    "Legitimate CASH_OUT": {
        "type": "CASH_OUT",
        "amount": 3500.0,
        "oldbalanceOrg": 42000.0,
        "newbalanceOrig": 38500.0,
        "oldbalanceDest": 12000.0,
        "newbalanceDest": 15500.0,
        "nameDest": "C2048537720",
    },
}


def api_health() -> dict:
    try:
        return requests.get(f"{API_URL}/health", timeout=3).json()
    except Exception as exc:  # noqa: BLE001
        return {"status": "unreachable", "error": str(exc)}


tab_live, tab_stream = st.tabs(["🔎 Live scoring", "📈 Streaming monitor"])

with tab_live:
    health = api_health()
    cols = st.columns(3)
    cols[0].metric("API status", health.get("status", "unknown"))
    cols[1].metric("Model loaded", str(health.get("model_loaded", False)))
    cols[2].metric("Algorithm", health.get("algorithm") or "—")

    if health.get("status") == "unreachable":
        st.warning(
            f"Scoring API not reachable at {API_URL}. Start it with "
            "`uvicorn fraud_platform.serving.app:app`."
        )

    st.subheader("Score a transaction")
    preset_name = st.selectbox("Preset", list(PRESETS.keys()))
    preset = PRESETS[preset_name]

    c1, c2, c3 = st.columns(3)
    tx_type = c1.selectbox(
        "type",
        list(CONFIG.features.transaction_types),
        index=list(CONFIG.features.transaction_types).index(preset["type"]),
    )
    amount = c1.number_input("amount", value=float(preset["amount"]), min_value=0.0)
    old_org = c2.number_input(
        "oldbalanceOrg", value=float(preset["oldbalanceOrg"]), min_value=0.0
    )
    new_org = c2.number_input(
        "newbalanceOrig", value=float(preset["newbalanceOrig"]), min_value=0.0
    )
    old_dest = c3.number_input(
        "oldbalanceDest", value=float(preset["oldbalanceDest"]), min_value=0.0
    )
    new_dest = c3.number_input(
        "newbalanceDest", value=float(preset["newbalanceDest"]), min_value=0.0
    )

    if st.button("Score", type="primary"):
        payload = {
            "type": tx_type,
            "amount": amount,
            "oldbalanceOrg": old_org,
            "newbalanceOrig": new_org,
            "oldbalanceDest": old_dest,
            "newbalanceDest": new_dest,
            "nameDest": preset.get("nameDest", ""),
        }
        try:
            resp = requests.post(f"{API_URL}/score", json=payload, timeout=5)
            resp.raise_for_status()
            result = resp.json()
            prob = result["fraud_probability"]
            st.progress(min(prob, 1.0))
            if result["is_fraud"]:
                st.error(f"🚨 FRAUD — probability {prob:.2%}")
            elif not result["scored"]:
                st.info(result.get("reason", "Not scored"))
            else:
                st.success(f"✅ Legitimate — fraud probability {prob:.2%}")
            st.json(result)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Scoring failed: {exc}")

with tab_stream:
    st.subheader("Streaming fraud monitor")
    st.caption(f"Reading Spark output from `{SCORED_PATH}`")

    if not SCORED_PATH.exists() or not any(SCORED_PATH.glob("*.parquet")):
        st.info(
            "No streaming output yet. Start Kafka, run the producer "
            "(`python -m fraud_platform.streaming.producer`) and the Spark job "
            "(`python -m fraud_platform.streaming.spark_stream`)."
        )
    else:
        try:
            df = pd.read_parquet(SCORED_PATH)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read parquet: {exc}")
            df = pd.DataFrame()

        if not df.empty:
            total = len(df)
            flagged = int(df.get("predicted_fraud", pd.Series(dtype=int)).sum())
            amount_flagged = float(
                df.loc[df.get("predicted_fraud", 0) == 1, "amount"].sum()
            )
            m1, m2, m3 = st.columns(3)
            m1.metric("Transactions scored", f"{total:,}")
            m2.metric("Flagged as fraud", f"{flagged:,}")
            m3.metric("Amount flagged", f"{amount_flagged:,.0f}")

            if "step" in df.columns:
                by_step = df.groupby("step")["predicted_fraud"].sum().reset_index()
                fig = px.bar(
                    by_step,
                    x="step",
                    y="predicted_fraud",
                    title="Flagged transactions over time (step = 1 hour)",
                )
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Latest alerts")
            alerts = df[df.get("predicted_fraud", 0) == 1]
            st.dataframe(alerts.tail(50), use_container_width=True)
