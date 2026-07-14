"""Build a self-contained HTML report from the platform's real run artifacts.

Pulls together the trained model's metadata, the Spark streaming parquet output,
and a couple of live API scores into one standalone HTML file you can open in a
browser — a portable snapshot of what the pipeline produced.

    python scripts/build_report.py --out run_outputs/report.html
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import pandas as pd
import requests

# Allow running as a plain script (``python scripts/build_report.py``).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fraud_platform.config import CONFIG


def _card(title: str, body: str) -> str:
    return f'<div class="card"><h2>{escape(title)}</h2>{body}</div>'


def _kv_table(d: dict) -> str:
    rows = "".join(
        f"<tr><td>{escape(str(k))}</td><td>{escape(str(v))}</td></tr>"
        for k, v in d.items()
    )
    return f"<table>{rows}</table>"


def _model_section() -> str:
    meta_path = Path(CONFIG.model.metadata_path)
    if not meta_path.exists():
        return _card(
            "Model", "<p>No model metadata found — run <code>make train</code>.</p>"
        )
    meta = json.loads(meta_path.read_text())
    m = meta.get("metrics", {})
    metrics = {
        "Algorithm": meta.get("algorithm"),
        "Trained at": meta.get("trained_at"),
        "Train / test rows": f"{meta.get('train_rows'):,} / {meta.get('test_rows'):,}",
        "Fraud rate": f"{meta.get('fraud_rate', 0):.4%}",
        "ROC-AUC": f"{m.get('roc_auc', 0):.4f}",
        "PR-AUC": f"{m.get('pr_auc', 0):.4f}",
        "Fraud precision": f"{m.get('precision_fraud', 0):.4f}",
        "Fraud recall": f"{m.get('recall_fraud', 0):.4f}",
        "Fraud F1": f"{m.get('f1_fraud', 0):.4f}",
    }
    cm = m.get("confusion_matrix")
    cm_html = ""
    if cm:
        cm_html = (
            "<h3>Confusion matrix</h3>"
            "<table class='cm'><tr><th></th><th>Pred 0</th><th>Pred 1</th></tr>"
            f"<tr><th>Actual 0</th><td>{cm[0][0]:,}</td><td>{cm[0][1]:,}</td></tr>"
            f"<tr><th>Actual 1</th><td>{cm[1][0]:,}</td><td>{cm[1][1]:,}</td></tr>"
            "</table>"
        )
    imp = meta.get("feature_importances") or {}
    imp_html = ""
    if imp:
        top = list(imp.items())[:8]
        maxv = max(v for _, v in top) or 1
        bars = "".join(
            f"<div class='barrow'><span class='barlabel'>{escape(k)}</span>"
            f"<span class='bar' style='width:{int(300*v/maxv)}px'></span>"
            f"<span class='barval'>{v:.3f}</span></div>"
            for k, v in top
        )
        imp_html = f"<h3>Top feature importances</h3>{bars}"
    return _card("Model metrics", _kv_table(metrics) + cm_html + imp_html)


def _streaming_section() -> str:
    path = CONFIG.spark.output_path
    if not glob.glob(f"{path}/*.parquet"):
        return _card(
            "Streaming results",
            "<p>No Spark output yet — run the producer + "
            "<code>spark_stream --available-now</code>.</p>",
        )
    df = pd.read_parquet(path)
    total = len(df)
    flagged = int(df["predicted_fraud"].sum())
    actual = int(df["isFraud"].sum())
    tp = int(((df.predicted_fraud == 1) & (df.isFraud == 1)).sum())
    fp = int(((df.predicted_fraud == 1) & (df.isFraud == 0)).sum())
    fn = int(((df.predicted_fraud == 0) & (df.isFraud == 1)).sum())
    amount_flagged = float(df.loc[df.predicted_fraud == 1, "amount"].sum())
    summary = {
        "Transactions scored": f"{total:,}",
        "Predicted fraud": f"{flagged:,}",
        "Actual fraud (label)": f"{actual:,}",
        "True / False positives": f"{tp:,} / {fp:,}",
        "False negatives": f"{fn:,}",
        "Amount flagged": f"{amount_flagged:,.0f}",
    }
    sample = df[df.predicted_fraud == 1][
        ["step", "type", "amount", "nameOrig", "fraud_probability", "isFraud"]
    ].head(10)
    tbl = sample.to_html(index=False, float_format=lambda x: f"{x:.2f}")
    return _card(
        "Streaming results (Kafka → Spark → model)",
        _kv_table(summary) + "<h3>Sample flagged transactions</h3>" + tbl,
    )


def _api_section() -> str:
    base = f"http://localhost:{int(CONFIG.serving.port)}"
    examples = [
        (
            "Fraudulent TRANSFER (account drained)",
            {
                "type": "TRANSFER",
                "amount": 181000,
                "oldbalanceOrg": 181000,
                "newbalanceOrig": 0,
                "oldbalanceDest": 0,
                "newbalanceDest": 0,
                "nameDest": "C1666544295",
            },
        ),
        (
            "Legitimate CASH_OUT",
            {
                "type": "CASH_OUT",
                "amount": 3500,
                "oldbalanceOrg": 42000,
                "newbalanceOrig": 38500,
                "oldbalanceDest": 12000,
                "newbalanceDest": 15500,
                "nameDest": "C2048537720",
            },
        ),
        ("PAYMENT (never scored)", {"type": "PAYMENT", "amount": 4200}),
    ]
    rows = []
    for label, payload in examples:
        try:
            r = requests.post(f"{base}/score", json=payload, timeout=3).json()
            verdict = (
                "🚨 FRAUD"
                if r.get("is_fraud")
                else "— not scored" if not r.get("scored") else "✅ legit"
            )
            rows.append(
                f"<tr><td>{escape(label)}</td>"
                f"<td>{r.get('fraud_probability', 0):.3f}</td>"
                f"<td>{verdict}</td></tr>"
            )
        except Exception as exc:  # API not running
            rows.append(
                f"<tr><td>{escape(label)}</td><td colspan=2>API "
                f"unreachable: {escape(str(exc))}</td></tr>"
            )
    table = (
        "<table><tr><th>Example</th><th>Fraud prob.</th><th>Verdict</th></tr>"
        + "".join(rows)
        + "</table>"
    )
    return _card("Live API scores (POST /score)", table)


def build(out: Path) -> None:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    style = """
    <style>
      body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;
           background:#0f172a;color:#e2e8f0;padding:24px}
      h1{margin:0 0 4px} .sub{color:#94a3b8;margin-bottom:20px}
      .card{background:#1e293b;border:1px solid #334155;border-radius:12px;
            padding:20px;margin-bottom:18px}
      h2{margin:0 0 12px;color:#38bdf8;font-size:18px}
      h3{color:#cbd5e1;font-size:14px;margin:16px 0 8px}
      table{border-collapse:collapse;width:100%;font-size:13px}
      td,th{border:1px solid #334155;padding:6px 10px;text-align:left}
      th{background:#0f172a;color:#94a3b8}
      .cm td{text-align:center;font-weight:bold}
      .barrow{display:flex;align-items:center;gap:8px;margin:3px 0;font-size:12px}
      .barlabel{width:150px;color:#cbd5e1} .barval{color:#94a3b8}
      .bar{height:14px;background:linear-gradient(90deg,#38bdf8,#818cf8);
           border-radius:3px;display:inline-block}
      code{background:#0f172a;padding:1px 5px;border-radius:4px}
    </style>"""
    html = f"""<!doctype html><html><head><meta charset="utf-8">
    <title>Fraud Detection Platform — Results</title>{style}</head><body>
    <h1>💳 Real-Time Fraud Detection Platform</h1>
    <div class="sub">Results report · generated {generated}</div>
    {_model_section()}
    {_streaming_section()}
    {_api_section()}
    <div class="sub">Kafka &rarr; Spark &rarr; FastAPI &rarr; dashboard</div>
    </body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Wrote report -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HTML results report")
    parser.add_argument("--out", default="run_outputs/report.html")
    build(Path(parser.parse_args().out))


if __name__ == "__main__":
    main()
