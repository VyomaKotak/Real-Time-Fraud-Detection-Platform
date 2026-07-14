"""Spark Structured Streaming job: read transactions from Kafka, score them
with the trained model, and write fraud scores back to Kafka + parquet.

Flow:
    Kafka(input_topic) --> parse JSON --> pandas_udf(model) --> score
        --> Kafka(output_topic) (alerts + scores)
        --> parquet sink (data/scored) for the dashboard / offline analysis

The model is loaded **once per executor** inside the pandas UDF via a module
level cache, so the joblib file is not re-read for every micro-batch. Scoring
reuses ``FraudModel`` so it is byte-for-byte identical to the FastAPI service.

Submit:
    spark-submit \
      --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
      src/streaming/spark_stream.py

or simply ``python -m fraud_platform.streaming.spark_stream`` if the kafka package is on
the Spark classpath / auto-fetched.

Note: this module deliberately does NOT use ``from __future__ import
annotations``. PySpark's ``pandas_udf`` infers the UDF eval type from the
function's *runtime* type hints (``pd.Series -> pd.Series``); stringized
annotations would break that inference with an UNSUPPORTED_SIGNATURE error.
"""

import argparse

from fraud_platform.config import CONFIG
from fraud_platform.logging_config import get_logger

logger = get_logger(__name__)

# --- Executor-side model cache ------------------------------------------------
# Populated lazily on each executor the first time the UDF runs there.
_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        from fraud_platform.serving.model import FraudModel

        _MODEL = FraudModel().load()
    return _MODEL


# Spark's Arrow integration (used by pandas_udf) reflectively accesses
# JDK-internal APIs (sun.misc.Unsafe / DirectByteBuffer) that are locked down on
# Java 17+. This is the module-open set Spark itself applies via spark-class;
# we set it explicitly so the job runs under a plain ``python -m`` launch too.
_ADD_OPENS = [
    "java.base/java.lang",
    "java.base/java.lang.invoke",
    "java.base/java.io",
    "java.base/java.net",
    "java.base/java.nio",
    "java.base/java.util",
    "java.base/java.util.concurrent",
    "java.base/sun.nio.ch",
    "java.base/sun.util.calendar",
    "java.base/jdk.internal.ref",
    "java.base/jdk.internal.misc",
]
_ARROW_JAVA_OPTS = " ".join(f"--add-opens={m}=ALL-UNNAMED" for m in _ADD_OPENS)


def _ensure_runtime_java_opts() -> None:
    """Make ``python -m fraud_platform.streaming.spark_stream`` work out of the box.

    Runs *before* the Spark JVM launches and sets two things the job cannot run
    without on a modern JDK:

    * ``JAVA_TOOL_OPTIONS`` — the Arrow ``--add-opens`` flags. We append here
      (rather than via ``.config`` or ``--driver-java-options``) because in
      local mode the driver JVM is already running by the time SparkSession
      config is read, and because the JVM reads this env var raw — no shell
      quoting to mangle the space-separated flags. Existing values (e.g. proxy /
      truststore settings) are preserved.
    * ``PYSPARK_SUBMIT_ARGS`` — the Kafka connector matching the installed
      PySpark version. Respected verbatim if the user already set it.
    """
    import os

    import pyspark

    existing_opts = os.environ.get("JAVA_TOOL_OPTIONS", "")
    if "java.base/java.nio" not in existing_opts:
        os.environ["JAVA_TOOL_OPTIONS"] = f"{existing_opts} {_ARROW_JAVA_OPTS}".strip()

    if not os.environ.get("PYSPARK_SUBMIT_ARGS"):
        package = f"org.apache.spark:spark-sql-kafka-0-10_2.12:{pyspark.__version__}"
        os.environ["PYSPARK_SUBMIT_ARGS"] = f"--packages {package} pyspark-shell"


def _warn_if_unsupported_java() -> None:
    """Warn loudly if the JDK is newer than Spark 3.5 supports (8/11/17).

    Spark 3.5's bundled Arrow Java cannot allocate off-heap memory on Java 18+
    (``DirectByteBuffer.<init>(long, int) not available``), which crashes every
    ``pandas_udf`` — exactly the code path this job depends on. No amount of
    ``--add-opens`` fixes it; the only fix is to run under Java 8/11/17. We
    surface this early with an actionable message instead of a deep stack trace.
    """
    import os
    import re
    import shutil
    import subprocess

    java_home = os.environ.get("JAVA_HOME")
    java_bin = f"{java_home}/bin/java" if java_home else shutil.which("java")
    if not java_bin:
        return
    try:
        out = subprocess.run(
            [java_bin, "-version"], capture_output=True, text=True, timeout=10
        ).stderr
    except Exception:
        return
    match = re.search(r'version "(\d+)', out)
    if not match:
        return
    major = int(match.group(1))
    if major > 17:
        logger.warning(
            "Java %d detected. Spark 3.5 supports Java 8/11/17 only; its Arrow "
            "integration crashes pandas_udf on Java 18+ (DirectByteBuffer "
            "error). Point JAVA_HOME at a JDK 17, e.g. "
            "export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64",
            major,
        )


def build_spark(app_name: str):
    _ensure_runtime_java_opts()
    _warn_if_unsupported_java()
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.streaming.stopGracefullyOnShutdown", "true")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )


def transaction_schema():
    from pyspark.sql.types import (
        DoubleType,
        IntegerType,
        StringType,
        StructField,
        StructType,
    )

    return StructType(
        [
            StructField("step", IntegerType()),
            StructField("type", StringType()),
            StructField("amount", DoubleType()),
            StructField("nameOrig", StringType()),
            StructField("oldbalanceOrg", DoubleType()),
            StructField("newbalanceOrig", DoubleType()),
            StructField("nameDest", StringType()),
            StructField("oldbalanceDest", DoubleType()),
            StructField("newbalanceDest", DoubleType()),
            StructField("isFraud", IntegerType()),
            StructField("isFlaggedFraud", IntegerType()),
        ]
    )


def make_score_udf():
    """A scalar pandas_udf that scores a batch of transactions per call."""
    import pandas as pd
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import DoubleType

    @pandas_udf(DoubleType())
    def score_udf(
        tx_type: pd.Series,
        amount: pd.Series,
        old_org: pd.Series,
        new_org: pd.Series,
        old_dest: pd.Series,
        new_dest: pd.Series,
        name_dest: pd.Series,
    ) -> pd.Series:
        model = _get_model()
        frame = pd.DataFrame(
            {
                "type": tx_type,
                "amount": amount,
                "oldbalanceOrg": old_org,
                "newbalanceOrig": new_org,
                "oldbalanceDest": old_dest,
                "newbalanceDest": new_dest,
                "nameDest": name_dest,
            }
        )
        return model.score_frame(frame).reset_index(drop=True)

    return score_udf


def run(to_console: bool = False, available_now: bool = False) -> None:
    from pyspark.sql.functions import col, from_json, lit, struct, to_json, when

    spark = build_spark(CONFIG.spark.app_name)
    spark.sparkContext.setLogLevel("WARN")

    # available_now: process the whole existing topic backlog once, then stop
    # (great for demos / backfills). Otherwise: tail new messages continuously.
    starting_offsets = "earliest" if available_now else "latest"
    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", CONFIG.kafka.bootstrap_servers)
        .option("subscribe", CONFIG.kafka.input_topic)
        .option("startingOffsets", starting_offsets)
        .option("maxOffsetsPerTrigger", CONFIG.spark.max_offsets_per_trigger)
        .load()
    )

    parsed = raw.select(
        from_json(col("value").cast("string"), transaction_schema()).alias("tx")
    ).select("tx.*")

    threshold = float(CONFIG.model.decision_threshold)
    score_udf = make_score_udf()

    scored = parsed.withColumn(
        "fraud_probability",
        score_udf(
            col("type"),
            col("amount"),
            col("oldbalanceOrg"),
            col("newbalanceOrig"),
            col("oldbalanceDest"),
            col("newbalanceDest"),
            col("nameDest"),
        ),
    ).withColumn(
        "predicted_fraud",
        when(col("fraud_probability") >= lit(threshold), lit(1)).otherwise(lit(0)),
    )

    trigger = (
        {"availableNow": True}
        if available_now
        else {"processingTime": CONFIG.spark.trigger_interval}
    )

    if to_console:
        query = (
            scored.writeStream.format("console")
            .option("truncate", "false")
            .outputMode("append")
            .trigger(**trigger)
            .start()
        )
        query.awaitTermination()
        return

    # Sink 1: publish every score back to Kafka (key = origin account).
    kafka_out = scored.select(
        col("nameOrig").alias("key"),
        to_json(
            struct(
                "step",
                "type",
                "amount",
                "nameOrig",
                "nameDest",
                "fraud_probability",
                "predicted_fraud",
                "isFraud",
            )
        ).alias("value"),
    )
    kafka_query = (
        kafka_out.writeStream.format("kafka")
        .option("kafka.bootstrap.servers", CONFIG.kafka.bootstrap_servers)
        .option("topic", CONFIG.kafka.output_topic)
        .option("checkpointLocation", f"{CONFIG.spark.checkpoint_dir}/kafka")
        .outputMode("append")
        .trigger(**trigger)
        .start()
    )

    # Sink 2: persist scores as parquet for the dashboard / offline analysis.
    parquet_query = (
        scored.writeStream.format("parquet")
        .option("path", CONFIG.spark.output_path)
        .option("checkpointLocation", f"{CONFIG.spark.checkpoint_dir}/parquet")
        .outputMode("append")
        .trigger(**trigger)
        .start()
    )

    logger.info(
        "Streaming: %s -> [%s, %s] | threshold=%s",
        CONFIG.kafka.input_topic,
        CONFIG.kafka.output_topic,
        CONFIG.spark.output_path,
        threshold,
    )
    if available_now:
        # Batch mode: both queries terminate on their own. Await each one so we
        # don't exit (and tear down the second sink mid-write) the moment the
        # first finishes — awaitAnyTermination would return too early here.
        for query in (kafka_query, parquet_query):
            query.awaitTermination()
    else:
        # Continuous mode: queries run forever; block until any one stops/fails.
        spark.streams.awaitAnyTermination()


def main() -> None:
    parser = argparse.ArgumentParser(description="Spark streaming fraud scorer")
    parser.add_argument(
        "--console",
        action="store_true",
        help="Write scores to the console instead of Kafka/parquet sinks",
    )
    parser.add_argument(
        "--available-now",
        action="store_true",
        help="Process the existing topic backlog once (earliest offsets) and "
        "exit, instead of continuously tailing new messages.",
    )
    args = parser.parse_args()
    run(to_console=args.console, available_now=args.available_now)


if __name__ == "__main__":
    main()
