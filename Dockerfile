# Pinned to bookworm: Spark 3.5 supports Java 8/11/17 only, and bookworm ships
# openjdk-17 (Debian trixie dropped it, and its default-jre is Java 21, which
# crashes Spark's Arrow / pandas_udf).
FROM python:3.11-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

WORKDIR /app

# Install dependencies first (better layer caching) using only the project
# metadata, then the source. SETUPTOOLS_USE_DISTUTILS=stdlib works around a
# PySpark sdist build failure on Debian images (setuptools install_layout error).
COPY pyproject.toml README.md LICENSE ./
COPY fraud_platform ./fraud_platform
RUN SETUPTOOLS_USE_DISTUTILS=stdlib pip install --no-cache-dir .

# Scripts (data download, pipeline runner) live outside the package.
COPY scripts ./scripts
COPY config ./config

EXPOSE 8000 8501

# Default: the scoring API. Override `command` in docker-compose for other roles.
CMD ["uvicorn", "fraud_platform.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
