"""Configuration loader.

Loads ``config/config.yaml`` and exposes it as nested attribute access, with
environment-variable overrides for the values most likely to change between
environments (Kafka brokers, model path, API host/port, ...).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Repository root = two levels up from this file (src/config.py -> repo/).
ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"

# Map of ENV_VAR -> dotted path inside the config tree.
_ENV_OVERRIDES: dict[str, str] = {
    "DATA_RAW_PATH": "data.raw_path",
    "MODEL_PATH": "model.path",
    "KAFKA_BOOTSTRAP_SERVERS": "kafka.bootstrap_servers",
    "KAFKA_INPUT_TOPIC": "kafka.input_topic",
    "KAFKA_OUTPUT_TOPIC": "kafka.output_topic",
    "SERVING_HOST": "serving.host",
    "SERVING_PORT": "serving.port",
    "DASHBOARD_API_URL": "dashboard.api_url",
    "SPARK_CHECKPOINT_DIR": "spark.checkpoint_dir",
}


class Config(dict):
    """A dict that also supports attribute access and nested ``get``.

    ``cfg.kafka.bootstrap_servers`` and ``cfg["kafka"]["bootstrap_servers"]``
    are equivalent. ``cfg.get_path("kafka.input_topic")`` reads a dotted path.
    """

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[item] = value
        return value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def _set_path(tree: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = tree
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _coerce(value: str) -> Any:
    """Best-effort coercion of env-var strings to int/float/bool."""
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


@lru_cache(maxsize=1)
def load_config(path: str | os.PathLike | None = None) -> Config:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(config_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    for env_var, dotted in _ENV_OVERRIDES.items():
        if env_var in os.environ:
            _set_path(raw, dotted, _coerce(os.environ[env_var]))

    return Config(raw)


# Convenience singleton: ``from fraud_platform.config import CONFIG``.
CONFIG = load_config()
