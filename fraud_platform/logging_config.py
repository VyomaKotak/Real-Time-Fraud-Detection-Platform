"""Central logging configuration.

Call :func:`setup_logging` once at process start (the CLI entry points do this).
Library modules just use ``logger = logging.getLogger(__name__)`` and log at the
appropriate level; the level is controlled by the ``LOG_LEVEL`` env var.
"""

from __future__ import annotations

import logging
import os

_CONFIGURED = False


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once, idempotently.

    Level resolution: explicit ``level`` arg > ``LOG_LEVEL`` env var > INFO.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    resolved = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=resolved,
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, ensuring logging is configured first."""
    setup_logging()
    return logging.getLogger(name)
