# Contributing

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Workflow

- Format and lint before committing (pre-commit does this automatically):
  ```bash
  make format      # ruff --fix + black
  make lint        # ruff check + black --check
  ```
- Run the tests:
  ```bash
  make test        # pytest with coverage
  ```
- Keep the feature-engineering logic in `fraud_platform/features/engineering.py`
  as the single source of truth — training and serving both depend on it, so any
  change there must keep them in sync (the tests guard this).

## Conventions

- Line length 88 (black). Imports sorted by ruff (isort rules).
- Library code logs via `fraud_platform.logging_config.get_logger(__name__)`
  rather than `print`.
- New behaviour ships with a test in `tests/`.

## CI

Every push runs lint (ruff + black) and the test suite on Python 3.10–3.12.
