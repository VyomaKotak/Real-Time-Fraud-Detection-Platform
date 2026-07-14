.PHONY: help install data train test lint format api dashboard report producer stream stream-docker kafka-up kafka-down clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package + dev dependencies (editable)
	pip install -e ".[dev]"
	pre-commit install

data:  ## Download PaySim (or generate synthetic data)
	python scripts/download_data.py

train:  ## Train the fraud-detection model
	fraud-train

test:  ## Run the test suite with coverage
	pytest --cov=fraud_platform --cov-report=term-missing

lint:  ## Lint with ruff + black --check
	ruff check .
	black --check .

format:  ## Auto-format with ruff --fix + black
	ruff check --fix .
	black .

api:  ## Run the FastAPI scoring service
	uvicorn fraud_platform.serving.app:app --host 0.0.0.0 --port 8000 --reload

dashboard:  ## Run the Streamlit dashboard
	streamlit run fraud_platform/dashboard/app.py

report:  ## Build a self-contained HTML results report
	python scripts/build_report.py --out run_outputs/report.html

producer:  ## Replay PaySim into Kafka
	fraud-produce

stream:  ## Start the Spark Structured Streaming scorer (needs local Java 17)
	fraud-stream

stream-docker:  ## Run the whole Kafka + Spark pipeline in Docker (no local Java)
	docker compose -f docker-compose.streaming.yml up --build

kafka-up:  ## Start Kafka + Zookeeper via docker compose
	docker compose up -d zookeeper kafka

kafka-down:  ## Stop the docker compose stack
	docker compose down

clean:  ## Remove generated artifacts (data, models, checkpoints)
	rm -rf checkpoints spark-warehouse mlruns data/scored
	rm -f models/*.joblib models/*.json data/paysim.csv
