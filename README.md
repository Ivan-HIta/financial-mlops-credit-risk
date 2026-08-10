# Financial MLOps Credit Risk Platform

End-to-end, synthetic credit-risk MLOps demo. It shows how a financial model moves from
feature engineering to experiment metadata, API serving, monitoring, drift detection and
controlled retraining.

## What it proves

- Reproducible training with time-aware validation and leakage-safe features.
- Model artifact plus explicit model card and threshold decision.
- FastAPI serving contract with health, prediction and monitoring endpoints.
- Population Stability Index (PSI) drift checks and alert thresholds.
- Docker, CI tests, security headers and a retraining command suitable for orchestration.
- Responsible finance: synthetic data, explainability and human review of decisions.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.train --output artifacts
uvicorn src.api:app --reload
pytest -q
```

Open `http://localhost:8000/docs` for the API contract.

## Demo narrative

“I can ship a risk model as a governed service: the training run is reproducible, the
threshold is documented, predictions are observable and drift creates an explicit action
instead of silently degrading decisions.”

This project does not use real customer, bank or employer data.

