# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Contribution Policy

Never add Claude as a co-author or contributor in commit messages, PR descriptions, comments, or anywhere else in the codebase.

## Project Overview

MLOps demo for real-time credit card fraud detection. Architecture: Streamlit frontend → FastAPI backend → XGBoost/RandomForest model, with Prometheus + Grafana observability.

Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) — 284,807 transactions, 492 fraud (0.17% imbalance). Features are V1–V28 (PCA-anonymized) + Amount + Time.

## Commands

### Full local stack
```bash
cp .env.example .env
docker-compose -f infra/docker-compose.yml up --build
```
Services: Frontend `localhost:8501`, API `localhost:8000` (docs at `/docs`), Prometheus `localhost:9090`, Grafana `localhost:3000` (admin/admin).

### Model training & evaluation
```bash
pip install -r model/requirements.txt
python model/train.py        # trains LR, KNN, RF, XGB; logs to MLflow; saves artifacts to model/artifacts/
python model/evaluate.py     # generates ROC/PR curves, confusion matrices, classification reports
python model/extract_examples.py  # extracts hardcoded examples for the Streamlit frontend
mlflow ui --backend-store-uri sqlite:///model/artifacts/mlruns.db  # view experiment tracking UI at localhost:5000
```

### Linting (used in CI)
```bash
pip install ruff
ruff check api/
```

### Load testing
```bash
k6 run load-testing/script.js
BASE_URL=https://fraud-detection-api.fly.dev k6 run load-testing/script.js
```

### Production deploy (Fly.io)
```bash
python model/train.py                          # generate artifacts first
fly deploy                                     # API
fly deploy --config fly.frontend.toml          # Frontend
```

## Architecture

### API (`api/`)
- `main.py` — FastAPI app. Endpoints: `GET /health`, `POST /predict`, `GET /alerts`, `GET /model-stats` (champion vs. challenger metrics), `GET /drift` (Evidently drift report vs. training baseline), `GET /metrics` (Prometheus).
- `predictor.py` — Loads champion (XGBoost, 80% traffic) and challenger (RandomForest, 20%) at startup. `predict()` scales Amount, routes randomly by weight, runs `predict_proba()`, logs features + result to `prediction_log` deque (maxlen=2000) for drift detection. Returns `(is_fraud, confidence, scaled_amount, model_name)`.
- `schemas.py` — Pydantic models: `TransactionRequest`, `PredictionResponse` (includes `model_used` field), `AlertEvent`, `DriftResponse`.

### Model (`model/`)
- `train.py` — Full pipeline: stratified 80/20 split, drop Time + StandardScaler on Amount in `scale_amount()`, SMOTE on training data, trains 4 models. Logs every run to MLflow (`model/artifacts/mlruns.db`), enforces performance gates (PR-AUC ≥ 0.80, ROC-AUC ≥ 0.95), registers XGBoost as `fraud-detector-champion` and best-by-PR-AUC as `fraud-detector-challenger` in Production stage. Saves `training_reference.parquet` (5000-row stratified sample) for drift detection.
- `evaluate.py` — Evaluates on original imbalanced test set AND SMOTE-balanced set; threshold sweep across [0.3, 0.4, 0.5, 0.6, 0.7].
- `artifacts/` — `xgboost.joblib` (champion), `random_forest.joblib` (challenger), `scaler.joblib`, `training_reference.parquet`, `mlruns.db` (MLflow SQLite store).

### Frontend (`frontend/app.py`)
Streamlit dashboard with 5 hardcoded example transactions (3 legit, 2 fraud). Displays pipeline trace, model routing (champion/challenger), confidence, and risk level.

### Infrastructure (`infra/`)
- Docker Compose: api (Python 3.11), frontend (Python 3.9), mlflow UI (port 5000), prometheus, grafana.
- `infra/Dockerfile` — API image for both local Compose and Fly.io production deploy.
- Prometheus scrapes `api:8000/metrics` every 15s. Metrics include per-model fraud counters, confidence histograms, and routing counts.

## Key Design Decisions

- **Champion/challenger routing** — XGBoost (champion, 80% traffic) vs. RandomForest (challenger, 20%). Weights configurable via `CHAMPION_WEIGHT` env var. `/model-stats` compares live fraud rates and confidence per model.
- **Performance gates in training** — `train.py` raises `ValueError` if XGBoost or best model falls below `MIN_PR_AUC=0.80` / `MIN_ROC_AUC=0.95`. CI fails fast on model regression.
- **MLflow registry** — All runs logged to `model/artifacts/mlruns.db` (SQLite). Champion registered as `fraud-detector-champion`, challenger as `fraud-detector-challenger`, both in Production stage. MLflow UI available at port 5000 in Docker Compose.
- **Drift detection** — `training_reference.parquet` (5000-row stratified sample) saved at train time. `/drift` endpoint runs Evidently `DataDriftPreset` against the rolling `prediction_log` deque. Requires ≥50 predictions.
- **Fraud threshold** — Default 0.5, configurable via `FRAUD_THRESHOLD` env var.
- **In-memory alerts** — The `/alerts` and `prediction_log` stores reset on container restart.
- **SMOTE on training only** — Oversampling applied after train/test split so the test set reflects real-world imbalance.
- **Feature set** — `Time` is kept through `prepare_split` but dropped in `scale_amount` before model training. Amount is scaled; V1–V28 passed raw (already PCA-transformed). API feature columns: V1–V28 + Amount (29 features).

## CI

`.github/workflows/ci.yml` runs on push/PR to main:
1. `ruff check api/` for linting
2. `pytest tests/ -v` — 35 tests, no real model files required
3. `validate_gates.py` — reads `model_metadata.json`, fails if PR-AUC or ROC-AUC below threshold
4. Docker build + push to GHCR
5. `fly deploy` (main branch only, requires `production` environment approval)
