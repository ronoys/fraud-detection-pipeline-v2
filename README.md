# Fraud Detection — MLOps Pipeline

Real-time credit card fraud detection system built for MSML605. The focus is the **pipeline architecture**, not the model: champion/challenger live traffic routing, MLflow experiment tracking with performance-gated model registration, Evidently drift detection, a 5-job CI/CD pipeline with automated retraining, and production deployment on Fly.io.

Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) — 284,807 transactions, 492 fraud (0.17%).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  GitHub Actions CI/CD                                                       │
│                                                                             │
│  On push/PR:   lint → test (35 tests) → validate-model → build → deploy    │
│  On schedule:  retrain.yml (weekly) → gate check → commit metadata → deploy │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼ Fly.io (production)
┌─────────────────────────────────────────────────────────────────────────────┐
│  Streamlit Frontend  (:8501)                                                │
│  5 pre-loaded transactions (3 legit / 2 fraud) with pipeline trace UI      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │  HTTP POST /predict
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  FastAPI Inference Service  (:8000)                                         │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Champion / Challenger Router                                        │   │
│  │  ┌──────────────────────────┐    ┌──────────────────────────────┐   │   │
│  │  │  XGBoost  (champion)     │    │  RandomForest  (challenger)   │   │   │
│  │  │  80% traffic             │    │  20% traffic                  │   │   │
│  │  │  PR-AUC: 0.860           │    │  PR-AUC: 0.874               │   │   │
│  │  └──────────────────────────┘    └──────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Rolling prediction log (2000 cap) ──► /drift  (Evidently DataDriftPreset) │
│  /model-stats: live fraud rate + avg confidence per model                   │
│  /metrics ──► Prometheus ──► Grafana (:3000)                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  MLflow Model Registry  (:5000)                                             │
│  fraud-detector-champion   → Production stage  (XGBoost)                   │
│  fraud-detector-challenger → Production stage  (RandomForest)              │
│  All runs logged: PR-AUC, ROC-AUC, F1, precision, recall, training time    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Design

### Why champion/challenger instead of atomic model swap

Most deployments replace the model entirely on retrain. This project routes 80% of live traffic to the champion (XGBoost) and 20% to the challenger (RandomForest). `/model-stats` compares their live fraud rates and confidence distributions in real time. This means a regression in the challenger is visible before it would ever affect the majority of traffic — the standard approach gives you no such signal until it's too late.

### Why the performance gate runs in two places

`train.py` raises `ValueError` and aborts if either model falls below `PR-AUC ≥ 0.80` / `ROC-AUC ≥ 0.95`. The CI's `validate-model` job independently reads `model_metadata.json` and applies the same gate. This means a committed metadata file with degraded metrics blocks the deploy even if training passed — the two layers catch different failure modes (training-time regression vs. accidental metadata overwrite).

F1 is intentionally excluded from gates: at 0.17% fraud prevalence, F1 is dominated by threshold choice and misleads. PR-AUC is threshold-independent.

### Why drift detection uses a saved reference sample

The `/drift` endpoint runs Evidently's `DataDriftPreset` against `training_reference.parquet` — a 5,000-row stratified sample saved at train time. The alternative (re-downloading and re-splitting the dataset at inference time) would couple the serving layer to training infrastructure. The reference file is a stable artifact that travels with the model.

### Why the retrain workflow commits metadata but not model files

`*.joblib` files are gitignored (50–200 MB). The retrain workflow uploads them as GitHub Actions artifacts (30-day retention), commits only `model_metadata.json` and `training_reference.parquet`, and then calls `fly deploy` using the freshly trained artifacts in the workspace. The next CI run reads the committed metadata for gate validation.

---

## Model Training

**Four models trained and compared on every run:**

| Model | PR-AUC | ROC-AUC | F1 | Training Time |
|---|---|---|---|---|
| XGBoost | 0.860 | 0.970 | 0.595 | ~1.4s |
| **RandomForest** | **0.874** | **0.968** | **0.839** | ~46s |
| LogisticRegression | 0.715 | 0.974 | 0.745 | ~2s |
| KNN | 0.610 | 0.934 | 0.819 | ~0.04s |

**XGBoost is the champion despite lower PR-AUC** because it trains ~33× faster, enabling practical weekly retraining. RandomForest is the challenger — its higher recall makes it worth monitoring against XGBoost in live traffic.

**Preprocessing pipeline:**
1. Stratified 80/20 train/test split
2. Drop `Time`, scale `Amount` with `StandardScaler` (fit on train only)
3. SMOTE applied to training set only — test set reflects real-world 0.17% imbalance
4. Each model evaluated on the imbalanced test set (metric that matters in production)

**MLflow tracking:** every run logs params, metrics, and the model artifact. XGBoost is registered as `fraud-detector-champion` and the best-by-PR-AUC model as `fraud-detector-challenger`, both promoted to Production stage automatically.

```bash
python model/train.py
mlflow ui --backend-store-uri sqlite:///model/artifacts/mlruns.db   # → localhost:5000
```

---

## CI/CD Pipeline

### On every push to `main` or PR

```
lint (ruff)
    │
    ├── test (pytest, 35 tests, no real models required)
    │
    └── validate-model (.github/scripts/validate_gates.py)
              │
              └── build-and-push (GHCR, tags: sha-*, branch, latest)
                        │
                        └── deploy (Fly.io, main branch only)
```

The `test` and `validate-model` jobs run in parallel after `lint`. Both must pass before `build-and-push` starts. The `deploy` job only runs on direct pushes to `main` (not PRs), and requires the `production` GitHub environment for approval gating.

### Weekly retrain (`.github/workflows/retrain.yml`)

Triggered every Sunday at 02:00 UTC or manually via `workflow_dispatch` (with configurable gate thresholds as inputs):

1. Downloads Kaggle dataset (requires `KAGGLE_USERNAME` / `KAGGLE_KEY` secrets)
2. Runs `python model/train.py` — gates are enforced inside; workflow fails if they're not met
3. Uploads model artifacts as GitHub Actions artifacts (30-day retention)
4. Commits updated `model_metadata.json` to `main`
5. Calls `fly deploy` with the freshly trained artifacts

### Required GitHub secrets

| Secret | Where used |
|---|---|
| `FLY_API_TOKEN` | Both workflows — Fly.io deploy |
| `KAGGLE_USERNAME` | `retrain.yml` — dataset download |
| `KAGGLE_KEY` | `retrain.yml` — dataset download |

---

## Local Setup

```bash
cp .env.example .env
docker-compose -f infra/docker-compose.yml up --build
```

| Service | URL | Notes |
|---|---|---|
| Frontend | http://localhost:8501 | Streamlit demo |
| API | http://localhost:8000 | FastAPI + auto docs at `/docs` |
| MLflow | http://localhost:5000 | Experiment tracking UI |
| Prometheus | http://localhost:9090 | Raw metrics |
| Grafana | http://localhost:3000 | Dashboards (admin / admin) |

**Model artifacts must exist before starting the stack.** Run training first if `model/artifacts/` is empty:

```bash
pip install -r model/requirements.txt
python model/train.py
```

---

## API Reference

### `POST /predict`

Scores a transaction. Routes 80% to XGBoost, 20% to RandomForest.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 1000.0,
    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
    "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
    "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
    "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
    "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
    "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
    "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02,
    "Amount": 149.62
  }'
```

Response:
```json
{
  "fraud": true,
  "confidence": 0.9863,
  "transaction_id": "a1b2c3d4-...",
  "timestamp": "2026-05-02T17:00:00+00:00",
  "model_used": "xgboost",
  "pipeline_steps": [
    { "name": "Transaction received", "detail": "Amount=$149.62, 28 PCA features" },
    { "name": "Amount normalized",    "detail": "$149.62 → 0.2441 (StandardScaler)" },
    { "name": "XGBoost (champion) inference", "detail": "predict_proba returned fraud probability = 0.9863" },
    { "name": "Threshold applied",    "detail": "0.9863 ≥ 0.5 → FRAUD" }
  ]
}
```

### `GET /model-stats`

Live champion vs. challenger comparison since last restart.

```json
{
  "champion":   { "name": "xgboost",       "traffic_weight": 0.8, "prediction_count": 412, "fraud_rate": 0.031, "avg_confidence": 0.142 },
  "challenger": { "name": "random_forest", "traffic_weight": 0.2, "prediction_count": 108, "fraud_rate": 0.028, "avg_confidence": 0.138 },
  "total_predictions": 520
}
```

### `GET /drift`

Runs Evidently `DataDriftPreset` on the last ≤2000 predictions vs. the training reference. Requires at least 50 predictions.

```json
{
  "drift_detected": false,
  "drifted_features": 2,
  "total_features": 29,
  "share_drifted": 0.069,
  "feature_drift_scores": { "Amount": 0.031, "V1": 0.018, "V14": 0.091 },
  "predictions_analyzed": 847
}
```

### Other endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Returns status, champion/challenger names, prediction count |
| `GET /alerts` | Last 25 fraud alerts (in-memory, resets on restart) |
| `POST /notify` | Sends SMS via AWS SNS |
| `GET /metrics` | Prometheus metrics (scraped every 15s) |

---

## Prometheus Metrics

Custom metrics exposed at `/metrics`, all labeled by model:

| Metric | Type | Labels |
|---|---|---|
| `fraud_predictions_total` | Counter | `result` (fraud/legit), `model` |
| `fraud_confidence_score` | Histogram | `model` |
| `model_routing_total` | Counter | `model` |

---

## Tests

```bash
pip install -r api/requirements.txt pytest pytest-cov httpx
pytest tests/ -v
```

35 tests across three files. No model files required — the test fixtures patch `main.load_model` at the import boundary and inject mock models directly into the predictor module singletons.

| File | Coverage |
|---|---|
| `tests/test_api.py` | All endpoints, alert creation, drift gating, 422 validation |
| `tests/test_predictor.py` | Routing logic, thresholding, amount scaling, prediction logging |
| `tests/test_schemas.py` | Pydantic validation — required fields, bounds, negative amounts |

---

## Load Testing

```bash
k6 run load-testing/script.js                              # local
BASE_URL=https://fraud-detection-api.fly.dev k6 run load-testing/script.js  # production
```

Stages: 30s ramp to 10 VUs → 60s hold at 50 VUs → 15s ramp down.
Thresholds: P95 < 500ms, error rate < 1%.

---

## Production Deployment (Fly.io)

Initial setup (one-time):

```bash
fly auth login
fly apps create fraud-detection-api
fly secrets set FLY_API_TOKEN=...
```

Manual deploy:

```bash
python model/train.py          # generate artifacts
fly deploy                     # builds image, bakes in artifacts, deploys
```

After setup, every push to `main` deploys automatically via GitHub Actions. The `fly.toml` config sets 1GB shared VM, HTTPS enforced, health check on `/health`, auto-stop when idle.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CHAMPION_MODEL_PATH` | `model/artifacts/xgboost.joblib` | Champion model |
| `CHALLENGER_MODEL_PATH` | `model/artifacts/random_forest.joblib` | Challenger model |
| `SCALER_PATH` | `model/artifacts/scaler.joblib` | StandardScaler for Amount |
| `TRAINING_REFERENCE_PATH` | `model/artifacts/training_reference.parquet` | Drift baseline |
| `FRAUD_THRESHOLD` | `0.5` | Classification threshold |
| `CHAMPION_WEIGHT` | `0.8` | Fraction of traffic to champion |

---

## Repository Structure

```
fraud-detection/
├── api/                    # FastAPI inference service
│   ├── main.py             # Endpoints, Prometheus metrics, drift + stats routes
│   ├── predictor.py        # Model loading, champion/challenger routing, prediction log
│   └── schemas.py          # Pydantic request/response models
├── model/
│   ├── train.py            # 4-model training, MLflow logging, performance gates, artifact export
│   ├── evaluate.py         # ROC/PR curves, confusion matrices, threshold sweep
│   ├── extract_examples.py # Extracts hardcoded examples for frontend
│   └── artifacts/          # joblib models, scaler, training_reference.parquet, mlruns/
├── frontend/
│   └── app.py              # Streamlit transaction scoring demo
├── tests/
│   ├── conftest.py         # Fixtures — mock model injection, state isolation
│   ├── test_api.py         # Endpoint tests
│   ├── test_predictor.py   # Predictor unit tests
│   └── test_schemas.py     # Schema validation tests
├── infra/
│   ├── docker-compose.yml  # Local stack (api, frontend, mlflow, prometheus, grafana)
│   ├── Dockerfile          # API image (Python 3.11, bakes in model artifacts)
│   └── frontend.Dockerfile
├── .github/
│   ├── workflows/ci.yml         # 5-job pipeline: lint→test→validate→build→deploy
│   ├── workflows/retrain.yml    # Weekly retrain + gate check + deploy
│   └── scripts/validate_gates.py
├── monitoring/
│   └── prometheus.yml
├── load-testing/
│   └── script.js           # k6 ramp + threshold config
├── fly.toml                # Fly.io production config
└── pyproject.toml          # pytest + ruff config
```
