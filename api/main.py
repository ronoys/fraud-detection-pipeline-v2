import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from predictor import FEATURE_COLUMNS, load_model, predict, prediction_log
from schemas import AlertEvent, DriftResponse, PipelineStep, PredictionResponse, TransactionRequest

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading models...")
    load_model()
    logger.info("Models ready.")
    yield


app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection with champion/challenger model routing.",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

fraud_counter = Counter("fraud_predictions_total", "Fraud prediction counts", ["result", "model"])
confidence_histogram = Histogram(
    "fraud_confidence_score",
    "Fraud confidence distribution",
    ["model"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
model_routing_counter = Counter("model_routing_total", "Requests routed per model", ["model"])

_alerts: list[AlertEvent] = []


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["ops"])
def health():
    from predictor import _challenger
    return {
        "status": "ok",
        "champion": "xgboost",
        "challenger": "random_forest" if _challenger is not None else None,
        "predictions_logged": len(prediction_log),
    }


@app.get("/alerts", response_model=list[AlertEvent], tags=["ops"])
def alerts():
    return _alerts[-25:]


@app.get("/model-stats", tags=["ops"])
def model_stats():
    log = list(prediction_log)
    champion_preds = [p for p in log if p["model"] == "xgboost"]
    challenger_preds = [p for p in log if p["model"] == "random_forest"]

    def summarize(preds: list, name: str, weight: float) -> dict:
        if not preds:
            return {"name": name, "traffic_weight": weight, "prediction_count": 0}
        return {
            "name": name,
            "traffic_weight": weight,
            "prediction_count": len(preds),
            "fraud_rate": round(sum(p["prediction"] for p in preds) / len(preds), 4),
            "avg_confidence": round(sum(p["confidence"] for p in preds) / len(preds), 4),
        }

    from predictor import CHAMPION_WEIGHT
    return {
        "champion": summarize(champion_preds, "xgboost", CHAMPION_WEIGHT),
        "challenger": summarize(challenger_preds, "random_forest", 1 - CHAMPION_WEIGHT),
        "total_predictions": len(log),
    }


@app.get("/drift", response_model=DriftResponse, tags=["ops"])
def drift_report():
    log = list(prediction_log)
    if len(log) < 50:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 50 predictions for drift analysis, have {len(log)}.",
        )

    reference_path = Path(os.getenv("TRAINING_REFERENCE_PATH", "model/artifacts/training_reference.parquet"))
    if not reference_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Training reference not found. Run model/train.py to generate it.",
        )

    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
    except ImportError:
        raise HTTPException(status_code=501, detail="evidently not installed")

    reference = pd.read_parquet(reference_path)
    current = pd.DataFrame(log)[FEATURE_COLUMNS]

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    result = report.as_dict()

    drift_result = result["metrics"][0]["result"]
    feature_scores = {
        col: round(info["drift_score"], 4)
        for col, info in drift_result["drift_by_columns"].items()
    }

    return DriftResponse(
        drift_detected=drift_result["dataset_drift"],
        drifted_features=drift_result["number_of_drifted_columns"],
        total_features=drift_result["number_of_columns"],
        share_drifted=round(drift_result["share_of_drifted_columns"], 4),
        feature_drift_scores=feature_scores,
        predictions_analyzed=len(log),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict_fraud(transaction: TransactionRequest) -> PredictionResponse:
    transaction_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        is_fraud, confidence, scaled_amount, model_name = predict(transaction.model_dump())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    threshold = float(os.getenv("FRAUD_THRESHOLD", "0.5"))
    model_label = "XGBoost (champion)" if model_name == "xgboost" else "RandomForest (challenger)"

    pipeline_steps = [
        PipelineStep(
            name="Transaction received",
            detail=f"Amount=${transaction.Amount:.2f}, 28 PCA features",
        ),
        PipelineStep(
            name="Amount normalized",
            detail=f"${transaction.Amount:.2f} → {scaled_amount:.4f} (StandardScaler)",
        ),
        PipelineStep(
            name=f"{model_label} inference",
            detail=f"predict_proba returned fraud probability = {confidence:.4f}",
        ),
        PipelineStep(
            name="Threshold applied",
            detail=f"{confidence:.4f} {'≥' if is_fraud else '<'} {threshold} → {'FRAUD' if is_fraud else 'LEGITIMATE'}",
        ),
    ]

    response = PredictionResponse(
        fraud=is_fraud,
        confidence=confidence,
        transaction_id=transaction_id,
        timestamp=timestamp,
        pipeline_steps=pipeline_steps,
        model_used=model_name,
    )

    fraud_counter.labels(result="fraud" if is_fraud else "legit", model=model_name).inc()
    confidence_histogram.labels(model=model_name).observe(confidence)
    model_routing_counter.labels(model=model_name).inc()

    if is_fraud:
        alert = AlertEvent(
            transaction_id=transaction_id,
            amount=transaction.Amount,
            confidence=confidence,
            timestamp=timestamp,
        )
        _alerts.append(alert)
        logger.warning(
            "Fraud alert transaction_id=%s amount=%.2f confidence=%.4f model=%s",
            transaction_id, transaction.Amount, confidence, model_name,
        )

    return response
