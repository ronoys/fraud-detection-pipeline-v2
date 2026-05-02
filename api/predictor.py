import logging
import os
import random
from collections import deque
from pathlib import Path

import joblib
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
CHAMPION_WEIGHT = float(os.getenv("CHAMPION_WEIGHT", "0.8"))

_champion = None
_challenger = None
_scaler = None

# Rolling window of recent predictions for drift detection (capped at 2000)
prediction_log: deque = deque(maxlen=2000)


def load_model() -> None:
    global _champion, _challenger, _scaler

    champion_path = Path(os.getenv("CHAMPION_MODEL_PATH", "model/artifacts/xgboost.joblib"))
    challenger_path = Path(os.getenv("CHALLENGER_MODEL_PATH", "model/artifacts/random_forest.joblib"))
    scaler_path = Path(os.getenv("SCALER_PATH", "model/artifacts/scaler.joblib"))

    if not champion_path.exists():
        raise FileNotFoundError(
            f"Champion model not found at '{champion_path}'. Run model/train.py first."
        )
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler not found at '{scaler_path}'. Run model/train.py first."
        )

    _champion = joblib.load(champion_path)
    _scaler = joblib.load(scaler_path)
    logger.info("Champion (XGBoost) loaded from %s", champion_path)

    if challenger_path.exists():
        _challenger = joblib.load(challenger_path)
        logger.info("Challenger (RandomForest) loaded from %s", challenger_path)
    else:
        logger.warning("Challenger model not found at '%s' — running champion-only mode", challenger_path)

    warmup = {"Amount": 1.0, **{f"V{i}": 0.0 for i in range(1, 29)}}
    predict(warmup)
    logger.info("Warmup complete — champion-only=%s", _challenger is None)


def predict(features: dict) -> tuple[bool, float, float, str]:
    """
    Returns:
        (is_fraud, confidence, scaled_amount, model_name)
    """
    if _champion is None or _scaler is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    amount_frame = pd.DataFrame({"Amount": [features["Amount"]]})
    scaled_amount = float(_scaler.transform(amount_frame)[0][0])
    model_features = {**features, "Amount": scaled_amount}
    X = pd.DataFrame(
        [[model_features[col] for col in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )

    use_challenger = _challenger is not None and random.random() > CHAMPION_WEIGHT
    model = _challenger if use_challenger else _champion
    model_name = "random_forest" if use_challenger else "xgboost"

    fraud_prob: float = float(model.predict_proba(X)[0][1])
    threshold = float(os.getenv("FRAUD_THRESHOLD", "0.5"))
    is_fraud = fraud_prob >= threshold

    log_entry = {col: model_features[col] for col in FEATURE_COLUMNS}
    log_entry["prediction"] = int(is_fraud)
    log_entry["confidence"] = fraud_prob
    log_entry["model"] = model_name
    prediction_log.append(log_entry)

    return is_fraud, fraud_prob, scaled_amount, model_name
