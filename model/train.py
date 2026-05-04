"""
Train four fraud detection models on the Kaggle Credit Card Fraud dataset.

Usage:
    python train.py

Output:
    model/artifacts/<model_name>.joblib  — each trained model
    model/artifacts/scaler.joblib        — fitted StandardScaler for Amount
    model/artifacts/training_reference.parquet — sample for drift detection
    model/artifacts/model_metadata.json  — metrics and metadata
    MLflow registry: fraud-detector-champion (XGBoost), fraud-detector-challenger (RandomForest)
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import kagglehub

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

MLFLOW_TRACKING_URI = f"sqlite:///{ARTIFACTS_DIR / 'mlruns.db'}"
EXPERIMENT_NAME = "fraud-detection"

# Performance gate — CI will fail if best model falls below this
MIN_PR_AUC = 0.80
MIN_ROC_AUC = 0.95


def load_data() -> pd.DataFrame:
    logger.info("Downloading dataset via kagglehub...")
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    csv_path = Path(path) / "creditcard.csv"
    logger.info("Loading CSV from %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Dataset shape: %s", df.shape)
    counts = df["Class"].value_counts()
    logger.info("Class distribution — Legit: %d  Fraud: %d", counts[0], counts[1])
    return df


def prepare_split(df: pd.DataFrame):
    # Keep Time in X here so save_training_reference can use it; scale_amount drops it later
    X = df.drop(columns=["Class"])
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    logger.info("Train: %d  Test: %d", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test


def scale_amount(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler()
    X_train = X_train.drop(columns=["Time"], errors="ignore").copy()
    X_test = X_test.drop(columns=["Time"], errors="ignore").copy()
    X_train["Amount"] = scaler.fit_transform(X_train[["Amount"]])
    X_test["Amount"] = scaler.transform(X_test[["Amount"]])
    return X_train, X_test, scaler


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series):
    logger.info("Applying SMOTE to training data...")
    sm = SMOTE(random_state=42)
    X_sm, y_sm = sm.fit_resample(X_train, y_train)
    logger.info(
        "After SMOTE — size: %d  (Legit: %d  Fraud: %d)",
        len(y_sm), (y_sm == 0).sum(), (y_sm == 1).sum(),
    )
    return X_sm, y_sm


def build_models(neg: int, pos: int) -> dict:
    return {
        "logistic_regression": LogisticRegression(
            C=0.01, max_iter=1000, random_state=42
        ),
        "knn": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced"
        ),
        "xgboost": XGBClassifier(
            scale_pos_weight=neg / pos,
            eval_metric="logloss",
            random_state=42,
        ),
    }


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    X_arr = X_test.to_numpy()
    y_pred = model.predict(X_arr)
    y_prob = model.predict_proba(X_arr)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    return {
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_test, y_prob)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
    }


def save_training_reference(X_test_scaled: pd.DataFrame, y_test: pd.Series) -> None:
    """Save a stratified sample of the test set as the drift detection reference."""
    ref = X_test_scaled.copy()
    ref["Class"] = y_test.values

    legit = ref[ref["Class"] == 0].sample(n=min(4500, (ref["Class"] == 0).sum()), random_state=42)
    fraud = ref[ref["Class"] == 1].sample(n=min(500, (ref["Class"] == 1).sum()), random_state=42)
    reference = pd.concat([legit, fraud]).drop(columns=["Class"])

    out = ARTIFACTS_DIR / "training_reference.parquet"
    reference.to_parquet(out, index=False)
    logger.info("Training reference saved to %s (%d rows)", out, len(reference))


def print_table(results: dict) -> None:
    cols = ["roc_auc", "pr_auc", "f1", "precision", "recall", "training_time_seconds"]
    col_w = 16
    name_w = 24
    header = f"{'Model':<{name_w}}" + "".join(f"{c:>{col_w}}" for c in cols)
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for name, r in results.items():
        row = f"{name:<{name_w}}"
        for c in cols:
            val = r.get(c, 0)
            row += f"{val:>{col_w}.4f}" if c != "training_time_seconds" else f"{val:>{col_w - 1}.1f}s"
        print(row)
    print(sep)


def check_performance_gate(metrics: dict, model_name: str) -> None:
    pr_auc = metrics["pr_auc"]
    roc_auc = metrics["roc_auc"]
    if pr_auc < MIN_PR_AUC:
        raise ValueError(
            f"PERFORMANCE GATE FAILED: {model_name} PR-AUC={pr_auc:.4f} < minimum {MIN_PR_AUC}"
        )
    if roc_auc < MIN_ROC_AUC:
        raise ValueError(
            f"PERFORMANCE GATE FAILED: {model_name} ROC-AUC={roc_auc:.4f} < minimum {MIN_ROC_AUC}"
        )
    logger.info(
        "Performance gate passed — %s: PR-AUC=%.4f ROC-AUC=%.4f",
        model_name, pr_auc, roc_auc,
    )


def register_model(run_id: str, registry_name: str, stage: str = "Production") -> None:
    client = mlflow.MlflowClient()
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, registry_name)
    client.transition_model_version_stage(registry_name, mv.version, stage)
    logger.info("Registered %s v%s → %s stage", registry_name, mv.version, stage)


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_data()
    dataset_shape = list(df.shape)
    class_dist = {str(k): int(v) for k, v in df["Class"].value_counts().items()}

    X_train, X_test, y_train, y_test = prepare_split(df)

    neg_train = int((y_train == 0).sum())
    pos_train = int((y_train == 1).sum())
    logger.info(
        "Original train class counts — Legit: %d  Fraud: %d  (scale_pos_weight=%.1f)",
        neg_train, pos_train, neg_train / pos_train,
    )

    X_train_scaled, X_test_scaled, scaler = scale_amount(X_train, X_test)
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.joblib")
    logger.info("Scaler saved to artifacts/scaler.joblib")

    save_training_reference(X_test_scaled, y_test)

    X_sm, y_sm = apply_smote(X_train_scaled, y_train)

    models = build_models(neg_train, pos_train)
    all_results = {}
    run_ids = {}

    for name, model in models.items():
        logger.info("--- Training: %s ---", name)
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params({
                "model_type": name,
                "smote": True,
                "test_size": 0.2,
                "random_state": 42,
                "fraud_threshold": float(os.getenv("FRAUD_THRESHOLD", "0.5")),
            })

            t0 = time.time()
            model.fit(X_sm, y_sm)
            duration = round(time.time() - t0, 2)
            logger.info("%s trained in %.1fs", name, duration)

            metrics = evaluate_model(model, X_test_scaled, y_test)
            metrics["training_time_seconds"] = duration
            all_results[name] = metrics
            run_ids[name] = run.info.run_id

            mlflow.log_metrics({
                "pr_auc": metrics["pr_auc"],
                "roc_auc": metrics["roc_auc"],
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "training_time_seconds": duration,
            })

            if name == "xgboost":
                mlflow.xgboost.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")

            joblib.dump(model, ARTIFACTS_DIR / f"{name}.joblib")
            logger.info(
                "%s → PR-AUC=%.4f  ROC-AUC=%.4f  F1=%.4f",
                name, metrics["pr_auc"], metrics["roc_auc"], metrics["f1"],
            )

    best_name = max(all_results, key=lambda n: all_results[n]["pr_auc"])

    # Performance gate — blocks CI if champion/challenger don't meet minimums
    check_performance_gate(all_results["xgboost"], "xgboost (champion)")
    check_performance_gate(all_results[best_name], f"{best_name} (challenger)")

    # Register XGBoost as champion and best-by-PR-AUC as challenger
    register_model(run_ids["xgboost"], "fraud-detector-champion")
    if best_name != "xgboost":
        register_model(run_ids[best_name], "fraud-detector-challenger")

    metadata = {
        "best_model_name": best_name,
        "champion_model": "xgboost",
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_shape": dataset_shape,
        "class_distribution": class_dist,
        "performance_gates": {"min_pr_auc": MIN_PR_AUC, "min_roc_auc": MIN_ROC_AUC},
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        "models": all_results,
    }
    with open(ARTIFACTS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved to artifacts/model_metadata.json")

    print_table(all_results)

    best = all_results[best_name]
    print(f"\nChampion  : xgboost (PR-AUC={all_results['xgboost']['pr_auc']:.4f}) — deployed at 80% traffic")
    print(f"Challenger: {best_name} (PR-AUC={best['pr_auc']:.4f}) — deployed at 20% traffic")
    print(f"MLflow UI : mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()
