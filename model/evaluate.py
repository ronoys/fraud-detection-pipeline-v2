"""
Evaluate the best trained fraud detection model.

Usage:
    python evaluate.py

Output (saved to model/artifacts/evaluation/):
    roc_curve.png
    pr_curve.png
    confusion_matrix.png
    classification_report.txt
"""

import logging
from pathlib import Path

import joblib
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
EVAL_DIR = ARTIFACTS_DIR / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def load_artifacts():
    model = joblib.load(ARTIFACTS_DIR / "xgboost.joblib")
    scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
    logger.info("Loaded xgboost.joblib and scaler.joblib")
    return model, scaler


def load_test_data(scaler):
    """Reproduce the exact same test split used during training."""
    logger.info("Downloading dataset via kagglehub...")
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    csv_path = Path(path) / "creditcard.csv"
    df = pd.read_csv(csv_path)

    df = df.drop(columns=["Time"])
    X = df.drop(columns=["Class"])
    y = df["Class"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    X_test = X_test.copy()
    X_test["Amount"] = scaler.transform(X_test[["Amount"]])
    logger.info("Test set: %d samples  (Legit: %d  Fraud: %d)",
                len(y_test), (y_test == 0).sum(), (y_test == 1).sum())
    return X_test, y_test


def apply_smote_to_test(X_test: pd.DataFrame, y_test: pd.Series):
    """Oversample the minority class in the test set to 50/50 balance using SMOTE."""
    sm = SMOTE(random_state=42)
    X_bal, y_bal = sm.fit_resample(X_test, y_test)
    logger.info(
        "SMOTE-balanced test set: %d samples  (Legit: %d  Fraud: %d)",
        len(y_bal), (y_bal == 0).sum(), (y_bal == 1).sum(),
    )
    return X_bal, y_bal


def plot_roc_curve(y_test, y_prob, suffix: str = "") -> None:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, color="#2563eb", label=f"ROC AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    title = "ROC Curve — Fraud Detection"
    if suffix:
        title += f" ({suffix})"
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = f"roc_curve{'_' + suffix if suffix else ''}.png"
    out = EVAL_DIR / fname
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_pr_curve(y_test, y_prob, suffix: str = "") -> None:
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    baseline = (y_test == 1).mean()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, lw=2, color="#16a34a", label=f"AP = {ap:.4f}")
    ax.axhline(y=baseline, color="k", linestyle="--", lw=1,
               label=f"Random baseline ({baseline:.4f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    title = "Precision-Recall Curve — Fraud Detection"
    if suffix:
        title += f" ({suffix})"
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = f"pr_curve{'_' + suffix if suffix else ''}.png"
    out = EVAL_DIR / fname
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_confusion_matrix(y_test, y_pred, suffix: str = "") -> None:
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    labels = ["Legit", "Fraud"]

    annot = np.array([
        [f"{cm_norm[i,j]*100:.1f}%\n({cm[i,j]:,})" for j in range(2)]
        for i in range(2)
    ])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_norm,
        annot=False,
        fmt="",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        vmin=0,
        vmax=1,
    )
    for i in range(2):
        for j in range(2):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(
                j + 0.5, i + 0.5, annot[i, j],
                ha="center", va="center",
                fontsize=13, color=color,
            )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    title = "Confusion Matrix — Fraud Detection"
    if suffix:
        title += f" ({suffix})"
    ax.set_title(title, fontsize=14)
    fig.tight_layout()
    fname = f"confusion_matrix{'_' + suffix if suffix else ''}.png"
    out = EVAL_DIR / fname
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)


def _section(title: str) -> None:
    width = 52
    print(f"\n┌{'─' * width}┐")
    print(f"│  {title:<{width - 2}}│")
    print(f"└{'─' * width}┘")


def threshold_analysis(y_test, y_prob) -> None:
    _section("Threshold Analysis")
    print(f"  {'Threshold':>10}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}")
    print(f"  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 10}")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred_t = (y_prob >= t).astype(int)
        p = precision_score(y_test, y_pred_t, zero_division=0)
        r = recall_score(y_test, y_pred_t, zero_division=0)
        f = f1_score(y_test, y_pred_t, zero_division=0)
        print(f"  {t:>10.1f}  {p:>10.4f}  {r:>10.4f}  {f:>10.4f}")


def print_fraud_examples(X_test: pd.DataFrame, y_test: pd.Series, y_pred, y_prob) -> None:
    fraud_idx = (y_test == 1) & (y_pred == 1)
    hits = X_test[fraud_idx].copy()
    hits["fraud_prob"] = y_prob[fraud_idx]
    hits = hits.sort_values("fraud_prob", ascending=False).head(3)

    _section("Top Detected Fraud Examples")
    v_cols = [f"V{i}" for i in range(1, 29)]
    for rank, (_, row) in enumerate(hits.iterrows(), 1):
        print(f"\n  Example {rank}  (fraud_prob = {row['fraud_prob']:.4f})")
        print(f"  {'─' * 32}")
        for col in v_cols:
            print(f"    {col:<6}  {row[col]:>10.4f}")
        print(f"    {'Amount (scaled)':<6}  {row['Amount']:>10.4f}")


def run_evaluation(label: str, suffix: str, model, X_test, y_test) -> None:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    plot_roc_curve(y_test, y_prob, suffix=suffix)
    plot_pr_curve(y_test, y_prob, suffix=suffix)
    plot_confusion_matrix(y_test, y_pred, suffix=suffix)

    report = classification_report(y_test, y_pred, target_names=["Legit", "Fraud"])
    report_file = f"classification_report{'_' + suffix if suffix else ''}.txt"
    (EVAL_DIR / report_file).write_text(report)
    logger.info("Saved %s", report_file)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    _section(f"Confusion Matrix  [{label}]")
    print(f"  {'':>16}  {'Pred Legit':>12}  {'Pred Fraud':>12}")
    print(f"  {'─' * 16}  {'─' * 12}  {'─' * 12}")
    print(f"  {'Actual Legit':>16}  {tn:>12,}  {fp:>12,}")
    print(f"  {'Actual Fraud':>16}  {fn:>12,}  {tp:>12,}")

    _section(f"Classification Report  [{label}]")
    print(report)

    _section(f"Summary Metrics  [{label}]")
    print(f"  ROC-AUC  :  {roc_auc_score(y_test, y_prob):.4f}")
    print(f"  PR-AUC   :  {average_precision_score(y_test, y_prob):.4f}")

    threshold_analysis(y_test, y_prob)
    print_fraud_examples(X_test, y_test, y_pred, y_prob)


def main():
    model, scaler = load_artifacts()
    X_test, y_test = load_test_data(scaler)

    _section("Evaluation on Original (Imbalanced) Test Set")
    run_evaluation("imbalanced", "", model, X_test, y_test)

    X_bal, y_bal = apply_smote_to_test(X_test, y_test)
    _section("Evaluation on SMOTE-Balanced Test Set")
    run_evaluation("smote_balanced", "smote_balanced", model, X_bal, y_bal)

    print(f"\n  Artifacts saved to {EVAL_DIR}\n")


if __name__ == "__main__":
    main()
