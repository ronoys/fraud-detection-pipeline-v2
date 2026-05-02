"""
Read model_metadata.json and fail if champion or challenger miss performance gates.
Used by CI to block deploys on model regression.
"""

import json
import sys
from pathlib import Path

# F1 is excluded: it's threshold-dependent and misleading for severely imbalanced data.
# PR-AUC and ROC-AUC are threshold-independent and better suited as regression gates.
GATES = {
    "pr_auc": 0.80,
    "roc_auc": 0.95,
}

metadata_path = Path("model/artifacts/model_metadata.json")
if not metadata_path.exists():
    print("ERROR: model/artifacts/model_metadata.json not found.")
    print("Run model/train.py to generate artifacts, then commit model_metadata.json.")
    sys.exit(1)

with open(metadata_path) as f:
    metadata = json.load(f)

models = metadata.get("models", {})
champion = metadata.get("champion_model", "xgboost")
challenger = metadata.get("best_model_name", champion)

check_models = {champion, challenger}
failed = False

for model_name in check_models:
    if model_name not in models:
        print(f"ERROR: model '{model_name}' not found in metadata.")
        sys.exit(1)

    metrics = models[model_name]
    role = "champion" if model_name == champion else "challenger"
    print(f"\n--- {model_name} ({role}) ---")

    for metric, threshold in GATES.items():
        value = metrics.get(metric, 0.0)
        status = "PASS" if value >= threshold else "FAIL"
        print(f"  {metric}: {value:.4f} (min={threshold}) [{status}]")
        if value < threshold:
            failed = True

if failed:
    print("\nPERFORMANCE GATE FAILED — blocking deploy.")
    sys.exit(1)

print("\nAll performance gates passed.")
