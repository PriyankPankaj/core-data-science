"""
Phase 5 (continued): Threshold optimization.

The default 0.5 classification threshold isn't necessarily optimal for a
business context where missing a churner (false negative) may be more
costly than a false alarm (false positive). This script systematically
compares thresholds and reports the real trade-offs.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score
import json

RESULTS_DIR = "results"
THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]


if __name__ == "__main__":
    model = joblib.load(f"{RESULTS_DIR}/best_model.pkl")
    X_test = pd.read_csv(f"{RESULTS_DIR}/X_test.csv")
    y_test = pd.read_csv(f"{RESULTS_DIR}/y_test.csv").squeeze()

    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"Total test customers: {len(y_test)}")
    print(f"Actual churners in test set: {y_test.sum()}")
    print(f"\n{'Threshold':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Predicted Churn Count':<22}")
    print("-" * 65)

    results = []
    for t in THRESHOLDS:
        y_pred = (y_proba >= t).astype(int)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        predicted_churn_count = int(y_pred.sum())

        print(f"{t:<10} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {predicted_churn_count:<22}")

        results.append({
            "threshold": t,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "predicted_churn_count": predicted_churn_count,
        })

    # Find the threshold that maximizes F1 (a reasonable default recommendation,
    # though the actual best choice depends on real business cost of FP vs FN)
    best_f1_result = max(results, key=lambda r: r["f1"])

    print(f"\nThreshold maximizing F1: {best_f1_result['threshold']} "
          f"(F1={best_f1_result['f1']:.4f})")

    print("\n=== Business trade-off interpretation ===")
    print("Lower threshold (e.g. 0.30): catches more actual churners (higher "
          "recall) but flags more false positives, increasing retention-campaign "
          "costs spent on customers who wouldn't have churned anyway.")
    print("Higher threshold (e.g. 0.70): fewer false alarms (higher precision) "
          "but misses more real churners (lower recall), representing lost "
          "revenue from unaddressed churn.")
    print("The right choice depends on the relative cost of a missed churner "
          "vs. a wasted retention offer, which is a business decision, not a "
          "purely statistical one.")

    with open(f"{RESULTS_DIR}/threshold_optimization.json", "w") as f:
        json.dump({
            "results_by_threshold": results,
            "best_f1_threshold": best_f1_result["threshold"],
        }, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}/threshold_optimization.json")