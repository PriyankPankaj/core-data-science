"""
Phase 13: Model comparison.

Pulls REAL, already-measured test metrics from each model's saved JSON
results (Phases 6-11) — nothing recomputed or placeholder here.
"""
import json
import pandas as pd

RESULTS_DIR = "results"


def load(name):
    with open(f"{RESULTS_DIR}/{name}") as f:
        return json.load(f)


if __name__ == "__main__":
    logreg = load("logistic_regression_results.json")
    svm = load("svm_results.json")
    rf = load("random_forest_results.json")
    xgb = load("xgboost_results.json")

    rows = []

    for variant, r in logreg.items():
        tm = r["test_metrics"]
        rows.append({
            "model": "Logistic Regression", "kernel_or_variant": variant,
            "cv_score": r["cv_roc_auc"], "cv_metric": "roc_auc",
            "test_accuracy": tm["accuracy"], "test_precision": tm["precision"],
            "test_recall": tm["recall"], "test_f1": tm["f1"],
            "test_roc_auc": tm["roc_auc"], "test_pr_auc": tm["pr_auc"],
            "training_time_s": r["training_time_sec"], "best_params": str(r["best_params"]),
        })

    for variant, r in svm.items():
        tm = r["test_metrics"]
        rows.append({
            "model": "SVM", "kernel_or_variant": variant,
            "cv_score": r["cv_f1"], "cv_metric": "f1",
            "test_accuracy": tm["accuracy"], "test_precision": tm["precision"],
            "test_recall": tm["recall"], "test_f1": tm["f1"],
            "test_roc_auc": tm["roc_auc"], "test_pr_auc": tm["pr_auc"],
            "training_time_s": r["training_time_sec"], "best_params": str(r["best_params"]),
        })

    tm = rf["test_metrics"]
    rows.append({
        "model": "Random Forest", "kernel_or_variant": "-",
        "cv_score": rf["cv_f1"], "cv_metric": "f1",
        "test_accuracy": tm["accuracy"], "test_precision": tm["precision"],
        "test_recall": tm["recall"], "test_f1": tm["f1"],
        "test_roc_auc": tm["roc_auc"], "test_pr_auc": tm["pr_auc"],
        "training_time_s": rf["training_time_sec"], "best_params": str(rf["best_params"]),
    })

    tm = xgb["test_metrics"]
    rows.append({
        "model": "XGBoost", "kernel_or_variant": "-",
        "cv_score": xgb["cv_f1"], "cv_metric": "f1",
        "test_accuracy": tm["accuracy"], "test_precision": tm["precision"],
        "test_recall": tm["recall"], "test_f1": tm["f1"],
        "test_roc_auc": tm["roc_auc"], "test_pr_auc": tm["pr_auc"],
        "training_time_s": xgb["training_time_sec"], "best_params": str(xgb["best_params"]),
    })

    df = pd.DataFrame(rows)
    df = df.sort_values("test_f1", ascending=False).reset_index(drop=True)

    print("=== Full Model Comparison (sorted by test F1) ===\n")
    print(df[["model", "kernel_or_variant", "test_f1", "test_recall", "test_roc_auc", "training_time_s"]].to_string(index=False))

    best_by_f1 = df.iloc[0]
    print(f"\nBest by F1: {best_by_f1['model']} ({best_by_f1['kernel_or_variant']}) — F1={best_by_f1['test_f1']:.4f}")

    best_by_recall = df.loc[df["test_recall"].idxmax()]
    print(f"Best by Recall: {best_by_recall['model']} ({best_by_recall['kernel_or_variant']}) — Recall={best_by_recall['test_recall']:.4f}")

    print(f"\n=== Model selection rationale ===")
    print(f"For an at-risk early-warning system, recall matters more than accuracy: "
          f"a false negative (missed at-risk student) is costlier than a false positive "
          f"(unnecessary attention). {best_by_recall['model']} ({best_by_recall['kernel_or_variant']}) "
          f"catches the most at-risk students ({best_by_recall['test_recall']*100:.0f}%), "
          f"while {best_by_f1['model']} ({best_by_f1['kernel_or_variant']}) offers the best "
          f"overall balance (F1={best_by_f1['test_f1']:.4f}).")

    df.to_csv(f"{RESULTS_DIR}/model_comparison.csv", index=False)
    print(f"\nSaved to {RESULTS_DIR}/model_comparison.csv")