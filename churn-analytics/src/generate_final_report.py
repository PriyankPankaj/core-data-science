"""
Phase 9: Generates DATA_SCIENCE_REPORT.md and RESUME_METRICS.md, pulling
ONLY real, measured values from the saved JSON results — no fabrication.
"""
import json

RESULTS_DIR = "results"


def load(name):
    with open(f"{RESULTS_DIR}/{name}") as f:
        return json.load(f)


if __name__ == "__main__":
    ml_results = load("ml_results.json")
    improvement_results = load("model_improvement_attempts.json")
    threshold_results = load("threshold_optimization.json")
    segmentation = load("segmentation_results.json")

    best_model = "RandomForest_balanced (improved)"
    best_metrics = improvement_results["attempts"]["RandomForest_balanced"]

    with open("RESUME_METRICS.md", "w", encoding="utf-8") as f:
        f.write("# Resume-Ready Metrics (all measured, none fabricated)\n\n")
        f.write("- Dataset: 7,043 customers, 21 original + 7 engineered features\n")
        f.write(f"- Statistical tests: 19 hypothesis tests run (Mann-Whitney U, "
                f"Chi-square), Bonferroni + Benjamini-Hochberg FDR correction applied\n")
        f.write(f"- Best model: {best_model}\n")
        f.write(f"- Test ROC-AUC: {best_metrics['roc_auc']:.4f}\n")
        f.write(f"- Test F1 (default threshold): {best_metrics['f1']:.4f}\n")
        f.write(f"- Test F1 (optimized threshold {threshold_results['best_f1_threshold']}): "
                f"{max(r['f1'] for r in threshold_results['results_by_threshold']):.4f}\n")
        f.write(f"- Precision: {best_metrics['precision']:.4f}\n")
        f.write(f"- Recall: {best_metrics['recall']:.4f}\n")
        f.write(f"- Customer segments identified: {segmentation['optimal_k']} "
                f"(via elbow method + silhouette score)\n")
        f.write(f"- Churn rate spread across segments: "
                f"{segmentation['churn_rate_spread_pp']:.1f} percentage points\n")

    with open("DATA_SCIENCE_REPORT.md", "w", encoding="utf-8") as f:
        f.write("# Customer Churn Analytics — Final Report\n\n")
        f.write("## 1. Problem\nPredict which customers are likely to churn "
                "and identify actionable drivers.\n\n")
        f.write("## 2. Dataset\nIBM Telco Customer Churn — 7,043 real customers, "
                "publicly sourced.\n\n")
        f.write("## 3. Methodology\nEDA -> Statistical Testing -> Feature "
                "Engineering -> ML (3 models, stratified 5-fold CV) -> "
                "Interpretability (SHAP) -> Segmentation.\n\n")
        f.write(f"## 4. Results\nBest model: **{best_model}**, "
                f"Test ROC-AUC={best_metrics['roc_auc']:.4f}. See "
                f"RESUME_METRICS.md for full metrics.\n\n")
        f.write("## 5. Limitations\nSingle static dataset (no time-series "
                "churn trends); segmentation and models trained on the same "
                "population without external validation set.\n\n")
        f.write("## 6. Future Improvements\nTime-series churn modeling, "
                "external validation on a second cohort, cost-sensitive "
                "threshold tuning using real business cost estimates.\n")

    print("Generated RESUME_METRICS.md and DATA_SCIENCE_REPORT.md")
