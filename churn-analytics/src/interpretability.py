"""
Phase 6: Model interpretability.

Permutation importance (model-agnostic), native model importance
(coefficients for linear models, feature_importances_ for tree-based),
and SHAP values for both global and individual-customer explanations.

Feature importance is descriptive of the model's learned associations,
NOT causal evidence — stated explicitly per spec.
"""
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import json

RESULTS_DIR = "results"


def get_feature_names(pipeline):
    """Extracts the actual post-transformation feature names from the
    fitted ColumnTransformer, so importance scores map to real columns,
    not opaque indices."""
    preprocessor = pipeline.named_steps["preprocessor"]
    return preprocessor.get_feature_names_out()


def compute_permutation_importance(pipeline, X_test, y_test, feature_names):
    result = permutation_importance(
        pipeline, X_test, y_test, n_repeats=10, random_state=42, scoring="roc_auc"
    )

    # permutation_importance operates on raw X_test (pre-pipeline), so the
    # "feature_names" here are the ORIGINAL columns, not post-encoding ones
    original_features = X_test.columns.tolist()

    importance_df = pd.DataFrame({
        "feature": original_features,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)

    return importance_df


def compute_native_importance(pipeline, feature_names):
    clf = pipeline.named_steps["clf"]

    if hasattr(clf, "coef_"):
        importance = clf.coef_[0]
        importance_type = "coefficient"
    elif hasattr(clf, "feature_importances_"):
        importance = clf.feature_importances_
        importance_type = "feature_importance"
    else:
        return None, None

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", key=abs, ascending=False)

    return importance_df, importance_type


def compute_shap_values(pipeline, X_test, feature_names, sample_size=200):
    preprocessor = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["clf"]

    X_test_transformed = preprocessor.transform(X_test)
    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()

    # Use a sample for SHAP (full dataset can be slow, especially for
    # KernelExplainer fallback) — 200 is enough for stable global patterns
    sample_idx = np.random.RandomState(42).choice(
        len(X_test_transformed), min(sample_size, len(X_test_transformed)), replace=False
    )
    X_sample = X_test_transformed[sample_idx]

    if hasattr(clf, "coef_"):
        explainer = shap.LinearExplainer(clf, X_sample)
    elif hasattr(clf, "feature_importances_"):
        explainer = shap.TreeExplainer(clf)
    else:
        explainer = shap.KernelExplainer(clf.predict_proba, X_sample[:50])

    shap_values = explainer.shap_values(X_sample)

    # Handle different SHAP output shapes across explainer types
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # class 1 (churn) for classifiers returning per-class arrays
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    return shap_values, X_sample, sample_idx


def plot_shap_summary(shap_values, X_sample, feature_names, save_path):
    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def explain_individual_customer(shap_values, X_sample, feature_names, sample_idx, X_test, customer_idx=0):
    """Picks one customer and shows their top contributing features."""
    shap_row = shap_values[customer_idx]
    top_indices = np.argsort(np.abs(shap_row))[::-1][:5]

    explanation = []
    for idx in top_indices:
        explanation.append({
            "feature": feature_names[idx],
            "shap_value": float(shap_row[idx]),
            "direction": "increases churn risk" if shap_row[idx] > 0 else "decreases churn risk",
        })
    return explanation


if __name__ == "__main__":
    pipeline = joblib.load(f"{RESULTS_DIR}/best_model.pkl")
    X_test = pd.read_csv(f"{RESULTS_DIR}/X_test.csv")
    y_test = pd.read_csv(f"{RESULTS_DIR}/y_test.csv").squeeze()

    feature_names = get_feature_names(pipeline)
    print(f"Total features after preprocessing: {len(feature_names)}")

    print("\n=== Permutation Importance (top 10, original features) ===")
    perm_importance_df = compute_permutation_importance(pipeline, X_test, y_test, feature_names)
    print(perm_importance_df.head(10).to_string(index=False))

    print("\n=== Native Model Importance (top 10, post-encoding features) ===")
    native_importance_df, importance_type = compute_native_importance(pipeline, feature_names)
    if native_importance_df is not None:
        print(f"(type: {importance_type})")
        print(native_importance_df.head(10).to_string(index=False))

    print("\n=== Computing SHAP values (sample of 200 test customers) ===")
    shap_values, X_sample, sample_idx = compute_shap_values(pipeline, X_test, feature_names)
    print(f"SHAP values computed: shape {shap_values.shape}")

    plot_shap_summary(shap_values, X_sample, feature_names, f"{RESULTS_DIR}/shap_summary.png")
    print("Saved shap_summary.png")

    print("\n=== Individual Customer Explanation (first test customer) ===")
    individual_explanation = explain_individual_customer(
        shap_values, X_sample, feature_names, sample_idx, X_test, customer_idx=0
    )
    for item in individual_explanation:
        print(f"  {item['feature']}: SHAP={item['shap_value']:.4f} ({item['direction']})")

    with open(f"{RESULTS_DIR}/interpretability_results.json", "w") as f:
        json.dump({
            "permutation_importance_top10": perm_importance_df.head(10).to_dict("records"),
            "native_importance_type": importance_type,
            "native_importance_top10": native_importance_df.head(10).to_dict("records") if native_importance_df is not None else None,
            "example_individual_explanation": individual_explanation,
        }, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}/interpretability_results.json")