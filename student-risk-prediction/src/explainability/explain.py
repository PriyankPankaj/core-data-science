"""
Phase 14: Model interpretability.

Permutation importance + SHAP for Random Forest (our best-F1 model).
Linear SVM coefficient analysis (per spec's explicit requirement for
linear SVM interpretability). Never framed as causal — descriptive of
learned associations only.
"""
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import json

DATA_PATH = "data/student_performance_engineered.csv"
RESULTS_DIR = "results"
RANDOM_SEED = 42
LEAKAGE_COLUMNS = ["G1", "G2", "G3"]


def prepare_data(df):
    y = df["at_risk"]
    X = df.drop(columns=LEAKAGE_COLUMNS + ["at_risk"])
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    return X, y, numerical_cols, categorical_cols


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    X, y, numerical_cols, categorical_cols = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    # === Random Forest: permutation importance + SHAP ===
    rf_pipeline = joblib.load(f"{RESULTS_DIR}/random_forest.pkl")

    print("=== Random Forest: Permutation Importance (top 10, original features) ===")
    perm_result = permutation_importance(
        rf_pipeline, X_test, y_test, n_repeats=10, random_state=RANDOM_SEED, scoring="f1"
    )
    perm_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": perm_result.importances_mean,
        "importance_std": perm_result.importances_std,
    }).sort_values("importance_mean", ascending=False)
    print(perm_df.head(10).to_string(index=False))

    print("\n=== Random Forest: SHAP ===")
    preprocessor = rf_pipeline.named_steps["preprocessor"]
    rf_clf = rf_pipeline.named_steps["clf"]
    X_test_transformed = preprocessor.transform(X_test)
    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(rf_clf)
    shap_values = explainer.shap_values(X_test_transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    plt.figure()
    shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/shap_summary_rf.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved shap_summary_rf.png")

    # === Linear SVM: coefficient analysis (spec-required) ===
    print("\n=== Linear SVM: Coefficient Analysis ===")
    preprocessor2 = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_cols),
    ])
    X_train_t = preprocessor2.fit_transform(X_train)
    linear_svm = SVC(kernel="linear", C=0.01, class_weight="balanced", random_state=RANDOM_SEED)
    linear_svm.fit(X_train_t, y_train)

    feature_names2 = preprocessor2.get_feature_names_out()
    coef_df = pd.DataFrame({
        "feature": feature_names2,
        "coefficient": linear_svm.coef_[0],
    }).sort_values("coefficient", key=abs, ascending=False)
    print(coef_df.head(10).to_string(index=False))

    print("\n=== IMPORTANT: Interpretation Caveat ===")
    print("These results describe LEARNED ASSOCIATIONS the models found in "
          "observational data, not causal relationships. A feature ranking "
          "highly does not mean it causes academic risk — only that it is "
          "predictive within this model given this dataset.")

    with open(f"{RESULTS_DIR}/explainability_results.json", "w") as f:
        json.dump({
            "rf_permutation_importance_top10": perm_df.head(10).to_dict("records"),
            "linear_svm_coefficients_top10": coef_df.head(10).to_dict("records"),
            "causal_interpretation_caveat": (
                "These are learned associations, not causal claims. "
                "No feature can be said to 'cause' academic risk based on "
                "this observational analysis alone."
            ),
        }, f, indent=2)

    print(f"\nSaved to {RESULTS_DIR}/explainability_results.json")