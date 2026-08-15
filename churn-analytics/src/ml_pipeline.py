"""
Phase 5: ML pipeline for churn prediction.

Logistic Regression, Random Forest, XGBoost — with a proper preprocessing
pipeline (fit only on train, applied to test) to avoid leakage, stratified
5-fold CV, and light hyperparameter tuning via GridSearchCV.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)
from xgboost import XGBClassifier
import json
import joblib

ENGINEERED_PATH = "data/telco_churn_engineered.csv"
RESULTS_DIR = "results"
RANDOM_SEED = 42


def prepare_data(df):
    # Drop customerID (identifier, not predictive — documented per spec)
    # Target: Churn (Yes/No -> 1/0)
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["customerID", "Churn"])

    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    # bool columns from feature engineering need explicit handling
    bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category" , "string"]).columns.tolist()

    # Convert bools to int (0/1) - simpler than one-hot for binary flags
    for col in bool_cols:
        X[col] = X[col].astype(int)
    numerical_cols += bool_cols

    return X, y, numerical_cols, categorical_cols


def build_preprocessor(numerical_cols, categorical_cols):
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_cols),
    ])


def get_models_and_grids():
    return {
        "LogisticRegression": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
            {"clf__C": [0.1, 1.0, 10.0]},
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=RANDOM_SEED),
            {"clf__n_estimators": [100, 200], "clf__max_depth": [10, 20, None]},
        ),
        "XGBoost": (
            XGBClassifier(random_state=RANDOM_SEED, eval_metric="logloss"),
            {"clf__n_estimators": [100, 200], "clf__max_depth": [3, 6]},
        ),
    }


def evaluate_model(y_true, y_pred, y_proba):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


if __name__ == "__main__":
    df = pd.read_csv(ENGINEERED_PATH)
    X, y, numerical_cols, categorical_cols = prepare_data(df)

    print(f"Features: {len(numerical_cols)} numerical, {len(categorical_cols)} categorical")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    print(f"\nTrain set: {len(X_train)}, Test set: {len(X_test)}")

    preprocessor = build_preprocessor(numerical_cols, categorical_cols)
    models_and_grids = get_models_and_grids()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    all_results = {}
    fitted_pipelines = {}

    for name, (model, param_grid) in models_and_grids.items():
        print(f"\n=== Training {name} ===")

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("clf", model),
        ])

        grid_search = GridSearchCV(
            pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        best_pipe = grid_search.best_estimator_
        fitted_pipelines[name] = best_pipe

        print(f"Best params: {grid_search.best_params_}")
        print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

        # Evaluate on held-out test set
        y_pred = best_pipe.predict(X_test)
        y_proba = best_pipe.predict_proba(X_test)[:, 1]
        test_metrics = evaluate_model(y_test, y_pred, y_proba)

        print(f"Test set metrics:")
        for k, v in test_metrics.items():
            if k != "confusion_matrix":
                print(f"  {k}: {v:.4f}")
        print(f"  confusion_matrix: {test_metrics['confusion_matrix']}")

        # Cross-validated score distribution (mean + std, not just point estimate)
        cv_scores = grid_search.cv_results_["mean_test_score"][grid_search.best_index_]
        cv_std = grid_search.cv_results_["std_test_score"][grid_search.best_index_]

        all_results[name] = {
            "best_params": grid_search.best_params_,
            "cv_roc_auc_mean": float(cv_scores),
            "cv_roc_auc_std": float(cv_std),
            "test_metrics": test_metrics,
        }

    print("\n\n=== Model Comparison Summary ===")
    for name, results in all_results.items():
        tm = results["test_metrics"]
        print(f"{name}: CV ROC-AUC={results['cv_roc_auc_mean']:.4f}±{results['cv_roc_auc_std']:.4f}, "
              f"Test ROC-AUC={tm['roc_auc']:.4f}, Test F1={tm['f1']:.4f}")

    # Save results
    with open(f"{RESULTS_DIR}/ml_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Save the best model (by test ROC-AUC) for later phases
    best_model_name = max(all_results, key=lambda k: all_results[k]["test_metrics"]["roc_auc"])
    joblib.dump(fitted_pipelines[best_model_name], f"{RESULTS_DIR}/best_model.pkl")
    print(f"\nBest model by test ROC-AUC: {best_model_name} (saved to results/best_model.pkl)")

    # Save train/test split for reuse in later phases (explainability, threshold tuning)
    X_test.to_csv(f"{RESULTS_DIR}/X_test.csv", index=False)
    y_test.to_csv(f"{RESULTS_DIR}/y_test.csv", index=False)