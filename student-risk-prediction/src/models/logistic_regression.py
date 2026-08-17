"""
Phase 6: Baseline Logistic Regression.

Establishes an interpretable baseline before the SVM-focused phases.
Preprocessing lives entirely inside the sklearn Pipeline (fit only on
train, never on the full dataset before splitting) to avoid leakage.
class_weight="balanced" is tested, not assumed to help.
"""
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)
import json
import joblib

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


def build_preprocessor(numerical_cols, categorical_cols):
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_cols),
    ])


def evaluate(y_true, y_pred, y_proba):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    X, y, numerical_cols, categorical_cols = prepare_data(df)

    print(f"Features: {len(numerical_cols)} numerical, {len(categorical_cols)} categorical")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    preprocessor = build_preprocessor(numerical_cols, categorical_cols)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    results = {}
    fitted_pipelines = {}

    for weight_setting in [None, "balanced"]:
        label = "unweighted" if weight_setting is None else "balanced"
        print(f"\n=== Logistic Regression ({label}) ===")

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("clf", LogisticRegression(max_iter=2000, class_weight=weight_setting, random_state=RANDOM_SEED)),
        ])

        param_grid = {"clf__C": [0.01, 0.1, 1.0, 10.0, 100.0]}
        grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)

        start = time.time()
        grid.fit(X_train, y_train)
        train_time = time.time() - start

        best_pipe = grid.best_estimator_
        fitted_pipelines[label] = best_pipe

        start = time.time()
        y_pred = best_pipe.predict(X_test)
        y_proba = best_pipe.predict_proba(X_test)[:, 1]
        inference_time = time.time() - start

        metrics = evaluate(y_test, y_pred, y_proba)

        print(f"Best C: {grid.best_params_['clf__C']}")
        print(f"CV ROC-AUC: {grid.best_score_:.4f}")
        for k, v in metrics.items():
            if k != "confusion_matrix":
                print(f"  {k}: {v:.4f}")
        print(f"  confusion_matrix: {metrics['confusion_matrix']}")
        print(f"  training_time: {train_time:.2f}s, inference_time: {inference_time:.4f}s")

        results[f"LogisticRegression_{label}"] = {
            "best_params": grid.best_params_,
            "cv_roc_auc": float(grid.best_score_),
            "test_metrics": metrics,
            "training_time_sec": float(train_time),
            "inference_time_sec": float(inference_time),
        }

    best_name = max(results, key=lambda k: results[k]["test_metrics"]["f1"])
    print(f"\nBest LogReg variant by test ROC-AUC: {best_name}")

    joblib.dump(fitted_pipelines[best_name.replace("LogisticRegression_", "")], f"{RESULTS_DIR}/baseline_logreg.pkl")
    X_test.to_csv(f"{RESULTS_DIR}/X_test.csv", index=False)
    y_test.to_csv(f"{RESULTS_DIR}/y_test.csv", index=False)

    with open(f"{RESULTS_DIR}/logistic_regression_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {RESULTS_DIR}/logistic_regression_results.json")