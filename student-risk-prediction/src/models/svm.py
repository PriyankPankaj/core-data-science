"""
Phase 7-9: SVM models (Linear and RBF) with hyperparameter tuning.

SVM is the central model of this project. StandardScaler lives inside
the Pipeline, never applied before train/test split.
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
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


def train_svm(X_train, X_test, y_train, y_test, preprocessor, cv, kernel, param_grid, label):
    print(f"\n=== SVM ({label}) ===")

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", SVC(kernel=kernel, probability=True, class_weight="balanced", random_state=RANDOM_SEED)),
    ])

    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1", n_jobs=-1)

    start = time.time()
    grid.fit(X_train, y_train)
    train_time = time.time() - start

    best_pipe = grid.best_estimator_

    start = time.time()
    y_pred = best_pipe.predict(X_test)
    y_proba = best_pipe.predict_proba(X_test)[:, 1]
    inference_time = time.time() - start

    metrics = evaluate(y_test, y_pred, y_proba)

    print(f"Best params: {grid.best_params_}")
    print(f"CV F1: {grid.best_score_:.4f}")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v:.4f}")
    print(f"  confusion_matrix: {metrics['confusion_matrix']}")
    print(f"  training_time: {train_time:.2f}s, inference_time: {inference_time:.4f}s")

    return {
        "best_params": grid.best_params_,
        "cv_f1": float(grid.best_score_),
        "test_metrics": metrics,
        "training_time_sec": float(train_time),
        "inference_time_sec": float(inference_time),
    }, best_pipe


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    X, y, numerical_cols, categorical_cols = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    preprocessor = build_preprocessor(numerical_cols, categorical_cols)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    results = {}
    pipelines = {}

    # Linear SVM
    linear_grid = {"clf__C": [0.01, 0.1, 1, 10, 100]}
    results["SVM_Linear"], pipelines["SVM_Linear"] = train_svm(
        X_train, X_test, y_train, y_test, preprocessor, cv, "linear", linear_grid, "Linear"
    )

    # RBF SVM
    rbf_grid = {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.001, 0.01, 0.1, 1]}
    results["SVM_RBF"], pipelines["SVM_RBF"] = train_svm(
        X_train, X_test, y_train, y_test, preprocessor, cv, "rbf", rbf_grid, "RBF"
    )

    print("\n=== Linear vs RBF Comparison ===")
    for name, r in results.items():
        tm = r["test_metrics"]
        print(f"{name}: F1={tm['f1']:.4f}, Recall={tm['recall']:.4f}, "
              f"ROC-AUC={tm['roc_auc']:.4f}, train_time={r['training_time_sec']:.2f}s")

    best_svm = max(results, key=lambda k: results[k]["test_metrics"]["f1"])
    print(f"\nBest SVM kernel by F1: {best_svm}")

    joblib.dump(pipelines[best_svm], f"{RESULTS_DIR}/best_svm.pkl")

    with open(f"{RESULTS_DIR}/svm_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {RESULTS_DIR}/svm_results.json")