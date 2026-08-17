"""
Phase 12: Feature selection experiments.

Compares: all features, statistical selection (F-test), Mutual
Information, RFE, and SVM-RFE. Feature reduction is only kept if it
demonstrably helps ROC-AUC/F1/recall/training time — not removed just
to look better.
"""
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, recall_score
import json

DATA_PATH = "data/student_performance_engineered.csv"
RESULTS_DIR = "results"
RANDOM_SEED = 42
LEAKAGE_COLUMNS = ["G1", "G2", "G3"]
N_FEATURES_TO_SELECT = 15


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


def evaluate_feature_set(X_train_t, X_test_t, y_train, y_test, cv, label):
    clf = SVC(kernel="rbf", C=1, gamma=0.01, probability=True, class_weight="balanced", random_state=RANDOM_SEED)

    start = time.time()
    cv_scores = cross_val_score(clf, X_train_t, y_train, cv=cv, scoring="f1")
    clf.fit(X_train_t, y_train)
    train_time = time.time() - start

    y_pred = clf.predict(X_test_t)
    y_proba = clf.predict_proba(X_test_t)[:, 1]

    result = {
        "feature_set": label,
        "n_features": X_train_t.shape[1],
        "cv_f1_mean": float(cv_scores.mean()),
        "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
        "training_time_sec": float(train_time),
    }
    print(f"{label}: n_features={result['n_features']}, F1={result['test_f1']:.4f}, "
          f"Recall={result['test_recall']:.4f}, ROC-AUC={result['test_roc_auc']:.4f}, "
          f"train_time={result['training_time_sec']:.3f}s")
    return result


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    X, y, numerical_cols, categorical_cols = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    preprocessor = build_preprocessor(numerical_cols, categorical_cols)
    X_train_full = preprocessor.fit_transform(X_train)
    X_test_full = preprocessor.transform(X_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    results = []

    results.append(evaluate_feature_set(X_train_full, X_test_full, y_train, y_test, cv, "All Features"))

    selector_stat = SelectKBest(f_classif, k=N_FEATURES_TO_SELECT)
    X_train_stat = selector_stat.fit_transform(X_train_full, y_train)
    X_test_stat = selector_stat.transform(X_test_full)
    results.append(evaluate_feature_set(X_train_stat, X_test_stat, y_train, y_test, cv, "Statistical (F-test)"))

    selector_mi = SelectKBest(mutual_info_classif, k=N_FEATURES_TO_SELECT)
    X_train_mi = selector_mi.fit_transform(X_train_full, y_train)
    X_test_mi = selector_mi.transform(X_test_full)
    results.append(evaluate_feature_set(X_train_mi, X_test_mi, y_train, y_test, cv, "Mutual Information"))

    rfe = RFE(LogisticRegression(max_iter=2000, random_state=RANDOM_SEED), n_features_to_select=N_FEATURES_TO_SELECT)
    X_train_rfe = rfe.fit_transform(X_train_full, y_train)
    X_test_rfe = rfe.transform(X_test_full)
    results.append(evaluate_feature_set(X_train_rfe, X_test_rfe, y_train, y_test, cv, "RFE (LogReg)"))

    svm_rfe = RFE(SVC(kernel="linear", random_state=RANDOM_SEED), n_features_to_select=N_FEATURES_TO_SELECT)
    X_train_svmrfe = svm_rfe.fit_transform(X_train_full, y_train)
    X_test_svmrfe = svm_rfe.transform(X_test_full)
    results.append(evaluate_feature_set(X_train_svmrfe, X_test_svmrfe, y_train, y_test, cv, "SVM-RFE"))

    print(f"\n=== Feature Selection Summary (full feature count: {X_train_full.shape[1]}) ===")
    best = max(results, key=lambda r: r["test_f1"])
    print(f"Best by F1: {best['feature_set']} (F1={best['test_f1']:.4f}, n_features={best['n_features']})")

    baseline_f1 = results[0]["test_f1"]
    improved = best["test_f1"] > baseline_f1 and best["feature_set"] != "All Features"
    print(f"Feature reduction improves F1 over all-features baseline: {improved}")

    pd.DataFrame(results).to_csv(f"{RESULTS_DIR}/feature_selection_results.csv", index=False)

    with open(f"{RESULTS_DIR}/feature_selection_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {RESULTS_DIR}/feature_selection_results.csv and .json")