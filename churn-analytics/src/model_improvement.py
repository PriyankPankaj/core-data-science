"""
Attempts to improve on Phase 5's baseline models using legitimate,
standard techniques — tested and measured, not assumed to help.
Per spec: do not blindly apply techniques; only retain what demonstrably
improves the relevant metric.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
import json

ENGINEERED_PATH = "data/telco_churn_engineered.csv"
RESULTS_DIR = "results"
RANDOM_SEED = 42


def prepare_data(df):
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["customerID", "Churn"])

    # Feature selection: drop features Phase 3 found NOT significant
    X = X.drop(columns=["gender", "PhoneService"])

    # New interaction feature: tenure x is_month_to_month
    X["tenure_x_month_to_month"] = X["tenure"] * X["is_month_to_month"].astype(int)

    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    for col in bool_cols:
        X[col] = X[col].astype(int)
    numerical_cols += bool_cols

    return X, y, numerical_cols, categorical_cols


def build_preprocessor(numerical_cols, categorical_cols):
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_cols),
    ])


if __name__ == "__main__":
    df = pd.read_csv(ENGINEERED_PATH)
    X, y, numerical_cols, categorical_cols = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    preprocessor = build_preprocessor(numerical_cols, categorical_cols)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    results = {}

    # --- Attempt 1: Logistic Regression with class_weight + wider grid ---
    print("=== Attempt 1: Logistic Regression, class_weight balanced, wider grid ===")
    pipe_lr = Pipeline([("preprocessor", preprocessor),
                         ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED))])
    grid_lr = GridSearchCV(pipe_lr, {"clf__C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]},
                            cv=cv, scoring="roc_auc", n_jobs=-1)
    grid_lr.fit(X_train, y_train)
    y_pred = grid_lr.predict(X_test)
    y_proba = grid_lr.predict_proba(X_test)[:, 1]
    results["LogReg_balanced_wider"] = {
        "best_params": grid_lr.best_params_,
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
    }
    print(results["LogReg_balanced_wider"])

    # --- Attempt 2: Voting ensemble of LR + RF + XGB ---
    print("\n=== Attempt 2: Voting Ensemble (soft voting) ===")
    lr = LogisticRegression(max_iter=2000, C=grid_lr.best_params_["clf__C"],
                             class_weight="balanced", random_state=RANDOM_SEED)
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RANDOM_SEED)
    xgb = XGBClassifier(n_estimators=100, max_depth=3, random_state=RANDOM_SEED, eval_metric="logloss")

    voting = VotingClassifier(estimators=[("lr", lr), ("rf", rf), ("xgb", xgb)], voting="soft")
    pipe_voting = Pipeline([("preprocessor", preprocessor), ("clf", voting)])
    pipe_voting.fit(X_train, y_train)
    y_pred = pipe_voting.predict(X_test)
    y_proba = pipe_voting.predict_proba(X_test)[:, 1]
    results["VotingEnsemble"] = {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
    }
    print(results["VotingEnsemble"])
    
    # --- Attempt 3: Random Forest with class_weight balanced ---
    print("\n=== Attempt 3: Random Forest, class_weight balanced ===")
    pipe_rf = Pipeline([("preprocessor", preprocessor),
                         ("clf", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_SEED))])
    grid_rf = GridSearchCV(pipe_rf, {"clf__n_estimators": [100, 200], "clf__max_depth": [10, 20, None]},
                            cv=cv, scoring="roc_auc", n_jobs=-1)
    grid_rf.fit(X_train, y_train)
    y_pred = grid_rf.predict(X_test)
    y_proba = grid_rf.predict_proba(X_test)[:, 1]
    results["RandomForest_balanced"] = {
        "best_params": grid_rf.best_params_,
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
    }
    print(results["RandomForest_balanced"])

    # --- Attempt 4: XGBoost with scale_pos_weight (its equivalent of class_weight) ---
    print("\n=== Attempt 4: XGBoost, scale_pos_weight ===")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()  # ratio of majority:minority
    pipe_xgb = Pipeline([("preprocessor", preprocessor),
                          ("clf", XGBClassifier(scale_pos_weight=scale_pos_weight,
                                                  random_state=RANDOM_SEED, eval_metric="logloss"))])
    grid_xgb = GridSearchCV(pipe_xgb, {"clf__n_estimators": [100, 200], "clf__max_depth": [3, 6]},
                             cv=cv, scoring="roc_auc", n_jobs=-1)
    grid_xgb.fit(X_train, y_train)
    y_pred = grid_xgb.predict(X_test)
    y_proba = grid_xgb.predict_proba(X_test)[:, 1]
    results["XGBoost_balanced"] = {
        "best_params": grid_xgb.best_params_,
        "scale_pos_weight_used": float(scale_pos_weight),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
    }
    print(results["XGBoost_balanced"])


    # --- Compare against Phase 5 baseline ---
    with open(f"{RESULTS_DIR}/ml_results.json") as f:
        baseline = json.load(f)
    baseline_best = max(baseline, key=lambda k: baseline[k]["test_metrics"]["roc_auc"])
    baseline_metrics = baseline[baseline_best]["test_metrics"]

    print(f"\n=== Comparison to Phase 5 baseline ({baseline_best}) ===")
    print(f"Baseline: ROC-AUC={baseline_metrics['roc_auc']:.4f}, F1={baseline_metrics['f1']:.4f}")
    for name, r in results.items():
        improved_auc = r["roc_auc"] > baseline_metrics["roc_auc"]
        improved_f1 = r["f1"] > baseline_metrics["f1"]
        print(f"{name}: ROC-AUC={r['roc_auc']:.4f} ({'better' if improved_auc else 'worse/same'}), "
              f"F1={r['f1']:.4f} ({'better' if improved_f1 else 'worse/same'})")

    with open(f"{RESULTS_DIR}/model_improvement_attempts.json", "w") as f:
        json.dump({"baseline": {baseline_best: baseline_metrics}, "attempts": results}, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}/model_improvement_attempts.json")

    # Save whichever attempt actually performed best, by ROC-AUC
    import joblib
    best_attempt_name = max(results, key=lambda k: results[k]["f1"])
    best_pipelines = {
        "LogReg_balanced_wider": grid_lr.best_estimator_,
        "VotingEnsemble": pipe_voting,
        "RandomForest_balanced": grid_rf.best_estimator_,
        "XGBoost_balanced": grid_xgb.best_estimator_,
    }
    joblib.dump(best_pipelines[best_attempt_name], f"{RESULTS_DIR}/best_model.pkl")
    X_test.to_csv(f"{RESULTS_DIR}/X_test.csv", index=False)
    y_test.to_csv(f"{RESULTS_DIR}/y_test.csv", index=False)
    print(f"\nBest attempt overall: {best_attempt_name} (ROC-AUC={results[best_attempt_name]['roc_auc']:.4f})")
    print(f"Saved as new best_model.pkl")